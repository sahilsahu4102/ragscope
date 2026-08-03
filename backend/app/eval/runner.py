"""
RAGScope — Eval Runner

Orchestrates a full evaluation run:
1. Load dataset (golden QA pairs)
2. For each sample: retrieve → generate → compute metrics
3. Aggregate metrics across all samples
4. Store results with config snapshot

The eval runner is the heart of the differentiator — it ties
the entire retrieval + generation pipeline to measurable quality.
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.eval.llm_judge import LLMJudge
from app.eval.metrics.generation_metrics import compute_all_generation_metrics
from app.eval.metrics.retrieval_metrics import compute_all_retrieval_metrics
from app.models import Dataset, EvalRun, EvalSample
from app.observability.tracer import create_span, get_tracer
from app.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger()
tracer = get_tracer("eval")


class EvalRunner:
    """
    Runs a complete evaluation: dataset → retrieve → generate → metrics.

    Stores per-sample and aggregate results for dashboard visualization.
    """

    def __init__(
        self,
        db: AsyncSession,
        use_llm_judge: bool = True,
    ):
        self.db = db
        self.use_llm_judge = use_llm_judge
        self.judge = LLMJudge() if use_llm_judge else None

    async def run(
        self,
        dataset: Dataset,
        run_name: str | None = None,
        config_overrides: dict | None = None,
    ) -> EvalRun:
        """
        Execute a full evaluation run on a dataset.

        Args:
            dataset: The golden dataset to evaluate against
            run_name: Human-readable name for this run
            config_overrides: Override retrieval/generation config for A/B testing
        """
        config = config_overrides or {}
        run_config = {
            "embedding_model": settings.ollama_embedding_model,
            "chunk_size": settings.default_chunk_size,
            "chunk_overlap": settings.default_chunk_overlap,
            "retrieval_top_k": config.get("top_k", settings.retrieval_top_k),
            "rerank_top_k": config.get("rerank_top_k", settings.rerank_top_k),
            "rrf_k": config.get("rrf_k", settings.rrf_k),
            "use_reranker": config.get("use_reranker", True),
            "use_hybrid": config.get("use_hybrid", True),
            "query_transform": config.get("query_transform", "none"),
            # None => whatever settings.reranker_backend says. Recorded in the
            # snapshot so a run is reproducible from its own config.
            "reranker_backend": config.get("reranker_backend", settings.reranker_backend),
            "sparse_backend": config.get("sparse_backend", settings.sparse_backend),
            # Generation overrides so a config sweep can be evaluated for
            # quality, not just timed.
            "model": config.get("model", settings.ollama_model),
            "num_predict": config.get("num_predict", settings.generation_num_predict),
            "max_chunk_chars": config.get("max_chunk_chars", settings.generation_max_chunk_chars),
            "use_llm_judge": self.use_llm_judge,
        }

        with create_span(
            tracer,
            "eval_run",
            "CHAIN",
            {
                "eval.dataset_id": str(dataset.id),
                "eval.dataset_name": dataset.name,
                "eval.sample_count": dataset.sample_count,
            },
        ):
            # 1. Create the eval run record
            eval_run = EvalRun(
                dataset_id=dataset.id,
                name=run_name or f"eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
                status="running",
                config_snapshot=run_config,
                total_samples=dataset.sample_count,
            )
            self.db.add(eval_run)
            await self.db.flush()

            run_start = time.time()
            all_retrieval_metrics: list[dict] = []
            all_generation_metrics: list[dict] = []
            passed = 0
            failed = 0

            # 2. Process each sample
            samples_data = dataset.samples or []
            pipeline = RetrievalPipeline(self.db)

            for idx, sample in enumerate(samples_data):
                try:
                    sample_result = await self._evaluate_sample(
                        pipeline=pipeline,
                        eval_run_id=eval_run.id,
                        sample_index=idx,
                        question=sample["question"],
                        gold_answer=sample.get("gold_answer"),
                        gold_contexts=sample.get("gold_contexts", []),
                        config=run_config,
                    )

                    if sample_result.get("retrieval_metrics"):
                        all_retrieval_metrics.append(sample_result["retrieval_metrics"])
                    if sample_result.get("generation_metrics"):
                        all_generation_metrics.append(sample_result["generation_metrics"])

                    passed += 1

                except Exception as e:
                    logger.error(
                        "Sample evaluation failed",
                        sample_index=idx,
                        error=str(e),
                    )
                    failed += 1

                    # Store failed sample
                    failed_sample = EvalSample(
                        eval_run_id=eval_run.id,
                        sample_index=idx,
                        question=sample["question"],
                        gold_answer=sample.get("gold_answer"),
                        gold_contexts=sample.get("gold_contexts"),
                        metrics={"error": str(e)},
                    )
                    self.db.add(failed_sample)

            # 3. Aggregate metrics
            aggregate = self._aggregate_metrics(all_retrieval_metrics, all_generation_metrics)

            # 4. Update eval run
            eval_run.status = "completed"
            eval_run.metrics = aggregate
            eval_run.passed_samples = passed
            eval_run.failed_samples = failed
            eval_run.total_latency_ms = round((time.time() - run_start) * 1000, 2)
            eval_run.completed_at = datetime.now(UTC)

            await self.db.commit()
            await self.db.refresh(eval_run)

            logger.info(
                "Eval run completed",
                run_id=str(eval_run.id),
                passed=passed,
                failed=failed,
                metrics=aggregate,
            )

            return eval_run

    async def _evaluate_sample(
        self,
        pipeline: RetrievalPipeline,
        eval_run_id: uuid.UUID,
        sample_index: int,
        question: str,
        gold_answer: str | None,
        gold_contexts: list[str],
        config: dict,
    ) -> dict:
        """Evaluate a single QA sample through the full pipeline."""
        with create_span(
            tracer,
            "eval_sample",
            "CHAIN",
            {"eval.sample_index": sample_index},
        ):
            # 1. Retrieve
            retrieval_start = time.time()
            retrieved_chunks = await pipeline.retrieve(
                query=question,
                top_k=config.get("retrieval_top_k", settings.retrieval_top_k),
                use_hybrid=config.get("use_hybrid", True),
                use_reranker=config.get("use_reranker", True),
                query_transform=config.get("query_transform", "none"),
                rrf_k=config.get("rrf_k", settings.rrf_k),
                reranker_backend=config.get("reranker_backend"),
                sparse_backend=config.get("sparse_backend"),
            )
            retrieval_latency = round((time.time() - retrieval_start) * 1000, 2)

            # 2. Generate answer
            from app.generation.generator import Generator

            generator = Generator(
                model=config.get("model"),
                num_predict=config.get("num_predict"),
                max_chunk_chars=config.get("max_chunk_chars"),
            )
            generation_start = time.time()

            context_text = "\n\n".join(
                f"[{i + 1}] {c['content']}" for i, c in enumerate(retrieved_chunks)
            )
            gen_result = await generator.generate(question, retrieved_chunks)
            generated_answer = gen_result["answer"]
            generation_latency = round((time.time() - generation_start) * 1000, 2)

            # 3. Compute retrieval metrics
            retrieval_metrics = compute_all_retrieval_metrics(retrieved_chunks, gold_contexts, k=10)

            # 4. Compute generation metrics (heuristic)
            retrieved_texts = [c.get("content", "") for c in retrieved_chunks]
            generation_metrics = compute_all_generation_metrics(
                question, generated_answer, retrieved_texts, gold_answer
            )

            # 5. LLM Judge (if enabled)
            judge_reasoning = None
            if self.judge:
                try:
                    judge_result = await self.judge.judge_sample(
                        question=question,
                        context=context_text,
                        answer=generated_answer,
                        gold_answer=gold_answer,
                    )
                    generation_metrics.update(
                        {
                            "faithfulness_judge": judge_result.get("faithfulness_judge", 0),
                            "relevance_judge": judge_result.get("relevance_judge", 0),
                        }
                    )
                    judge_reasoning = judge_result.get("reasoning", "")
                except Exception as e:
                    logger.warning("LLM judge failed for sample", error=str(e))

            # 6. Store sample result
            all_metrics = {**retrieval_metrics, **generation_metrics}
            eval_sample = EvalSample(
                eval_run_id=eval_run_id,
                sample_index=sample_index,
                question=question,
                gold_answer=gold_answer,
                gold_contexts=gold_contexts,
                generated_answer=generated_answer,
                retrieved_chunks=[
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "content": c.get("content", "")[:200],
                        "score": c.get("dense_score", 0),
                    }
                    for c in retrieved_chunks[:10]
                ],
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=generation_latency,
                metrics=all_metrics,
                judge_reasoning=judge_reasoning,
            )
            self.db.add(eval_sample)

            return {
                "retrieval_metrics": retrieval_metrics,
                "generation_metrics": generation_metrics,
            }

    @staticmethod
    def _aggregate_metrics(
        retrieval_metrics: list[dict],
        generation_metrics: list[dict],
    ) -> dict:
        """Average metrics across all samples."""
        aggregate = {}

        # Average retrieval metrics
        if retrieval_metrics:
            all_keys = retrieval_metrics[0].keys()
            for key in all_keys:
                values = [m[key] for m in retrieval_metrics if key in m]
                if values:
                    aggregate[key] = round(sum(values) / len(values), 4)

        # Average generation metrics
        if generation_metrics:
            all_keys = generation_metrics[0].keys()
            for key in all_keys:
                values = [m[key] for m in generation_metrics if key in m]
                if values:
                    aggregate[key] = round(sum(values) / len(values), 4)

        return aggregate
