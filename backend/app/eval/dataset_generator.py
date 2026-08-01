"""
RAGScope — Dataset Generator

Generates synthetic golden QA sets from ingested documents.
Uses LLM to create diverse question types:
- Single-hop factual (direct answer in one chunk)
- Multi-hop reasoning (requires combining multiple chunks)
- Analytical (requires interpretation/synthesis)
"""

from __future__ import annotations

import json
import random
import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.http_client import get_http_client
from app.models import Chunk, Dataset, Document
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("eval")

QA_GENERATION_PROMPT = """You are a QA dataset generator for evaluating RAG systems.

Given the following document chunks, generate {num_questions} diverse question-answer pairs.

## Requirements
- Each question MUST be answerable using only the provided chunks
- Questions must be self-contained: a reader who cannot see the chunks should
  still understand what is being asked. Never write "this document", "the
  passage above", "according to the text", or "chunk 2".
- Include a mix of question types:
  * Factual (direct answer from one chunk)
  * Reasoning (requires combining info from multiple chunks)
  * Analytical (requires interpretation)
- Each answer should be 1-3 sentences, grounded strictly in the chunks
- source_chunks must list the indices actually used

## Chunks
{chunks}

## Output Format (one per line, raw JSON, no markdown fences)
{{"question": "...", "answer": "...", "type": "factual|reasoning|analytical", "source_chunks": [0, 1]}}

Generate exactly {num_questions} question-answer pairs, one JSON per line:"""


class DatasetGenerator:
    """Generates synthetic evaluation datasets from ingested documents."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url

    # Chunks per LLM call. The model runs a 4096-token context, so a batch of
    # 4 chunks (~2k tokens) leaves room for instructions and the generated
    # output. Sending the whole corpus in one prompt silently truncates it.
    CHUNKS_PER_BATCH = 4
    QUESTIONS_PER_BATCH = 2

    # Chunks shorter than this rarely yield a meaningful question (headers,
    # page numbers, table fragments).
    MIN_CHUNK_TOKENS = 60

    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama for QA generation via the shared connection pool."""
        client = get_http_client()
        response = await client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": settings.ollama_keep_alive_value,
                "options": {"temperature": 0.7, "num_predict": 1024},
            },
        )
        response.raise_for_status()
        return response.json().get("response", "")

    @staticmethod
    def _parse_qa_lines(text: str) -> list[dict]:
        """Parse one-JSON-per-line output, tolerating markdown fences."""
        parsed: list[dict] = []
        for raw in text.strip().split("\n"):
            line = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                qa = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not qa.get("question") or not qa.get("answer"):
                continue
            parsed.append(qa)
        return parsed

    async def generate(
        self,
        name: str,
        document_ids: list[str] | None = None,
        num_questions: int = 20,
        version: str = "1.0",
        seed: int = 42,
    ) -> Dataset:
        """
        Generate a golden evaluation dataset from ingested documents.

        Chunks are sampled across the whole corpus and fed to the LLM in
        context-sized batches, since the generation model's context window is
        far smaller than the corpus.

        Args:
            name: Human-readable dataset name
            document_ids: Specific docs to use (None = all)
            num_questions: Target number of QA pairs
            version: Dataset version string
            seed: RNG seed for chunk sampling — keeps regeneration reproducible
        """
        with create_span(
            tracer,
            "generate_dataset",
            "CHAIN",
            {
                "dataset.name": name,
                "dataset.target_questions": num_questions,
            },
        ):
            # ── 1. Load candidate chunks ──────────────
            # selectinload is required: accessing chunk.document lazily inside
            # an async session raises MissingGreenlet.
            query = (
                select(Chunk)
                .join(Document)
                .options(selectinload(Chunk.document))
                .where(Chunk.token_count >= self.MIN_CHUNK_TOKENS)
            )
            if document_ids:
                query = query.where(Document.id.in_([uuid.UUID(did) for did in document_ids]))

            result = await self.db.execute(query)
            all_chunks = list(result.scalars().all())

            if not all_chunks:
                raise ValueError("No chunks found for dataset generation")

            # ── 2. Sample across the whole corpus ─────
            # Ordering by chunk_index and taking the head would only ever cover
            # the opening pages of each document. Seeded for reproducibility.
            batches_needed = max(1, -(-num_questions // self.QUESTIONS_PER_BATCH))
            sample_size = min(len(all_chunks), batches_needed * self.CHUNKS_PER_BATCH)

            rng = random.Random(seed)
            sampled = rng.sample(all_chunks, sample_size)

            # ── 3. Generate in context-sized batches ──
            samples: list[dict] = []
            seen_questions: set[str] = set()

            for batch_start in range(0, len(sampled), self.CHUNKS_PER_BATCH):
                if len(samples) >= num_questions:
                    break

                batch = sampled[batch_start : batch_start + self.CHUNKS_PER_BATCH]
                chunks_str = "\n\n---\n\n".join(
                    f"[Chunk {i}] (from: {c.document.filename})\n{c.content}"
                    for i, c in enumerate(batch)
                )

                prompt = QA_GENERATION_PROMPT.format(
                    num_questions=self.QUESTIONS_PER_BATCH,
                    chunks=chunks_str,
                )

                try:
                    response = await self._call_llm(prompt)
                except Exception as e:
                    logger.warning("QA generation call failed", batch=batch_start, error=str(e))
                    continue

                for qa in self._parse_qa_lines(response):
                    question = str(qa["question"]).strip()
                    # Cheap dedup — the model repeats itself across batches.
                    key = question.lower()
                    if key in seen_questions:
                        continue
                    seen_questions.add(key)

                    # Indices are batch-local; map them back to real chunks.
                    cited = [
                        batch[idx]
                        for idx in qa.get("source_chunks", [])
                        if isinstance(idx, int) and 0 <= idx < len(batch)
                    ]
                    if not cited:
                        # No usable attribution -> unusable for context recall.
                        continue
                    gold_contexts = [c.content for c in cited]

                    samples.append(
                        {
                            "question": question,
                            "gold_answer": str(qa["answer"]).strip(),
                            "gold_contexts": gold_contexts,
                            "question_type": qa.get("type", "factual"),
                            "metadata": {
                                # Only the chunks actually cited — not every
                                # chunk that happened to be in the batch.
                                "source_chunk_ids": [str(c.id) for c in cited],
                                "source_documents": sorted(
                                    {c.document.filename for c in cited}
                                ),
                            },
                        }
                    )

                logger.info(
                    "QA batch complete",
                    batch=batch_start // self.CHUNKS_PER_BATCH + 1,
                    of=batches_needed,
                    collected=len(samples),
                )

            if not samples:
                raise ValueError("LLM failed to generate valid QA pairs")

            samples = samples[:num_questions]

            # ── 4. Store as Dataset ───────────────────
            source_doc_ids = list({str(c.document_id) for c in sampled})
            dataset = Dataset(
                name=name,
                version=version,
                description=(
                    f"Auto-generated from {len(source_doc_ids)} documents, {len(samples)} QA pairs"
                ),
                sample_count=len(samples),
                samples=samples,
                source_documents=source_doc_ids,
            )
            self.db.add(dataset)
            await self.db.commit()
            await self.db.refresh(dataset)

            logger.info(
                "Dataset generated",
                dataset_id=str(dataset.id),
                name=name,
                samples=len(samples),
            )

            return dataset
