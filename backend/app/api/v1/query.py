"""
RAGScope — Query API Router (Phase 5)

Endpoints for RAG question-answering with citations, SSE streaming,
semantic caching, configurable query transformation, and guardrails.

Phase 5: Integrated guardrails pipeline (PII redaction, injection
detection, hallucination scoring) into the query path.
"""

import json
import time
import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import trace as otel_trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.caching.semantic_cache import SemanticCache
from app.config import settings
from app.db.session import async_session, get_db
from app.generation.generator import Generator
from app.guardrails import GuardrailsPipeline
from app.guardrails.pii import StreamingRedactor
from app.models import Query as QueryModel
from app.observability.collector import persist_trace
from app.observability.tracer import create_span, get_current_trace_id, get_tracer
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.schemas import Citation, QueryRequest, QueryResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/query", tags=["query"])
tracer = get_tracer("query")

# Module-level instances (shared across requests)
_semantic_cache = SemanticCache()
_guardrails = GuardrailsPipeline(
    enable_pii=getattr(settings, "enable_pii_redaction", True),
    enable_injection=getattr(settings, "enable_injection_detection", True),
    enable_hallucination=getattr(settings, "enable_hallucination_detection", True),
)


@router.post("", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question — full RAG pipeline with semantic caching.

    Pipeline:
      1. Check semantic cache (if enabled)
      2. Query transformation (rewrite / HyDE / decompose)
      3. Retrieval (dense or hybrid + RRF + rerank)
      4. Generation (Ollama LLM + grounded prompt + citations)
      5. Cache result for future similar queries
    """
    start = time.perf_counter()
    query_id: str | None = None

    with create_span(
        tracer,
        "rag_query",
        "CHAIN",
        {
            "query.text": request.question,
            "query.use_cache": request.use_cache,
            "query.query_transform": request.query_transform,
        },
    ):
        # Adopt the live OTel trace id so persisted spans link to this response.
        trace_id = get_current_trace_id() or uuid.uuid4().hex[:16]
        otel_trace.get_current_span().set_attribute("query.trace_id", trace_id)

        # ── 0a. Guardrails — input validation ────
        input_check = _guardrails.check_input(request.question)
        if input_check["blocked"]:
            raise HTTPException(
                status_code=400,
                detail=input_check["reason"],
            )
        # Use PII-redacted query for pipeline
        safe_question = input_check["redacted_query"]

        # ── 0b. Semantic cache check ──────────────
        if request.use_cache:
            cached = await _semantic_cache.get(safe_question)
            if cached:
                latency_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "Cache HIT — skipping pipeline",
                    trace_id=trace_id,
                    similarity=cached.get("cache_similarity"),
                    latency_ms=round(latency_ms, 1),
                )
                return QueryResponse(
                    answer=cached["answer"],
                    citations=[Citation(**c) for c in cached.get("citations", [])],
                    trace_id=trace_id,
                    latency_ms=round(latency_ms, 1),
                    cached=True,
                    cache_similarity=cached.get("cache_similarity"),
                )

        # ── 1. Retrieve ──────────────────────────
        retrieval_pipeline = RetrievalPipeline(db)
        chunks = await retrieval_pipeline.retrieve(
            query=safe_question,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranker=request.use_reranker,
            query_transform=request.query_transform,
            filters=request.filters.model_dump(exclude_none=True) if request.filters else None,
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please ingest documents first.",
            )

        # ── 2. Generate ──────────────────────────
        generator = Generator(
            model=request.model,
            num_predict=request.num_predict,
            max_chunk_chars=request.max_chunk_chars,
        )
        result = await generator.generate(
            question=safe_question,
            chunks=chunks,
        )

        # ── 2b. Guardrails — PII redaction only (fast, regex) ───
        # Groundedness scoring is an LLM call and runs post-response below.
        output_check = _guardrails.redact_output(result["answer"])
        result["answer"] = output_check["redacted_answer"]

        latency_ms = (time.perf_counter() - start) * 1000

        # ── 3. Store query record ─────────────────
        query_record = QueryModel(
            text=safe_question,
            answer=result["answer"],
            citations=result["citations"],
            trace_id=trace_id,
            retrieval_scores=[
                {
                    "chunk_id": c["chunk_id"],
                    "dense_score": c.get("dense_score", 0),
                    "sparse_score": c.get("sparse_score"),
                    "rrf_score": c.get("rrf_score"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in chunks
            ],
            latency_ms=round(latency_ms, 1),
            total_tokens=result.get("tokens_used"),
            total_cost_usd=result.get("cost_usd"),
            config_snapshot={
                "top_k": request.top_k,
                "use_hybrid": request.use_hybrid,
                "use_reranker": request.use_reranker,
                "query_transform": request.query_transform,
                "model": generator.model,
                "filters": request.filters.model_dump(exclude_none=True)
                if request.filters
                else None,
            },
        )
        db.add(query_record)
        await db.commit()
        await db.refresh(query_record)
        query_id = str(query_record.id)

        logger.info(
            "RAG query complete",
            trace_id=trace_id,
            latency_ms=round(latency_ms, 1),
            chunks_used=len(chunks),
            citations=len(result["citations"]),
        )

        response = QueryResponse(
            answer=result["answer"],
            citations=[Citation(**c) for c in result["citations"]],
            trace_id=trace_id,
            latency_ms=round(latency_ms, 1),
            tokens_used=result.get("tokens_used"),
        )

    # Everything below observes the answer rather than shaping it, so it runs
    # after the response is flushed: groundedness scoring (an LLM call),
    # semantic-cache write, and span persistence.
    background.add_task(
        _post_response_work,
        trace_id=trace_id,
        query_id=query_id,
        question=safe_question,
        answer=result["answer"],
        citations=result["citations"],
        chunks=chunks,
        latency_ms=round(latency_ms, 1),
        use_cache=request.use_cache,
    )
    return response


async def _post_response_work(
    trace_id: str,
    query_id: str | None,
    question: str,
    answer: str,
    citations: list[dict],
    chunks: list[dict],
    latency_ms: float,
    use_cache: bool,
) -> None:
    """Post-response work. Runs after the client has its answer.

    Failures here are logged, never raised — the user already has a valid
    response and must not be affected by observability work.
    """
    # ── Groundedness / hallucination scoring ──
    try:
        hallucination = await _guardrails.score_groundedness(
            answer=answer,
            context_chunks=chunks,
        )
        if hallucination:
            logger.info(
                "Groundedness scored",
                trace_id=trace_id,
                score=hallucination.get("groundedness_score"),
                is_hallucination=hallucination.get("is_hallucination"),
            )
    except Exception as e:
        logger.warning("Groundedness scoring failed", trace_id=trace_id, error=str(e))

    # ── Semantic cache write ──────────────────
    if use_cache:
        try:
            await _semantic_cache.put(
                query=question,
                answer=answer,
                citations=citations,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.warning("Cache store failed", trace_id=trace_id, error=str(e))

    # ── Span persistence ──────────────────────
    # Needs its own session: the request-scoped one is closed by now.
    try:
        async with async_session() as session:
            await persist_trace(session, trace_id, query_id=query_id)
    except Exception as e:
        logger.warning("Trace persistence failed", trace_id=trace_id, error=str(e))


@router.post("/stream")
async def query_rag_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question with streaming response (SSE).

    Feature parity with POST /query. This endpoint previously skipped input
    guardrails, output PII redaction, the query record and trace persistence —
    which mattered because it is the endpoint that exposes time-to-first-token,
    so it is the one most likely to be put in front of users.

    Output redaction cannot be applied per token: PII spans token boundaries,
    so "john" + "@example.com" passes when neither half matches alone.
    StreamingRedactor holds back a trailing window and only releases text once
    no pattern could still grow to cover it.
    """
    start = time.perf_counter()

    with create_span(
        tracer,
        "rag_query_stream",
        "CHAIN",
        {
            "query.text": request.question,
            "query.use_cache": request.use_cache,
            "query.streaming": True,
        },
    ):
        trace_id = get_current_trace_id() or uuid.uuid4().hex[:16]
        otel_trace.get_current_span().set_attribute("query.trace_id", trace_id)

        # ── 0a. Guardrails — input validation ────
        input_check = _guardrails.check_input(request.question)
        if input_check["blocked"]:
            raise HTTPException(status_code=400, detail=input_check["reason"])
        safe_question = input_check["redacted_query"]

        # ── 0b. Semantic cache check ─────────────
        if request.use_cache:
            cached = await _semantic_cache.get(safe_question)
            if cached:
                cached_answer = cached["answer"]

                async def cached_stream():
                    # Cached answers were redacted before storage, so they are
                    # replayed as-is.
                    for word in cached_answer.split():
                        yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
                    yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    cached_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

        # ── 1. Retrieve ──────────────────────────
        retrieval_pipeline = RetrievalPipeline(db)
        chunks = await retrieval_pipeline.retrieve(
            query=safe_question,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranker=request.use_reranker,
            query_transform=request.query_transform,
            filters=request.filters.model_dump(exclude_none=True) if request.filters else None,
        )

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        generator = Generator(
            model=request.model,
            num_predict=request.num_predict,
            max_chunk_chars=request.max_chunk_chars,
        )

    async def event_stream():
        # tail_chars trades PII safety against TTFT — the client sees nothing
        # until this many characters have been generated. See config for the
        # measured cost.
        redactor = StreamingRedactor(
            _guardrails.pii_redactor,
            tail_chars=settings.stream_pii_tail_chars,
        )
        try:
            async for token_json in generator.generate_stream(
                question=safe_question,
                chunks=chunks,
            ):
                payload = json.loads(token_json)
                safe = redactor.feed(payload.get("token", ""))
                if safe:
                    yield f"data: {json.dumps({'token': safe, 'done': False})}\n\n"

            tail = redactor.flush()
            if tail:
                yield f"data: {json.dumps({'token': tail, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.warning("Stream generation failed", trace_id=trace_id, error=str(e))
            yield f"data: {json.dumps({'token': '', 'done': True, 'error': 'generation failed'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # The client has the full answer by now, so persistence runs here
        # rather than in a BackgroundTask — a StreamingResponse's background
        # tasks fire only after the generator is exhausted anyway, and doing it
        # inline keeps the answer text in scope.
        answer = redactor.redacted_text
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        query_id: str | None = None

        try:
            async with async_session() as session:
                record = QueryModel(
                    text=safe_question,
                    answer=answer,
                    citations=[],
                    trace_id=trace_id,
                    retrieval_scores=[
                        {
                            "chunk_id": c["chunk_id"],
                            "dense_score": c.get("dense_score", 0),
                            "sparse_score": c.get("sparse_score"),
                            "rrf_score": c.get("rrf_score"),
                            "rerank_score": c.get("rerank_score"),
                        }
                        for c in chunks
                    ],
                    latency_ms=latency_ms,
                    config_snapshot={
                        "top_k": request.top_k,
                        "use_hybrid": request.use_hybrid,
                        "use_reranker": request.use_reranker,
                        "query_transform": request.query_transform,
                        "model": generator.model,
                        "streaming": True,
                    },
                )
                session.add(record)
                await session.commit()
                await session.refresh(record)
                query_id = str(record.id)
        except Exception as e:
            logger.warning("Stream query record failed", trace_id=trace_id, error=str(e))

        await _post_response_work(
            trace_id=trace_id,
            query_id=query_id,
            question=safe_question,
            answer=answer,
            citations=[],
            chunks=chunks,
            latency_ms=latency_ms,
            use_cache=request.use_cache,
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/cache/stats")
async def cache_stats():
    """Return semantic cache statistics."""
    stats = await _semantic_cache.stats()
    return stats


@router.delete("/cache")
async def invalidate_cache():
    """Clear the semantic cache."""
    deleted = await _semantic_cache.invalidate_all()
    return {"deleted": deleted, "message": "Semantic cache cleared"}
