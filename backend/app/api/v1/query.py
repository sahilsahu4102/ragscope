"""
RAGScope — Query API Router (Phase 5)

Endpoints for RAG question-answering with citations, SSE streaming,
semantic caching, configurable query transformation, and guardrails.

Phase 5: Integrated guardrails pipeline (PII redaction, injection
detection, hallucination scoring) into the query path.
"""

import time
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from opentelemetry import trace as otel_trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.caching.semantic_cache import SemanticCache
from app.db.session import get_db
from app.config import settings
from app.generation.generator import Generator
from app.guardrails import GuardrailsPipeline
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
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please ingest documents first.",
            )

        # ── 2. Generate ──────────────────────────
        generator = Generator()
        result = await generator.generate(
            question=safe_question,
            chunks=chunks,
        )

        # ── 2b. Guardrails — output validation ───
        output_check = await _guardrails.check_output(
            answer=result["answer"],
            context_chunks=chunks,
        )
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
            },
        )
        db.add(query_record)
        await db.commit()
        await db.refresh(query_record)
        query_id = str(query_record.id)

        # ── 4. Cache the result ───────────────────
        if request.use_cache:
            await _semantic_cache.put(
                query=safe_question,
                answer=result["answer"],
                citations=result["citations"],
                latency_ms=round(latency_ms, 1),
            )

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

    # Persist the full span tree after the root span has ended.
    await persist_trace(db, trace_id, query_id=query_id)
    return response


@router.post("/stream")
async def query_rag_stream(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question with streaming response (SSE).

    Streams tokens as they're generated by the LLM.
    Semantic cache is checked first — if hit, streams the cached answer.
    """
    # ── 0. Cache check ────────────────────────
    if request.use_cache:
        cached = await _semantic_cache.get(request.question)
        if cached:

            async def cached_stream():
                # Stream cached answer word by word for consistent UX
                import json

                for word in cached["answer"].split():
                    yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                cached_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    # ── 1. Retrieve (non-streaming) ───────────
    retrieval_pipeline = RetrievalPipeline(db)
    chunks = await retrieval_pipeline.retrieve(
        query=request.question,
        top_k=request.top_k,
        use_hybrid=request.use_hybrid,
        use_reranker=request.use_reranker,
        query_transform=request.query_transform,
    )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found.",
        )

    # ── 2. Stream generation ──────────────────
    generator = Generator()

    async def event_stream():
        async for token_json in generator.generate_stream(
            question=request.question,
            chunks=chunks,
        ):
            yield f"data: {token_json}\n\n"
        yield "data: [DONE]\n\n"

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
