"""
RAGScope — Retrieve Debug API Router (Phase 2)

Debug endpoint that returns retrieved chunks with per-stage scores
(dense, sparse, RRF, rerank) without triggering generation.
Now supports query transformation and RRF tuning.
"""

import time
import structlog

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.schemas import RetrieveRequest, RetrieveResponse, ChunkScore
from app.retrieval.pipeline import RetrievalPipeline

logger = structlog.get_logger()
router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("", response_model=RetrieveResponse)
async def retrieve_debug(
    request: RetrieveRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Debug retrieval — returns chunks with all intermediate scores.

    No generation is performed. This powers the Retrieval Inspector
    page, showing how chunks flow through dense → sparse → RRF → rerank.

    Supports:
      - Query transformation (rewrite, HyDE, decompose)
      - Hybrid mode toggle (dense-only vs dense+sparse+RRF)
      - Reranker toggle
      - RRF constant tuning
    """
    start = time.perf_counter()

    pipeline = RetrievalPipeline(db)
    chunks = await pipeline.retrieve(
        query=request.question,
        top_k=request.top_k,
        use_hybrid=request.use_hybrid,
        use_reranker=request.use_reranker,
        query_transform=request.query_transform,
        rrf_k=request.rrf_k,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    chunk_scores = [
        ChunkScore(
            chunk_id=c["chunk_id"],
            content=c["content"],
            document_name=c["document_name"],
            dense_score=c.get("dense_score"),
            sparse_score=c.get("sparse_score"),
            rrf_score=c.get("rrf_score"),
            rerank_score=c.get("rerank_score"),
            element_type=c.get("element_type"),
            page_number=c.get("metadata", {}).get("page_number"),
        )
        for c in chunks
    ]

    logger.info(
        "Debug retrieval complete",
        question_length=len(request.question),
        results=len(chunk_scores),
        mode="hybrid" if request.use_hybrid else "dense",
        reranked=request.use_reranker,
        query_transform=request.query_transform,
        latency_ms=round(latency_ms, 1),
    )

    return RetrieveResponse(
        chunks=chunk_scores,
        latency_ms=round(latency_ms, 1),
    )
