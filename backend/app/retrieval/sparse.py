"""
RAGScope — BM25 Sparse Retrieval (Phase 5 — Optimized)

In-memory BM25 search using rank-bm25 library.
Catches exact terms (product codes, proper nouns) that dense retrieval misses.

Phase 5: Module-level BM25 index cache — built once, reused across requests.
Invalidated when new documents are ingested. Saves ~200ms per hybrid query
by not re-loading all chunks from Postgres on every request.
"""

from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")

# ── Module-level BM25 index cache ──────────────
_cached_index: BM25Okapi | None = None
_cached_chunks: list[dict] = []
_cached_chunk_count: int = 0


def invalidate_bm25_cache() -> None:
    """Call after ingesting new documents to rebuild the index on next query."""
    global _cached_index, _cached_chunks, _cached_chunk_count
    _cached_index = None
    _cached_chunks = []
    _cached_chunk_count = 0
    logger.info("BM25 index cache invalidated")


class SparseRetriever:
    """
    BM25 keyword-based retrieval.

    Loads all chunks into an in-memory BM25 index, scores by term overlap.
    Complements dense retrieval for exact-match queries (codes, names, acronyms).

    Phase 5: Uses module-level index cache — built once on first query,
    reused across requests until invalidated by new ingestion.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _build_index(self) -> tuple[BM25Okapi | None, list[dict]]:
        """Build BM25 index from all chunks in the database."""
        global _cached_index, _cached_chunks, _cached_chunk_count

        # Check if cache is still valid (chunk count matches)
        from sqlalchemy import func
        count_result = await self.db.execute(select(func.count(Chunk.id)))
        current_count = count_result.scalar() or 0

        if _cached_index is not None and _cached_chunk_count == current_count:
            return _cached_index, _cached_chunks

        # Rebuild index
        result = await self.db.execute(
            select(
                Chunk.id,
                Chunk.content,
                Chunk.element_type,
                Chunk.chunk_index,
                Chunk.token_count,
                Chunk.metadata_,
                Chunk.document_id,
                Document.filename,
            ).join(Document, Chunk.document_id == Document.id)
        )
        rows = result.fetchall()

        if not rows:
            logger.warning("No chunks found for BM25 index")
            return None, []

        chunks: list[dict] = []
        tokenized_corpus: list[list[str]] = []

        for row in rows:
            chunks.append(
                {
                    "chunk_id": str(row.id),
                    "content": row.content,
                    "document_name": row.filename,
                    "element_type": row.element_type,
                    "chunk_index": row.chunk_index,
                    "token_count": row.token_count,
                    "metadata": row.metadata_ or {},
                }
            )
            tokenized_corpus.append(row.content.lower().split())

        index = BM25Okapi(tokenized_corpus)

        # Cache for reuse
        _cached_index = index
        _cached_chunks = chunks
        _cached_chunk_count = current_count

        logger.info("BM25 index built and cached", chunks=len(chunks))
        return index, chunks

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Retrieve top-k chunks by BM25 score.

        Returns list of dicts with sparse_score field.
        """
        with create_span(
            tracer,
            "sparse_retrieve",
            "RETRIEVER",
            {
                "retriever.type": "bm25",
                "retriever.top_k": top_k,
            },
        ):
            index, chunks = await self._build_index()

            if not chunks or index is None:
                return []

            tokenized_query = query.lower().split()
            scores = index.get_scores(tokenized_query)

            scored_chunks = [
                {**chunk, "sparse_score": float(score)}
                for chunk, score in zip(chunks, scores, strict=True)
                if score > 0
            ]

            scored_chunks.sort(key=lambda x: x["sparse_score"], reverse=True)
            results = scored_chunks[:top_k]

            logger.info(
                "BM25 retrieval complete",
                results=len(results),
                top_score=results[0]["sparse_score"] if results else 0,
            )
            return results
