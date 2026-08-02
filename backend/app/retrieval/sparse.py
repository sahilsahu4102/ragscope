"""
RAGScope — Sparse (lexical) Retrieval

Catches exact terms — product codes, proper nouns, acronyms — that dense
retrieval misses.

Two backends, selected by settings.sparse_backend.

  bm25 (default)
      In-process rank-bm25. Scores the entire corpus in numpy per query, holds
      a copy per worker process, and must be rebuilt after ingestion.

  postgres_fts
      GIN-indexed tsvector column ranked with ts_rank_cd. Shared across
      workers, no warm-up, stays correct after ingestion with no invalidation.

FTS was added on the assumption that BM25's O(N) scoring would lose at scale.
Measured (app/scripts/sparse_scale.py), it does not:

    rows       postgres_fts     bm25
    5,000          15.1 ms     6.4 ms
    25,000        104.4 ms    45.3 ms
    100,000       423.2 ms   188.8 ms

BM25 is ~2.2x faster at every size tested; there is no crossover in this
range. The reason is ranking, not lookup: question-shaped queries need OR
semantics (AND matches nothing — see _retrieve_fts), OR matches ~16% of the
corpus, and ts_rank_cd has to score and sort all of it. The GIN index finds
candidates quickly; scoring them is the cost.

So the default stayed bm25. postgres_fts is kept because its advantages are
real but operational rather than latency: no per-worker memory, no rebuild
stall after ingestion, and correctness across multiple workers.

Neither is good at 100k (189ms and 423ms both dominate the retrieval budget).
The real fix at that scale is a purpose-built lexical index — pg_search /
ParadeDB, or an external engine — not tuning either of these.
"""

from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
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
    Lexical retrieval, backed by Postgres FTS or in-process BM25.

    Complements dense retrieval for exact-match queries (codes, names,
    acronyms). Backend comes from settings.sparse_backend; pass `backend` to
    override per request for A/B comparison.
    """

    def __init__(self, db: AsyncSession, backend: str | None = None):
        self.db = db
        self.backend = (backend or settings.sparse_backend).lower()

    async def _retrieve_fts(self, query: str, top_k: int) -> list[dict]:
        """Rank with ts_rank_cd over the GIN-indexed tsvector column.

        Term combination matters more than it looks. Both plainto_tsquery and
        websearch_to_tsquery AND their terms together, so a natural-language
        question requires *every* term to appear in one chunk. Measured on this
        corpus, the question "Why does the synchronous nature of Llama 3
        16K-GPU training make it less fault-tolerant" matched 0 chunks under
        AND and 923 under OR. AND semantics would silently delete the sparse
        arm of hybrid retrieval for question-shaped queries.

        So the query is built by letting plainto_tsquery do normalisation —
        stemming ('synchronous' -> 'synchron'), stopword removal, punctuation
        handling — then rewriting its '&' operators to '|'. Hand-tokenising
        would lose the stemming and the dictionary.

        NULLIF guards a query that is entirely stopwords: to_tsquery('') raises,
        whereas NULL simply matches nothing.

        ts_rank_cd is cover-density ranking — it rewards matched terms
        appearing close together. It is not BM25; there is no document-length
        saturation term, so ranking differs from the legacy backend. That is
        why bm25 stays available for A/B rather than being deleted.
        """
        sql = text("""
            WITH q AS (
                SELECT to_tsquery(
                    'english',
                    NULLIF(replace(plainto_tsquery('english', :query)::text, '&', '|'), '')
                ) AS tsq
            )
            SELECT
                c.id,
                c.content,
                c.element_type,
                c.chunk_index,
                c.token_count,
                c.metadata,
                d.filename AS document_name,
                ts_rank_cd(c.content_tsv, q.tsq) AS score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            CROSS JOIN q
            WHERE q.tsq IS NOT NULL AND c.content_tsv @@ q.tsq
            ORDER BY score DESC
            LIMIT :top_k
        """)
        rows = (await self.db.execute(sql, {"query": query, "top_k": top_k})).fetchall()
        return [
            {
                "chunk_id": str(r.id),
                "content": r.content,
                "document_name": r.document_name,
                "element_type": r.element_type,
                "chunk_index": r.chunk_index,
                "token_count": r.token_count,
                "metadata": r.metadata or {},
                "sparse_score": float(r.score),
            }
            for r in rows
        ]

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
        Retrieve top-k chunks by lexical relevance.

        Returns list of dicts with sparse_score field.
        """
        with create_span(
            tracer,
            "sparse_retrieve",
            "RETRIEVER",
            {
                "retriever.type": self.backend,
                "retriever.top_k": top_k,
            },
        ):
            if self.backend == "postgres_fts":
                results = await self._retrieve_fts(query, top_k)
                logger.info(
                    "Sparse retrieval complete",
                    backend="postgres_fts",
                    results=len(results),
                    top_score=results[0]["sparse_score"] if results else 0,
                )
                return results

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
