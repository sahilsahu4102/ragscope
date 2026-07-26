"""
RAGScope — Dense Retrieval (Phase 5 — Staged Hybrid Filtering)

Cosine similarity search over pgvector embeddings with 3-stage
metadata filtering for production-grade retrieval:

  Stage 1 — Pre-filter (indexed): Narrow by document_id, element_type,
            date range BEFORE the vector scan. Reduces search space,
            saves pgvector from scanning irrelevant chunks.

  Stage 2 — ANN vector search: Cosine similarity on the filtered subset.
            pgvector applies WHERE + ORDER BY <=> in a single index scan.

  Stage 3 — Post-filter (lightweight): min_score threshold, token_count
            range. Applied after retrieval, before reranking. Low cost
            since it operates on the small top-k result set.

This mirrors the production pattern from Qdrant/Pinecone/Weaviate
but using PostgreSQL + pgvector as the vector store.
"""

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.embedders.embedder import get_embedder
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


class DenseRetriever:
    """
    Dense vector retrieval using pgvector cosine similarity
    with 3-stage metadata filtering.

    Queries the chunks table with `<=>` (cosine distance) operator,
    returns top-k chunks ranked by similarity score.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = get_embedder()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks most similar to the query.

        Args:
            query: User's question text
            top_k: Number of results to return
            filters: Optional metadata filters:
                - document_ids: list[str] — restrict to specific documents
                - element_types: list[str] — e.g. ["paragraph", "table"]
                - date_from: str (ISO) — chunks from documents created after
                - date_to: str (ISO) — chunks from documents created before
                - min_score: float — minimum cosine similarity (Stage 3)
                - min_tokens: int — minimum chunk token count (Stage 3)
                - max_tokens: int — maximum chunk token count (Stage 3)

        Returns:
            List of dicts with: chunk_id, content, score, document_name, metadata.
        """
        top_k = top_k or settings.retrieval_top_k
        filters = filters or {}

        with create_span(
            tracer,
            "dense_retrieve",
            "RETRIEVER",
            {
                "retriever.type": "dense",
                "retriever.top_k": top_k,
                "retriever.query": query,
                "retriever.has_filters": bool(filters),
                "retriever.filter_keys": list(filters.keys()) if filters else [],
            },
        ):
            # ── Stage 2a: Embed the query ─────────────
            with create_span(
                tracer,
                "embed_query",
                "EMBEDDING",
                {
                    "embedding.model_name": self.embedder.model_name(),
                },
            ):
                query_embeddings = await self.embedder.embed([query])
                query_vector = query_embeddings[0]

            vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

            # ── Stage 1: Pre-filter (indexed WHERE clauses) ──
            where_clauses = ["c.embedding IS NOT NULL"]
            params: dict = {"query_vector": vector_str, "top_k": top_k}

            # Filter by specific document IDs
            if filters.get("document_ids"):
                where_clauses.append("c.document_id = ANY(:doc_ids)")
                params["doc_ids"] = filters["document_ids"]

            # Filter by element types (paragraph, table, title, etc.)
            if filters.get("element_types"):
                where_clauses.append("c.element_type = ANY(:elem_types)")
                params["elem_types"] = filters["element_types"]

            # Filter by document creation date range
            if filters.get("date_from"):
                where_clauses.append("d.created_at >= :date_from")
                params["date_from"] = filters["date_from"]

            if filters.get("date_to"):
                where_clauses.append("d.created_at <= :date_to")
                params["date_to"] = filters["date_to"]

            where_sql = " AND ".join(where_clauses)

            # ── Stage 2b: ANN vector search on filtered subset ──
            # Fetch extra candidates for Stage 3 post-filtering
            fetch_limit = top_k * 2 if filters.get("min_score") else top_k

            # NOTE: use CAST(... AS vector), not `:query_vector::vector`.
            # SQLAlchemy's text() bind regex is `(?<![:\w$]):([\w$]+)(?![:\w$])`
            # — the leading colon of a `::` cast satisfies that trailing
            # lookahead, so `:query_vector` is silently NOT bound and Postgres
            # receives the literal text, failing with `syntax error at or near ":"`.
            sql = text(f"""
                SELECT
                    c.id,
                    c.content,
                    c.element_type,
                    c.metadata,
                    c.chunk_index,
                    c.token_count,
                    d.filename as document_name,
                    d.id as document_id,
                    1 - (c.embedding <=> CAST(:query_vector AS vector)) as score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE {where_sql}
                ORDER BY c.embedding <=> CAST(:query_vector AS vector)
                LIMIT :top_k
            """)

            params["top_k"] = fetch_limit

            result = await self.db.execute(sql, params)
            rows = result.fetchall()

            chunks = []
            for row in rows:
                chunks.append(
                    {
                        "chunk_id": str(row.id),
                        "content": row.content,
                        "document_name": row.document_name,
                        "document_id": str(row.document_id),
                        "dense_score": float(row.score),
                        "element_type": row.element_type,
                        "chunk_index": row.chunk_index,
                        "token_count": row.token_count,
                        "metadata": row.metadata or {},
                    }
                )

            # ── Stage 3: Post-filter (lightweight, on result set) ──
            raw_min_score = filters.get("min_score")
            if raw_min_score is not None:
                threshold = float(str(raw_min_score))
                chunks = [c for c in chunks if float(str(c["dense_score"])) >= threshold]

            raw_min_tokens = filters.get("min_tokens")
            if raw_min_tokens is not None:
                floor = int(str(raw_min_tokens))
                chunks = [
                    c for c in chunks
                    if c.get("token_count") is not None
                    and int(str(c["token_count"])) >= floor
                ]

            raw_max_tokens = filters.get("max_tokens")
            if raw_max_tokens is not None:
                ceiling = int(str(raw_max_tokens))
                chunks = [
                    c for c in chunks
                    if c.get("token_count") is not None
                    and int(str(c["token_count"])) <= ceiling
                ]

            # Trim to requested top_k after post-filtering
            chunks = chunks[:top_k]

            logger.info(
                "Dense retrieval complete",
                query_length=len(query),
                results=len(chunks),
                top_score=chunks[0]["dense_score"] if chunks else 0,
                pre_filters=len(where_clauses) - 1,  # minus the IS NOT NULL
                post_filters=sum(
                    1
                    for k in ("min_score", "min_tokens", "max_tokens")
                    if filters.get(k)
                ),
            )

            return chunks
