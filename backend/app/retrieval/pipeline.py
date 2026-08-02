"""
RAGScope — Full Retrieval Pipeline (Phase 5 — Optimized)

Orchestrates: query transform → dense → sparse → RRF fusion → rerank → cache.
All modes are switchable via request params for A/B experimentation.

Phase 5 optimizations:
  - Dense + sparse retrieval run concurrently via asyncio.gather() (~40% faster)
  - BM25 index is cached in-process across requests
"""

import asyncio

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session
from app.observability.tracer import create_span, get_tracer
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query_transform import QueryTransformer
from app.retrieval.rerankers import get_reranker
from app.retrieval.sparse import SparseRetriever

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


class RetrievalPipeline:
    """
    Full retrieval orchestrator.

    Supports multiple retrieval strategies selectable per-request:
      - Dense only (Phase 1 baseline)
      - Hybrid (dense + sparse + RRF fusion)
      - Hybrid + rerank (highest quality)

    Each step is independently traced for A/B comparison.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.dense_retriever = DenseRetriever(db)
        self.sparse_retriever = SparseRetriever(db)
        self.query_transformer = QueryTransformer()

    async def _sparse_isolated(
        self,
        query: str,
        top_k: int,
        backend: str | None = None,
    ) -> list[dict]:
        """Run sparse retrieval on a dedicated session.

        Needed because this is gathered concurrently with dense retrieval, and
        a single AsyncSession cannot service two operations at once.
        """
        async with async_session() as session:
            retriever = SparseRetriever(session, backend=backend)
            return await retriever.retrieve(query=query, top_k=top_k)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = False,
        use_reranker: bool = False,
        query_transform: str = "none",
        rrf_k: int = 60,
        filters: dict | None = None,
        reranker_backend: str | None = None,
        sparse_backend: str | None = None,
    ) -> list[dict]:
        """
        Run the full retrieval pipeline.

        Args:
            query: User's question
            top_k: Number of final results
            use_hybrid: Whether to combine dense + sparse via RRF
            use_reranker: Whether to apply cross-encoder reranking
            query_transform: "none", "rewrite", "hyde", or "decompose"
            rrf_k: RRF constant (default 60)
            filters: Optional metadata pre-filters dict
            reranker_backend: "cross_encoder" | "ollama". None uses the
                configured default. Exposed so A/B experiments can vary the
                reranker without changing global settings.
            sparse_backend: "postgres_fts" | "bm25". Same rationale.

        Returns:
            Ranked list of chunk dicts with scores from each stage.
        """
        with create_span(
            tracer,
            "retrieval_pipeline",
            "CHAIN",
            {
                "retrieval.mode": "hybrid" if use_hybrid else "dense",
                "retrieval.use_reranker": use_reranker,
                "retrieval.query_transform": query_transform,
                "retrieval.top_k": top_k,
                "retrieval.rrf_k": rrf_k,
                "retrieval.has_filters": bool(filters),
            },
        ):
            # ── 1. Query transformation ──────────────
            if query_transform != "none":
                with create_span(
                    tracer,
                    "query_transform",
                    "CHAIN",
                    {
                        "transform.method": query_transform,
                    },
                ):
                    transformed = await self.query_transformer.transform(
                        query, method=query_transform
                    )
                    # If decomposed, use first sub-query for now
                    # (multi-query retrieval is Phase 3)
                    if isinstance(transformed, list):
                        effective_query = transformed[0]
                    else:
                        effective_query = transformed
            else:
                effective_query = query

            # ── 2. Dense retrieval ────────────────────
            # Fetch more candidates if reranking will trim
            fetch_k = top_k * 4 if use_reranker else (top_k * 2 if use_hybrid else top_k)

            if not use_hybrid:
                # Dense-only path
                dense_results = await self.dense_retriever.retrieve(
                    query=effective_query,
                    top_k=fetch_k,
                    filters=filters,
                )
                candidates = dense_results
            else:
                # ── 3. Concurrent dense + sparse retrieval ──
                # Phase 5: Run both retrievers in parallel via asyncio.gather
                # This cuts ~40% off hybrid retrieval time since the two
                # are independent I/O-bound operations.
                # Sparse runs on its own session: an AsyncSession does not
                # support concurrent operations, and now that the FTS backend
                # is a plain DB query both sides genuinely overlap. Previously
                # this was masked by dense spending its first ~30ms on the HTTP
                # embedding call while BM25 finished its query.
                dense_results, sparse_results = await asyncio.gather(
                    self.dense_retriever.retrieve(
                        query=effective_query,
                        top_k=fetch_k,
                        filters=filters,
                    ),
                    self._sparse_isolated(
                        query=effective_query,
                        top_k=fetch_k,
                        backend=sparse_backend,
                    ),
                )

                # ── 4. RRF fusion ─────────────────────
                candidates = reciprocal_rank_fusion(
                    ranked_lists=[dense_results, sparse_results],
                    k=rrf_k,
                    top_k=fetch_k,
                )

            # ── 5. Reranking ─────────────────────────
            if use_reranker and candidates:
                # Backend defaults to settings.reranker_backend (cross_encoder).
                # Was hardcoded to the Ollama LLM reranker, which measured
                # 62-67s per query. A per-request override lets experiments
                # compare backends on identical inputs.
                use_cross_encoder = (
                    None
                    if reranker_backend is None
                    else reranker_backend.lower() == "cross_encoder"
                )
                reranker = get_reranker(use_cross_encoder=use_cross_encoder)
                candidates = await reranker.rerank(
                    query=effective_query,
                    chunks=candidates,
                    top_k=top_k,
                )
            else:
                candidates = candidates[:top_k]

            # Log pipeline results
            mode = (
                "hybrid+rerank"
                if (use_hybrid and use_reranker)
                else ("hybrid" if use_hybrid else ("dense+rerank" if use_reranker else "dense"))
            )

            logger.info(
                "Retrieval pipeline complete",
                mode=mode,
                query_transform=query_transform,
                returned=len(candidates),
                top_score=candidates[0].get(
                    "rerank_score",
                    candidates[0].get("rrf_score", candidates[0].get("dense_score", 0)),
                )
                if candidates
                else 0,
            )

            return candidates
