"""
RAGScope — Full Retrieval Pipeline (Phase 5 — Optimized)

Orchestrates: query transform → dense → sparse → RRF fusion → rerank → cache.
All modes are switchable via request params for A/B experimentation.

Phase 5 optimizations:
  - Dense + sparse retrieval run concurrently via asyncio.gather() (~40% faster)
  - BM25 index is cached in-process across requests

Strategies (Phase 8):
  - standard: the pipeline above
  - graph: adds an entity-graph arm (personalized PageRank) to the RRF fusion
  - agentic: multi-hop — decompose, retrieve per sub-question, one LLM-written
    follow-up query when evidence is missing, RRF over all hops
"""

import asyncio

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session
from app.observability.tracer import create_span, get_tracer
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.graph import GraphRetriever
from app.retrieval.multihop import MultiHopRetriever
from app.retrieval.query_transform import QueryTransformer
from app.retrieval.rerankers import get_reranker
from app.retrieval.sparse import SparseRetriever

logger = structlog.get_logger()
tracer = get_tracer("retrieval")

STRATEGIES = ("standard", "agentic", "graph")
AGENTIC_MAX_FOLLOWUPS = 1


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

    async def _graph_isolated(self, query: str, top_k: int) -> list[dict]:
        """Entity-graph retrieval on its own session, for the same reason."""
        async with async_session() as session:
            return await GraphRetriever(session).retrieve(query=query, top_k=top_k)

    async def _standard_isolated(self, query: str, **kwargs) -> list[dict]:
        """One standard retrieval on its own session, so multi-hop can run hops concurrently."""
        async with async_session() as session:
            return await RetrievalPipeline(session).retrieve(query=query, **kwargs)

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
        strategy: str = "standard",
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
            strategy: "standard" | "agentic" | "graph" (see module docstring).
                query_transform="decompose" runs the multi-hop path without
                the follow-up step.

        Returns:
            Ranked list of chunk dicts with scores from each stage.
        """
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown retrieval strategy {strategy!r}; expected one of {STRATEGIES}"
            )

        with create_span(
            tracer,
            "retrieval_pipeline",
            "CHAIN",
            {
                "retrieval.mode": "hybrid" if use_hybrid else "dense",
                "retrieval.strategy": strategy,
                "retrieval.use_reranker": use_reranker,
                "retrieval.query_transform": query_transform,
                "retrieval.top_k": top_k,
                "retrieval.rrf_k": rrf_k,
                "retrieval.has_filters": bool(filters),
            },
        ):
            # ── 0. Multi-hop: each hop is a full standard retrieval ──
            if strategy == "agentic" or query_transform == "decompose":
                multihop = MultiHopRetriever(
                    retrieve_one=lambda q: self._standard_isolated(
                        q,
                        top_k=top_k,
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                        rrf_k=rrf_k,
                        filters=filters,
                        reranker_backend=reranker_backend,
                        sparse_backend=sparse_backend,
                    ),
                    transformer=self.query_transformer,
                    max_followups=AGENTIC_MAX_FOLLOWUPS if strategy == "agentic" else 0,
                )
                return await multihop.retrieve(query, top_k=top_k)

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
                    # Only decompose returns a list, and it took the multi-hop path above.
                    effective_query = transformed if isinstance(transformed, str) else query
            else:
                effective_query = query

            # ── 2-3. Candidate generation ─────────────
            # Fetch more candidates if reranking will trim
            fetch_k = top_k * 4 if use_reranker else (top_k * 2 if use_hybrid else top_k)

            # Every arm runs concurrently via asyncio.gather (Phase 5: ~40% off
            # hybrid retrieval, since the arms are independent I/O). Sparse and
            # graph run on their own sessions: an AsyncSession does not support
            # concurrent operations, and once the FTS backend became a plain DB
            # query both sides genuinely overlapped. Previously this was masked
            # by dense spending its first ~30ms on the HTTP embedding call while
            # BM25 finished its query.
            arms = [
                self.dense_retriever.retrieve(
                    query=effective_query,
                    top_k=fetch_k,
                    filters=filters,
                )
            ]
            if use_hybrid:
                arms.append(
                    self._sparse_isolated(
                        query=effective_query,
                        top_k=fetch_k,
                        backend=sparse_backend,
                    )
                )
            if strategy == "graph":
                arms.append(self._graph_isolated(query=effective_query, top_k=fetch_k))
            ranked_lists = await asyncio.gather(*arms)

            # ── 4. RRF fusion ─────────────────────
            if len(ranked_lists) == 1:
                candidates = ranked_lists[0]
            else:
                candidates = reciprocal_rank_fusion(
                    ranked_lists=list(ranked_lists),
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
                strategy=strategy,
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
