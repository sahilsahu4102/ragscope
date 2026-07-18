"""
RAGScope — Retrieval Pipeline

Orchestrates the retrieval flow: query transform → retrieve → (rerank) → context assembly.
Phase 1: dense-only. Phase 2 adds hybrid/RRF + reranking.
"""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.dense import DenseRetriever
from app.observability.tracer import get_tracer, create_span

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


class RetrievalPipeline:
    """
    Retrieval orchestrator.
    
    Phase 1: Dense-only retrieval.
    Phase 2: Will add BM25 + RRF fusion + reranking.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.dense_retriever = DenseRetriever(db)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = False,
        use_reranker: bool = False,
    ) -> list[dict]:
        """
        Run the full retrieval pipeline.
        
        Returns ranked list of chunk dicts with scores.
        """
        with create_span(tracer, "retrieval_pipeline", "CHAIN", {
            "retrieval.mode": "hybrid" if use_hybrid else "dense",
            "retrieval.use_reranker": use_reranker,
            "retrieval.top_k": top_k,
        }):
            # Phase 1: Dense only
            # Phase 2: Will add sparse + RRF + reranker branches here
            candidates = await self.dense_retriever.retrieve(
                query=query,
                top_k=top_k * 4 if use_reranker else top_k,
            )

            # Return top-k (reranking will happen in Phase 2)
            results = candidates[:top_k]

            logger.info(
                "Retrieval pipeline complete",
                mode="dense",
                candidates=len(candidates),
                returned=len(results),
            )

            return results
