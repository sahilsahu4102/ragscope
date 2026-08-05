"""
RAGScope — API v1 Router

Aggregates all v1 endpoint routers and attaches abuse controls.

Rate limits are applied here rather than as global middleware so the tiers
match what each endpoint actually costs, and so health probes stay unlimited:

  query tier  — one LLM call per request
  heavy tier  — whole-corpus work (ingestion, evaluation, experiments,
                dataset generation), orders of magnitude more expensive
  read-only   — no limiter; cheap DB reads (traces, documents, analytics)

API-key auth applies to every router below when RAGSCOPE_API_KEY is set, and
is a no-op when it is not, so local development needs no setup.
"""

from fastapi import APIRouter, Depends

from app.api.v1.analytics import router as analytics_router
from app.api.v1.datasets import router as datasets_router
from app.api.v1.documents import router as documents_router
from app.api.v1.evaluate import router as evaluate_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.feedback import router as feedback_router
from app.api.v1.ingest import router as ingest_router
from app.api.v1.query import router as query_router
from app.api.v1.retrieve import router as retrieve_router
from app.api.v1.traces import router as traces_router
from app.security import limit_heavy, limit_query, require_api_key

router = APIRouter(tags=["v1"], dependencies=[Depends(require_api_key)])

QUERY_TIER = [Depends(limit_query)]
HEAVY_TIER = [Depends(limit_heavy)]

# Each request costs an LLM call.
router.include_router(query_router, dependencies=QUERY_TIER)
# Retrieval has no LLM call but still embeds + reranks, so it is not free.
router.include_router(retrieve_router, dependencies=QUERY_TIER)

# Whole-corpus work: parsing, embedding every chunk, or running an eval across
# a dataset. Minutes of compute per request.
router.include_router(ingest_router, dependencies=HEAVY_TIER)
router.include_router(datasets_router, dependencies=HEAVY_TIER)
router.include_router(evaluate_router, dependencies=HEAVY_TIER)
router.include_router(experiments_router, dependencies=HEAVY_TIER)

# Read-only, cheap: no limiter beyond the API key.
router.include_router(documents_router)
router.include_router(traces_router)
router.include_router(feedback_router)
router.include_router(analytics_router)


@router.get("/")
async def api_root():
    """API v1 root — lists available endpoints."""
    return {
        "api": "RAGScope",
        "version": "v1",
        "endpoints": {
            "ingest": "/api/v1/ingest",
            "documents": "/api/v1/documents",
            "query": "/api/v1/query",
            "retrieve": "/api/v1/retrieve",
            "datasets": "/api/v1/datasets",
            "evaluate": "/api/v1/evaluate",
            "experiments": "/api/v1/experiments",
            "traces": "/api/v1/traces",
            "feedback": "/api/v1/feedback",
        },
    }
