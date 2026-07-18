"""
RAGScope — API v1 Router

Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.datasets import router as datasets_router
from app.api.v1.evaluate import router as evaluate_router
from app.api.v1.ingest import router as ingest_router
from app.api.v1.query import router as query_router
from app.api.v1.retrieve import router as retrieve_router

router = APIRouter(tags=["v1"])

# Mount endpoint routers
router.include_router(ingest_router)
router.include_router(query_router)
router.include_router(retrieve_router)
router.include_router(datasets_router)
router.include_router(evaluate_router)


@router.get("/")
async def api_root():
    """API v1 root — lists available endpoints."""
    return {
        "api": "RAGScope",
        "version": "v1",
        "endpoints": {
            "ingest": "/api/v1/ingest",
            "query": "/api/v1/query",
            "retrieve": "/api/v1/retrieve",
            "datasets": "/api/v1/datasets",
            "evaluate": "/api/v1/evaluate",
            "experiments": "/api/v1/experiments",
            "traces": "/api/v1/traces",
            "feedback": "/api/v1/feedback",
        },
    }
