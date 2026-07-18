"""
RAGScope — API v1 Router

Aggregates all v1 endpoint routers.
"""

from fastapi import APIRouter

router = APIRouter(tags=["v1"])


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
            "evaluate": "/api/v1/evaluate",
            "experiments": "/api/v1/experiments",
            "traces": "/api/v1/traces",
            "feedback": "/api/v1/feedback",
            "datasets": "/api/v1/datasets",
        },
    }
