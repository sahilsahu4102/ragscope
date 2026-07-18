"""
RAGScope — Feedback API Router (Phase 4)

Collects thumbs up/down + optional corrections on answers, correlated back to
the query (and thus its trace) for later analysis.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models import Feedback, Query, Trace
from app.schemas.schemas import FeedbackItem, FeedbackRequest, FeedbackResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """Submit feedback for a query. Accepts query_id or trace_id."""
    query_id: uuid.UUID | None = None

    if request.query_id:
        try:
            query_id = uuid.UUID(request.query_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid query_id")
    elif request.trace_id:
        # Resolve trace -> query_id (accept our UUID or the OTel hex id).
        trace = (
            await db.execute(select(Trace).where(Trace.otel_trace_id == request.trace_id))
        ).scalar_one_or_none()
        if trace is None:
            try:
                trace = (
                    await db.execute(select(Trace).where(Trace.id == uuid.UUID(request.trace_id)))
                ).scalar_one_or_none()
            except ValueError:
                trace = None
        if trace is None or trace.query_id is None:
            raise HTTPException(status_code=404, detail="No query found for trace_id")
        query_id = trace.query_id
    else:
        raise HTTPException(status_code=400, detail="Provide query_id or trace_id")

    # Verify the query exists.
    query = (await db.execute(select(Query).where(Query.id == query_id))).scalar_one_or_none()
    if query is None:
        raise HTTPException(status_code=404, detail="Query not found")

    feedback = Feedback(
        query_id=query_id,
        rating=request.rating,
        correction=request.correction,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)

    logger.info("Feedback recorded", query_id=str(query_id), rating=request.rating)

    return FeedbackResponse(
        id=str(feedback.id),
        query_id=str(query_id),
        message="Feedback recorded",
    )


@router.get("", response_model=list[FeedbackItem])
async def list_feedback(db: AsyncSession = Depends(get_db)):
    """List feedback records, most recent first."""
    rows = (
        (await db.execute(select(Feedback).order_by(Feedback.created_at.desc()).limit(200)))
        .scalars()
        .all()
    )
    return [
        FeedbackItem(
            id=str(f.id),
            query_id=str(f.query_id),
            rating=f.rating,
            correction=f.correction,
            created_at=f.created_at.isoformat(),
        )
        for f in rows
    ]
