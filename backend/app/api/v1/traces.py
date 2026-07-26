"""
RAGScope — Traces API Router (Phase 4)

Read access to the persisted OpenTelemetry trace store. Powers the frontend
trace list and the span-waterfall viewer.
"""

import uuid
from datetime import datetime
from typing import overload

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models import Span, Trace
from app.schemas.schemas import SpanResponse, TraceDetail, TraceSummary

logger = structlog.get_logger()
router = APIRouter(prefix="/traces", tags=["traces"])


# Overloaded so non-nullable columns (created_at, start_time) narrow to `str`
# for the response schemas, while nullable ones (end_time) stay optional.
@overload
def _iso(dt: datetime) -> str: ...
@overload
def _iso(dt: None) -> None: ...
def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@router.get("", response_model=list[TraceSummary])
async def list_traces(
    limit: int = Query(default=50, ge=1, le=500),
    min_latency_ms: float | None = Query(default=None, ge=0),
    max_latency_ms: float | None = Query(default=None, ge=0),
    status: str | None = Query(default=None, description="Filter by ok|error"),
    db: AsyncSession = Depends(get_db),
):
    """List traces, most recent first, with optional latency/status filters."""
    stmt = select(Trace).order_by(Trace.created_at.desc())

    if min_latency_ms is not None:
        stmt = stmt.where(Trace.total_duration_ms >= min_latency_ms)
    if max_latency_ms is not None:
        stmt = stmt.where(Trace.total_duration_ms <= max_latency_ms)
    if status:
        stmt = stmt.where(Trace.status == status)

    stmt = stmt.limit(limit)
    traces = (await db.execute(stmt)).scalars().all()

    return [
        TraceSummary(
            id=str(t.id),
            otel_trace_id=t.otel_trace_id,
            query_id=str(t.query_id) if t.query_id else None,
            name=t.name,
            status=t.status,
            total_duration_ms=t.total_duration_ms,
            total_tokens=t.total_tokens,
            total_cost_usd=t.total_cost_usd,
            span_count=t.span_count,
            created_at=_iso(t.created_at),
        )
        for t in traces
    ]


@router.get("/{trace_id}", response_model=TraceDetail)
async def get_trace(
    trace_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a full trace with its span tree. Accepts our UUID or the OTel hex id."""
    trace = None

    # Try UUID lookup first, then fall back to the OTel trace id.
    try:
        trace = (
            await db.execute(select(Trace).where(Trace.id == uuid.UUID(trace_id)))
        ).scalar_one_or_none()
    except ValueError:
        trace = None

    if trace is None:
        trace = (
            await db.execute(select(Trace).where(Trace.otel_trace_id == trace_id))
        ).scalar_one_or_none()

    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")

    spans = (
        (await db.execute(select(Span).where(Span.trace_id == trace.id).order_by(Span.start_time)))
        .scalars()
        .all()
    )

    return TraceDetail(
        id=str(trace.id),
        otel_trace_id=trace.otel_trace_id,
        query_id=str(trace.query_id) if trace.query_id else None,
        name=trace.name,
        status=trace.status,
        total_duration_ms=trace.total_duration_ms,
        total_tokens=trace.total_tokens,
        total_cost_usd=trace.total_cost_usd,
        span_count=trace.span_count,
        created_at=_iso(trace.created_at),
        spans=[
            SpanResponse(
                id=str(s.id),
                otel_span_id=s.otel_span_id,
                parent_span_id=s.parent_span_id,
                span_kind=s.span_kind,
                name=s.name,
                status=s.status,
                start_time=_iso(s.start_time),
                end_time=_iso(s.end_time),
                duration_ms=s.duration_ms,
                tokens=s.tokens,
                cost_usd=s.cost_usd,
                attributes=s.attributes,
            )
            for s in spans
        ],
    )
