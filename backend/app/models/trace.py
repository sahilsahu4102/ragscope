"""
RAGScope — Trace & Span Models (Phase 4)

Persistent OpenTelemetry trace store. Every query produces a Trace with a
tree of Spans (CHAIN → RETRIEVER → RERANKER → LLM), captured from the live
OTel span pipeline and written to Postgres for the trace-viewer waterfall.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Trace(Base):
    """A complete trace for one request (query, eval sample, ingestion job)."""

    __tablename__ = "traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # OTel trace id (128-bit) rendered as 32-char hex — links spans together
    otel_trace_id = Column(String(32), nullable=False, unique=True, index=True)

    # Optional link back to the query that produced this trace
    query_id = Column(UUID(as_uuid=True), nullable=True, index=True)

    name = Column(String(200), nullable=True, comment="Root span name, e.g. rag_query")
    status = Column(String(20), nullable=False, default="ok", comment="ok | error")

    # Aggregates rolled up from child spans
    total_duration_ms = Column(Float, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    total_cost_usd = Column(Float, nullable=True)
    span_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    spans = relationship(
        "Span",
        back_populates="trace",
        cascade="all, delete-orphan",
        order_by="Span.start_time",
    )

    __table_args__ = (Index("ix_traces_created_at", "created_at"),)


class Span(Base):
    """A single span within a trace, with OpenInference kind + gen_ai attributes."""

    __tablename__ = "spans"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(
        UUID(as_uuid=True), ForeignKey("traces.id", ondelete="CASCADE"), nullable=False
    )

    # OTel span identity (16-char hex) + parent for tree reconstruction
    otel_span_id = Column(String(16), nullable=False, index=True)
    parent_span_id = Column(String(16), nullable=True)

    span_kind = Column(
        String(20),
        nullable=False,
        default="CHAIN",
        comment="OpenInference kind: CHAIN|RETRIEVER|RERANKER|LLM|EMBEDDING|...",
    )
    name = Column(String(200), nullable=False)
    status = Column(String(20), nullable=False, default="ok", comment="ok | error")

    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)

    # Per-span cost/token rollups (populated for LLM/EMBEDDING spans)
    tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)

    attributes = Column(
        JSONB,
        nullable=True,
        comment="Full span attributes: gen_ai.*, document.*, reranker.*, etc.",
    )

    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    trace = relationship("Trace", back_populates="spans")

    __table_args__ = (
        Index("ix_spans_trace_id", "trace_id"),
        Index("ix_spans_kind", "span_kind"),
    )
