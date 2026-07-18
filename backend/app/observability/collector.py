"""
RAGScope — Trace Collector (Phase 4)

A custom OpenTelemetry SpanProcessor that buffers finished spans in-process,
keyed by OTel trace id, then persists a full trace tree to Postgres on demand
from the async request context.

Why not a plain OTLP exporter? We want first-class, queryable traces in our own
database (for the trace-viewer waterfall) without standing up a separate
collector service. This processor keeps spans in memory just long enough for the
request handler to flush them through the app's async session — demonstrating
how OTel's span pipeline works internally while staying self-contained.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import UTC, datetime

import structlog
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanProcessor
from sqlalchemy.ext.asyncio import AsyncSession

from app.observability.cost import calculate_cost

logger = structlog.get_logger()

# OpenInference span-kind attribute key (mirrors tracer.py)
_SPAN_KIND_KEY = "openinference.span.kind"

# Cap on distinct traces held in memory; oldest are evicted first.
_MAX_BUFFERED_TRACES = 500


def _hex_trace_id(trace_id: int) -> str:
    return format(trace_id, "032x")


def _hex_span_id(span_id: int) -> str:
    return format(span_id, "016x")


def _ns_to_dt(ns: int | None) -> datetime | None:
    if not ns:
        return None
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=UTC)


def _jsonable_attributes(span: ReadableSpan) -> dict:
    """Coerce span attributes into a JSON-serializable dict."""
    out: dict = {}
    for key, value in (span.attributes or {}).items():
        if isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def _extract_tokens(attrs: dict) -> int | None:
    """Pull a total token count from gen_ai.* usage attributes if present."""
    total = attrs.get("gen_ai.usage.total_tokens")
    if total is not None:
        return int(total)
    inp = attrs.get("gen_ai.usage.input_tokens")
    out = attrs.get("gen_ai.usage.output_tokens")
    if inp is not None or out is not None:
        return int(inp or 0) + int(out or 0)
    return None


def _extract_cost(attrs: dict, tokens: int | None) -> float | None:
    """Resolve per-span cost from an explicit attribute or model + tokens."""
    if "cost.usd" in attrs:
        return float(attrs["cost.usd"])
    model = attrs.get("gen_ai.request.model") or attrs.get("gen_ai.system")
    if model and tokens:
        inp = int(attrs.get("gen_ai.usage.input_tokens") or 0)
        out = int(attrs.get("gen_ai.usage.output_tokens") or 0)
        if inp or out:
            return calculate_cost(str(model), inp, out)
    return None


class SpanCollector(SpanProcessor):
    """Buffers finished spans per trace for later async persistence.

    Registered directly on the TracerProvider. `on_end` runs synchronously in
    the thread that closes each span, so a lock is enough for safety.
    """

    def __init__(self, sampling_rate: float = 1.0):
        self._buffer: OrderedDict[str, list[ReadableSpan]] = OrderedDict()
        self._lock = threading.Lock()
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))

    def _sampled(self, trace_id: int) -> bool:
        if self.sampling_rate >= 1.0:
            return True
        if self.sampling_rate <= 0.0:
            return False
        # Deterministic per-trace: all spans of a trace share the decision.
        return (trace_id % 10_000) / 10_000 < self.sampling_rate

    def on_start(self, span, parent_context=None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        ctx = span.get_span_context()
        if ctx is None or not self._sampled(ctx.trace_id):
            return
        key = _hex_trace_id(ctx.trace_id)
        with self._lock:
            self._buffer.setdefault(key, []).append(span)
            self._buffer.move_to_end(key)
            while len(self._buffer) > _MAX_BUFFERED_TRACES:
                self._buffer.popitem(last=False)

    def drain(self, trace_id_hex: str) -> list[ReadableSpan]:
        """Remove and return all buffered spans for a trace."""
        with self._lock:
            return self._buffer.pop(trace_id_hex, [])

    def shutdown(self) -> None:
        with self._lock:
            self._buffer.clear()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


# ── Module-level singleton ────────────────────────────────
_collector: SpanCollector | None = None


def get_collector() -> SpanCollector | None:
    return _collector


def set_collector(collector: SpanCollector) -> None:
    global _collector
    _collector = collector


async def persist_trace(
    db: AsyncSession,
    trace_id_hex: str,
    query_id: str | None = None,
) -> str | None:
    """Flush buffered spans for a trace into the traces/spans tables.

    Returns the persisted Trace UUID (str) or None when nothing was buffered.
    Never raises into the request path — tracing must not break queries.
    """
    from app.models import Span as SpanModel
    from app.models import Trace as TraceModel

    collector = get_collector()
    if collector is None:
        return None

    readable = collector.drain(trace_id_hex)
    if not readable:
        return None

    try:
        span_rows: list[SpanModel] = []
        starts: list[datetime] = []
        ends: list[datetime] = []
        total_tokens = 0
        total_cost = 0.0
        any_error = False
        root_name: str | None = None

        for rs in readable:
            ctx = rs.get_span_context()
            attrs = _jsonable_attributes(rs)
            start_dt = _ns_to_dt(rs.start_time)
            end_dt = _ns_to_dt(rs.end_time)
            duration_ms = (
                round((rs.end_time - rs.start_time) / 1_000_000, 3)
                if (rs.start_time and rs.end_time)
                else None
            )

            tokens = _extract_tokens(attrs)
            cost = _extract_cost(attrs, tokens)
            if tokens:
                total_tokens += tokens
            if cost:
                total_cost += cost

            status_code = getattr(rs.status, "status_code", None)
            status = "error" if (status_code and status_code.name == "ERROR") else "ok"
            if status == "error":
                any_error = True

            if rs.parent is None:
                root_name = rs.name

            if start_dt:
                starts.append(start_dt)
            if end_dt:
                ends.append(end_dt)

            span_rows.append(
                SpanModel(
                    otel_span_id=_hex_span_id(ctx.span_id),
                    parent_span_id=(_hex_span_id(rs.parent.span_id) if rs.parent else None),
                    span_kind=str(attrs.get(_SPAN_KIND_KEY, "CHAIN")),
                    name=rs.name,
                    status=status,
                    start_time=start_dt,
                    end_time=end_dt,
                    duration_ms=duration_ms,
                    tokens=tokens,
                    cost_usd=round(cost, 8) if cost else None,
                    attributes=attrs,
                )
            )

        trace_start = min(starts) if starts else None
        trace_end = max(ends) if ends else None
        total_duration_ms = (
            round((trace_end - trace_start).total_seconds() * 1000, 3)
            if (trace_start and trace_end)
            else None
        )

        trace = TraceModel(
            otel_trace_id=trace_id_hex,
            query_id=query_id,
            name=root_name,
            status="error" if any_error else "ok",
            total_duration_ms=total_duration_ms,
            total_tokens=total_tokens or None,
            total_cost_usd=round(total_cost, 8) if total_cost else None,
            span_count=len(span_rows),
        )
        trace.spans = span_rows
        db.add(trace)
        await db.commit()
        await db.refresh(trace)

        logger.info(
            "Trace persisted",
            trace_id=trace_id_hex,
            spans=len(span_rows),
            duration_ms=total_duration_ms,
        )
        return str(trace.id)

    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Trace persistence failed", trace_id=trace_id_hex, error=str(e))
        await db.rollback()
        return None
