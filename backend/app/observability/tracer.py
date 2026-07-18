"""
RAGScope — OpenTelemetry Tracing Setup

Bootstrap the OTel tracer provider with OpenInference span conventions.
Wired from Phase 0 so every query is traced from day one; Phase 4 adds the
Postgres-backed SpanCollector, optional OTLP export, and gen_ai.* attribute
helpers for the full CHAIN → RETRIEVER → RERANKER → LLM span tree.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from app.observability.collector import SpanCollector, set_collector
from app.observability.cost import calculate_cost

# OpenInference span kind attribute key
OTEL_SPAN_KIND_KEY = "openinference.span.kind"

_VALID_SPAN_KINDS = {
    "LLM",
    "EMBEDDING",
    "RETRIEVER",
    "RERANKER",
    "CHAIN",
    "TOOL",
    "AGENT",
    "GUARDRAIL",
    "EVALUATOR",
    "PROMPT",
}


def setup_tracing(
    service_name: str = "ragscope",
    *,
    sampling_rate: float = 1.0,
    console_export: bool = False,
    otlp_endpoint: str = "",
) -> None:
    """Initialize OpenTelemetry with the Postgres SpanCollector.

    Args:
        service_name: resource service.name attribute.
        sampling_rate: fraction of traces persisted to Postgres (0..1).
        console_export: also print spans to stdout (dev only).
        otlp_endpoint: if set, export spans to an OTLP collector too.
    """
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.5.0",
        }
    )

    provider = TracerProvider(resource=resource)

    # Primary: buffer spans for Postgres persistence (trace viewer).
    collector = SpanCollector(sampling_rate=sampling_rate)
    set_collector(collector)
    provider.add_span_processor(collector)

    # Optional: console exporter for local debugging.
    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Optional: forward to an external OTLP collector (Grafana/Tempo/Phoenix).
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
        except Exception:  # pragma: no cover - optional dependency path
            pass

    trace.set_tracer_provider(provider)


def get_tracer(name: str = "ragscope") -> trace.Tracer:
    """Get a named tracer instance."""
    return trace.get_tracer(name)


def get_current_trace_id() -> str | None:
    """Return the active OTel trace id as 32-char hex, or None if unsampled."""
    ctx = trace.get_current_span().get_span_context()
    if ctx is None or not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x")


def create_span(
    tracer: trace.Tracer,
    name: str,
    span_kind_value: str,
    attributes: dict | None = None,
):
    """Create an OpenInference-annotated span.

    span_kind_value: one of the 10 OpenInference span kinds:
        LLM, EMBEDDING, RETRIEVER, RERANKER, CHAIN, TOOL,
        AGENT, GUARDRAIL, EVALUATOR, PROMPT
    """
    attrs = {OTEL_SPAN_KIND_KEY: span_kind_value}
    if attributes:
        attrs.update(_clean_attributes(attributes))
    return tracer.start_as_current_span(name, attributes=attrs)


def _clean_attributes(attributes: dict) -> dict:
    """Drop None values (OTel rejects them) and coerce unsupported types."""
    clean: dict = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = [v for v in value if v is not None]
        else:
            clean[key] = str(value)
    return clean


# ── gen_ai.* attribute builders (semantic conventions) ────────


def llm_attributes(
    *,
    system: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    operation: str = "chat",
) -> dict:
    """Build gen_ai.* + cost attributes for an LLM span."""
    total = input_tokens + output_tokens
    return {
        "gen_ai.system": system,
        "gen_ai.request.model": model,
        "gen_ai.operation.name": operation,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
        "gen_ai.usage.total_tokens": total,
        "cost.usd": calculate_cost(model, input_tokens, output_tokens),
    }


def retriever_attributes(*, retriever_type: str, top_k: int, documents: list[dict]) -> dict:
    """Build document.* attributes for a RETRIEVER span."""
    attrs: dict = {
        "retriever.type": retriever_type,
        "retriever.top_k": top_k,
        "retriever.document_count": len(documents),
    }
    for i, doc in enumerate(documents[:top_k]):
        attrs[f"retrieval.documents.{i}.document.id"] = str(doc.get("chunk_id", ""))
        score = doc.get("dense_score", doc.get("rrf_score", doc.get("rerank_score")))
        if score is not None:
            attrs[f"retrieval.documents.{i}.document.score"] = float(score)
    return attrs


def reranker_attributes(
    *,
    model_name: str,
    top_k: int,
    input_documents: int,
    output_documents: int,
) -> dict:
    """Build reranker.* attributes for a RERANKER span."""
    return {
        "reranker.model_name": model_name,
        "reranker.top_k": top_k,
        "reranker.input_documents": input_documents,
        "reranker.output_documents": output_documents,
    }
