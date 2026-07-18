"""
RAGScope — OpenTelemetry Tracing Setup

Bootstrap OTel tracer provider with OpenInference span conventions.
Wired from Phase 0 so every query is traced from day one.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

# OpenInference span kind attribute key
OTEL_SPAN_KIND_KEY = "openinference.span.kind"


def setup_tracing(service_name: str = "ragscope") -> None:
    """Initialize OpenTelemetry with a console exporter (OTLP in Phase 4)."""

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
        }
    )

    provider = TracerProvider(resource=resource)

    # Console exporter for development — replace with OTLP in Phase 4
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)


def get_tracer(name: str = "ragscope") -> trace.Tracer:
    """Get a named tracer instance."""
    return trace.get_tracer(name)


def create_span(
    tracer: trace.Tracer,
    name: str,
    span_kind_value: str,
    attributes: dict | None = None,
):
    """
    Create an OpenInference-annotated span.

    span_kind_value: One of the 10 OpenInference span kinds:
        LLM, EMBEDDING, RETRIEVER, RERANKER, CHAIN, TOOL,
        AGENT, GUARDRAIL, EVALUATOR, PROMPT
    """
    attrs = {OTEL_SPAN_KIND_KEY: span_kind_value}
    if attributes:
        attrs.update(attributes)
    return tracer.start_as_current_span(name, attributes=attrs)
