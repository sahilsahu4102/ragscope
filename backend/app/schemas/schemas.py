"""
RAGScope — Pydantic Schemas

Request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field

# ── Ingestion ─────────────────────────────────


class IngestRequest(BaseModel):
    """Request to ingest a document (URL-based)."""

    source_url: str | None = None
    chunker: str = Field(default="recursive", description="recursive | semantic | hierarchical")
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=512)


class IngestResponse(BaseModel):
    """Response after starting an ingestion job."""

    job_id: str
    document_id: str
    status: str
    message: str


class IngestStatusResponse(BaseModel):
    """Status of an ingestion job."""

    job_id: str
    document_id: str
    status: str
    chunks_created: int | None = None
    error: str | None = None


# ── Query ─────────────────────────────────────


class RetrievalFilters(BaseModel):
    """Metadata pre-filters for staged hybrid retrieval.

    Stage 1 (indexed, pre-filter): document_ids, element_types, date_from, date_to
    Stage 3 (post-filter): min_score, min_tokens, max_tokens
    """

    document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to specific document IDs"
    )
    element_types: list[str] | None = Field(
        default=None, description="Filter by chunk type: paragraph | table | title | list"
    )
    date_from: str | None = Field(
        default=None, description="ISO date — only documents created after this date"
    )
    date_to: str | None = Field(
        default=None, description="ISO date — only documents created before this date"
    )
    min_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum cosine similarity threshold"
    )
    min_tokens: int | None = Field(
        default=None, ge=1, description="Minimum chunk token count"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum chunk token count"
    )


class QueryRequest(BaseModel):
    """A user query to the RAG pipeline."""

    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = True
    use_hybrid: bool = True
    stream: bool = False
    query_transform: str = Field(
        default="none",
        description="Query transformation: none | rewrite | hyde | decompose",
    )
    use_cache: bool = Field(default=True, description="Check semantic cache before pipeline")
    filters: RetrievalFilters | None = Field(
        default=None, description="Metadata pre-filters for staged hybrid retrieval"
    )


class Citation(BaseModel):
    """A citation linking an answer claim to a source chunk."""

    chunk_id: str
    document_name: str
    content_snippet: str
    score: float
    page_number: int | None = None


class QueryResponse(BaseModel):
    """Response from the RAG pipeline."""

    answer: str
    citations: list[Citation]
    trace_id: str
    latency_ms: float
    tokens_used: int | None = None
    cached: bool = False
    cache_similarity: float | None = None


# ── Retrieval Debug ───────────────────────────


class ChunkScore(BaseModel):
    """A chunk with scores from each retrieval stage."""

    chunk_id: str
    content: str
    document_name: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    element_type: str | None = None
    page_number: int | None = None


class RetrieveRequest(BaseModel):
    """Debug retrieval request — returns chunks + scores, no generation."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    use_reranker: bool = True
    use_hybrid: bool = True
    query_transform: str = Field(default="none", description="none | rewrite | hyde | decompose")
    rrf_k: int = Field(default=60, ge=10, le=200, description="RRF constant")
    filters: RetrievalFilters | None = Field(
        default=None, description="Metadata pre-filters for staged hybrid retrieval"
    )


class RetrieveResponse(BaseModel):
    """Debug retrieval response with per-stage scores."""

    chunks: list[ChunkScore]
    query_transformed: str | None = None
    latency_ms: float


# ── Feedback ──────────────────────────────────


class FeedbackRequest(BaseModel):
    """User feedback on a query answer.

    Provide either query_id or trace_id (trace_id is resolved to the query
    that produced it). rating: -1 (down), 0 (neutral), 1 (up).
    """

    query_id: str | None = None
    trace_id: str | None = None
    rating: int = Field(..., ge=-1, le=1, description="-1 (down), 0 (neutral), 1 (up)")
    correction: str | None = None


class FeedbackResponse(BaseModel):
    """Confirmation of feedback submission."""

    id: str
    query_id: str
    message: str


class FeedbackItem(BaseModel):
    """A stored feedback record."""

    id: str
    query_id: str
    rating: int
    correction: str | None = None
    created_at: str


# ── Health ────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: str


# ── Evaluation ────────────────────────────────


class DatasetCreateRequest(BaseModel):
    """Request to create/generate an evaluation dataset."""

    name: str = Field(..., min_length=1, max_length=200)
    version: str = Field(default="1.0")
    document_ids: list[str] | None = None
    num_questions: int = Field(default=20, ge=5, le=100)


class DatasetUploadRequest(BaseModel):
    """Request to upload a pre-built evaluation dataset."""

    name: str = Field(..., min_length=1, max_length=200)
    version: str = Field(default="1.0")
    description: str | None = None
    samples: list[dict] = Field(
        ...,
        description="List of {question, gold_answer, gold_contexts, metadata}",
    )


class DatasetResponse(BaseModel):
    """Response for a dataset."""

    id: str
    name: str
    version: str
    description: str | None = None
    sample_count: int
    created_at: str


class DatasetDetailResponse(DatasetResponse):
    """Detailed dataset response including samples."""

    samples: list[dict]
    source_documents: list[str] | None = None


class EvalRunRequest(BaseModel):
    """Request to trigger an evaluation run."""

    dataset_id: str
    run_name: str | None = None
    use_llm_judge: bool = Field(default=True, description="Enable LLM-as-Judge scoring")
    config_overrides: dict | None = Field(
        default=None,
        description="Override config: top_k, rerank_top_k, rrf_k, "
        "use_reranker, use_hybrid, query_transform",
    )


class EvalMetricSummary(BaseModel):
    """Summary of a single metric with threshold check."""

    value: float
    threshold: float | None = None
    passed: bool | None = None


class EvalRunResponse(BaseModel):
    """Response for an evaluation run."""

    id: str
    dataset_id: str
    name: str | None = None
    status: str
    config_snapshot: dict | None = None
    metrics: dict | None = None
    total_samples: int | None = None
    passed_samples: int | None = None
    failed_samples: int | None = None
    total_latency_ms: float | None = None
    created_at: str
    completed_at: str | None = None


class EvalSampleResponse(BaseModel):
    """Per-sample evaluation result."""

    sample_index: int
    question: str
    gold_answer: str | None = None
    generated_answer: str | None = None
    metrics: dict | None = None
    judge_reasoning: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None


class EvalRunDetailResponse(EvalRunResponse):
    """Detailed eval run with per-sample results."""

    samples: list[EvalSampleResponse] = []


class RegressionGateResult(BaseModel):
    """CI regression gate result — pass/fail with metric details."""

    passed: bool
    thresholds: dict[str, float]
    actual: dict[str, float]
    failures: list[str]


# ── Experiments (A/B) ─────────────────────────


class ExperimentCreateRequest(BaseModel):
    """Create + run an A/B experiment comparing two pipeline configs."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    dataset_id: str
    config_a: dict = Field(default_factory=dict, description="Variant A config overrides")
    config_b: dict = Field(default_factory=dict, description="Variant B config overrides")
    use_llm_judge: bool = Field(default=False, description="Enable LLM judge for both runs")


class ExperimentResponse(BaseModel):
    """An experiment with its variant metrics and deltas."""

    id: str
    name: str
    description: str | None = None
    dataset_id: str
    status: str
    config_a: dict | None = None
    config_b: dict | None = None
    run_a_id: str | None = None
    run_b_id: str | None = None
    metrics_a: dict | None = None
    metrics_b: dict | None = None
    deltas: dict | None = None
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None


# ── Traces & Spans ────────────────────────────


class SpanResponse(BaseModel):
    """A single span within a trace."""

    id: str
    otel_span_id: str
    parent_span_id: str | None = None
    span_kind: str
    name: str
    status: str
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: float | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    attributes: dict | None = None


class TraceSummary(BaseModel):
    """Trace list-view summary."""

    id: str
    otel_trace_id: str
    query_id: str | None = None
    name: str | None = None
    status: str
    total_duration_ms: float | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    span_count: int
    created_at: str


class TraceDetail(TraceSummary):
    """Full trace with its span tree."""

    spans: list[SpanResponse] = []


# ── Analytics ─────────────────────────────────


class Percentiles(BaseModel):
    """Latency percentiles in milliseconds."""

    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    count: int = 0


class LatencyPoint(BaseModel):
    bucket: str
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    count: int = 0


class LatencyAnalytics(BaseModel):
    overall: Percentiles
    series: list[LatencyPoint] = []


class ModelCost(BaseModel):
    model: str
    queries: int
    tokens: int
    cost_usd: float


class CostPoint(BaseModel):
    bucket: str
    cost_usd: float
    queries: int


class CostAnalytics(BaseModel):
    total_cost_usd: float
    total_tokens: int
    avg_cost_per_query: float
    by_model: list[ModelCost] = []
    series: list[CostPoint] = []


class CacheAnalytics(BaseModel):
    entries: int
    hits: int
    misses: int
    hit_rate: float
    threshold: float
    ttl_seconds: int
    estimated_savings_usd: float


class ThroughputPoint(BaseModel):
    bucket: str
    count: int
    qps: float


class ThroughputAnalytics(BaseModel):
    total_queries: int
    avg_qps: float
    peak_qps: float
    series: list[ThroughputPoint] = []
