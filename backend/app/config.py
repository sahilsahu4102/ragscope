"""
RAGScope — Application Configuration

Pydantic Settings with .env validation, organized by service domain.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────
    app_name: str = "RAGScope"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Database ──────────────────────────────
    postgres_user: str = "ragscope"
    postgres_password: str = "changeme_in_production"
    postgres_db: str = "ragscope"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis ─────────────────────────────────
    redis_url: str = "redis://redis:6379/0"

    # ── Ollama (Self-Hosted LLM) ──────────────
    ollama_base_url: str = "http://ollama:11434"
    # llama3.2:3b (~2GB) fits fully in 4GB VRAM. llama3.1:8b needs 5.6GB and gets
    # split ~58% onto CPU, which is what pushed generation to 27-42s per call.
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_keep_alive: str = Field(
        default="-1",
        description="Ollama keep_alive: seconds as an integer ('-1' = never unload), "
        "or a duration with a unit ('10m'). The 5m default caused ~3.8s cold-start "
        "reloads on the query path.",
    )

    @property
    def ollama_keep_alive_value(self) -> int | str:
        """keep_alive coerced to what Ollama's JSON API actually accepts.

        Ollama parses a *string* as a Go duration, so a bare "-1" fails with
        'missing unit in duration'. Plain integers must be sent as JSON numbers;
        unit-suffixed values ("10m") stay strings.
        """
        try:
            return int(self.ollama_keep_alive)
        except ValueError:
            return self.ollama_keep_alive

    # ── Embedding Registry ────────────────────
    default_embedding_provider: str = Field(
        default="ollama",
        description="Options: 'ollama', 'gemini', 'bge-m3'",
    )
    gemini_api_key: str = ""
    gemini_embedding_model: str = "text-embedding-004"

    # ── Ingestion ─────────────────────────────
    upload_dir: str = "/app/uploads"
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50
    default_chunker: str = Field(
        default="recursive",
        description="Options: 'recursive', 'semantic', 'hierarchical'",
    )

    # ── Retrieval ─────────────────────────────
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    semantic_cache_threshold: float = 0.85

    hnsw_ef_search: int = Field(
        default=40,
        ge=1,
        description=(
            "HNSW search breadth. Higher = better recall, slower. Measured at "
            "100k vectors: ef=40 gives recall@10 of 1.000, so the default is "
            "not accuracy-limited here. Must be >= the number of rows fetched."
        ),
    )

    # ── Generation ────────────────────────────
    generation_num_predict: int = Field(
        default=2048,
        ge=1,
        description=(
            "Max tokens generated per answer. Caps worst-case latency: generation "
            "dominates end-to-end time, so this is the largest single lever."
        ),
    )
    generation_max_chunk_chars: int = Field(
        default=400,
        ge=0,
        description=(
            "Truncate each retrieved chunk to this many characters before it "
            "enters the prompt. 0 = no truncation. Measured (app/scripts/"
            "latency_sweep.py): with top_k=3 this took total p50 from 5,032ms "
            "to 4,840ms at unchanged retrieval metrics."
        ),
    )

    # ── Sparse retrieval ──────────────────────
    sparse_backend: str = Field(
        default="bm25",
        description=(
            "'bm25' (in-process rank-bm25) or 'postgres_fts' (GIN-indexed "
            "tsvector + ts_rank_cd). bm25 is the default because it measured "
            "~2.2x faster at every corpus size tested — 6.4 vs 15.1ms at 5k, "
            "45 vs 104ms at 25k, 189 vs 423ms at 100k (app/scripts/"
            "sparse_scale.py). postgres_fts avoids bm25's per-worker memory "
            "and its rebuild-on-ingest stall, so it is the better choice under "
            "multiple workers or frequent ingestion — but not for latency."
        ),
    )

    # ── Reranking ─────────────────────────────
    reranker_backend: str = Field(
        default="cross_encoder",
        description="'cross_encoder' (fast, local ONNX-able) or 'ollama' (LLM-scored, slow)",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description=(
            "Cross-encoder checkpoint. MiniLM-L-6 is 22M params (~30-60ms for 20 pairs "
            "on CPU). BAAI/bge-reranker-v2-m3 is 568M and ~50x slower — higher quality "
            "but far outside a real-time latency budget."
        ),
    )

    # ── Evaluation ───────────────────────────────
    eval_faithfulness_threshold: float = 0.80
    eval_context_recall_threshold: float = 0.70
    eval_context_precision_threshold: float = 0.60
    eval_judge_model: str = ""  # Empty = use ollama_model

    # ── Observability / Tracing ──────────────────
    trace_sampling_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of traces persisted to Postgres (1.0 = all)",
    )
    trace_console_export: bool = Field(
        default=False,
        description="Also print spans to stdout (dev debugging)",
    )
    otlp_endpoint: str = Field(
        default="",
        description="Optional OTLP HTTP endpoint to forward spans to",
    )

    # ── Guardrails (Phase 5) ─────────────────────
    enable_pii_redaction: bool = Field(
        default=True,
        description="Redact PII (email, phone, SSN, CC, IP) from queries and answers",
    )
    enable_injection_detection: bool = Field(
        default=True,
        description="Detect and block prompt injection attempts",
    )
    enable_hallucination_detection: bool = Field(
        default=True,
        description="Score answer groundedness and flag hallucinations",
    )
    injection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for blocking injections (0-1)",
    )
    hallucination_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Groundedness threshold below which answers are flagged (0-1)",
    )

    # ── Connection Pool Tuning ────────────────────
    db_pool_size: int = Field(default=10, description="SQLAlchemy pool_size")
    db_max_overflow: int = Field(default=20, description="SQLAlchemy max_overflow")
    db_pool_recycle: int = Field(
        default=1800,
        description="Recycle connections after N seconds (prevents stale connections)",
    )


settings = Settings()
