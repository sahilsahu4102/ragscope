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
    ollama_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"

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


settings = Settings()
