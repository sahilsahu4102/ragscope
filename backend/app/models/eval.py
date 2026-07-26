"""
RAGScope — Evaluation Models

Database models for the evaluation harness:
- Dataset: versioned golden QA sets
- EvalRun: a full evaluation run with config snapshot + aggregate metrics
- EvalSample: per-question results within an eval run

Declared with SQLAlchemy 2.0 `Mapped[...]` / `mapped_column(...)`. The schema
is identical to the legacy `Column()` form; the annotations just let type
checkers see `run.status` as `str` rather than `Column[str]`.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Dataset(Base):
    """A versioned golden evaluation dataset (collection of QA pairs)."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    samples: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of {question, gold_answer, gold_contexts, metadata}",
    )
    source_documents: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Document IDs used to generate this dataset",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # Relationships
    eval_runs: Mapped[list["EvalRun"]] = relationship(
        "EvalRun", back_populates="dataset", cascade="all, delete-orphan"
    )


class EvalRun(Base):
    """A complete evaluation run — metrics computed over a dataset."""

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Human-readable run name"
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        comment="pending | running | completed | failed",
    )

    # ── Config snapshot (what settings were used) ─────
    config_snapshot: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Full retrieval/generation config: embedding model, chunk size, "
        "reranker, top_k, rrf_k, query_transform, etc.",
    )

    # ── Aggregate metrics ─────────────────────────────
    metrics: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Aggregate metrics: faithfulness, context_recall, "
        "context_precision, ndcg, mrr, answer_relevance, hallucination_rate",
    )

    # ── Run metadata ──────────────────────────────────
    total_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passed_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failed_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="eval_runs")
    samples: Mapped[list["EvalSample"]] = relationship(
        "EvalSample", back_populates="eval_run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_eval_runs_dataset_id", "dataset_id"),)


class Experiment(Base):
    """An A/B experiment comparing two configs on the same dataset.

    Runs the eval pipeline once per config (variant A vs variant B) and stores
    the per-metric deltas with a heuristic significance flag.
    """

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        comment="pending | running | completed | failed",
    )

    # ── Variant configs (config_overrides for the eval runner) ────
    config_a: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Variant A config"
    )
    config_b: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Variant B config"
    )

    # ── Links to the two eval runs produced ──────────────────────
    run_a_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    run_b_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # ── Results ──────────────────────────────────────────────────
    metrics_a: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Aggregate metrics for variant A"
    )
    metrics_b: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Aggregate metrics for variant B"
    )
    deltas: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Per-metric {delta, pct_change, winner, significant} objects",
    )

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (Index("ix_experiments_dataset_id", "dataset_id"),)


class EvalSample(Base):
    """Per-question results within an eval run."""

    __tablename__ = "eval_samples"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Input ─────────────────────────────────────────
    question: Mapped[str] = mapped_column(Text, nullable=False)
    gold_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    gold_contexts: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, comment="Reference context passages"
    )

    # ── Pipeline output ───────────────────────────────
    generated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunks: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Retrieved chunks with scores: [{chunk_id, content, score, ...}]",
    )
    retrieval_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    generation_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Per-sample metrics ────────────────────────────
    metrics: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Per-sample: faithfulness, context_precision, context_recall, "
        "answer_relevance, answer_correctness, hallucination_score",
    )

    # ── LLM Judge output ─────────────────────────────
    judge_reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="LLM judge chain-of-thought"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # Relationships
    eval_run: Mapped["EvalRun"] = relationship("EvalRun", back_populates="samples")

    __table_args__ = (Index("ix_eval_samples_run_id", "eval_run_id"),)
