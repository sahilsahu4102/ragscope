"""
RAGScope — Evaluation Models

Database models for the evaluation harness:
- Dataset: versioned golden QA sets
- EvalRun: a full evaluation run with config snapshot + aggregate metrics
- EvalSample: per-question results within an eval run
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Dataset(Base):
    """A versioned golden evaluation dataset (collection of QA pairs)."""

    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False, default="1.0")
    description = Column(Text, nullable=True)
    sample_count = Column(Integer, nullable=False, default=0)
    samples = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of {question, gold_answer, gold_contexts, metadata}",
    )
    source_documents = Column(
        JSONB,
        nullable=True,
        comment="Document IDs used to generate this dataset",
    )
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    eval_runs = relationship("EvalRun", back_populates="dataset", cascade="all, delete-orphan")


class EvalRun(Base):
    """A complete evaluation run — metrics computed over a dataset."""

    __tablename__ = "eval_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    name = Column(String(200), nullable=True, comment="Human-readable run name")
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        comment="pending | running | completed | failed",
    )

    # ── Config snapshot (what settings were used) ─────
    config_snapshot = Column(
        JSONB,
        nullable=True,
        comment="Full retrieval/generation config: embedding model, chunk size, "
        "reranker, top_k, rrf_k, query_transform, etc.",
    )

    # ── Aggregate metrics ─────────────────────────────
    metrics = Column(
        JSONB,
        nullable=True,
        comment="Aggregate metrics: faithfulness, context_recall, "
        "context_precision, ndcg, mrr, answer_relevance, hallucination_rate",
    )

    # ── Run metadata ──────────────────────────────────
    total_samples = Column(Integer, nullable=True)
    passed_samples = Column(Integer, nullable=True)
    failed_samples = Column(Integer, nullable=True)
    total_latency_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    dataset = relationship("Dataset", back_populates="eval_runs")
    samples = relationship("EvalSample", back_populates="eval_run", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_eval_runs_dataset_id", "dataset_id"),)


class EvalSample(Base):
    """Per-question results within an eval run."""

    __tablename__ = "eval_samples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    eval_run_id = Column(
        UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    sample_index = Column(Integer, nullable=False)

    # ── Input ─────────────────────────────────────────
    question = Column(Text, nullable=False)
    gold_answer = Column(Text, nullable=True)
    gold_contexts = Column(JSONB, nullable=True, comment="Reference context passages")

    # ── Pipeline output ───────────────────────────────
    generated_answer = Column(Text, nullable=True)
    retrieved_chunks = Column(
        JSONB,
        nullable=True,
        comment="Retrieved chunks with scores: [{chunk_id, content, score, ...}]",
    )
    retrieval_latency_ms = Column(Float, nullable=True)
    generation_latency_ms = Column(Float, nullable=True)

    # ── Per-sample metrics ────────────────────────────
    metrics = Column(
        JSONB,
        nullable=True,
        comment="Per-sample: faithfulness, context_precision, context_recall, "
        "answer_relevance, answer_correctness, hallucination_score",
    )

    # ── LLM Judge output ─────────────────────────────
    judge_reasoning = Column(Text, nullable=True, comment="LLM judge chain-of-thought")

    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    eval_run = relationship("EvalRun", back_populates="samples")

    __table_args__ = (Index("ix_eval_samples_run_id", "eval_run_id"),)
