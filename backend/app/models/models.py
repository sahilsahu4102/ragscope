"""
RAGScope — SQLAlchemy Models

Core data models: Document, Chunk, Query, Feedback.
All tables use UUID primary keys for distributed-safe IDs.

Declared with SQLAlchemy 2.0 `Mapped[...]` / `mapped_column(...)`. The schema
is identical to the legacy `Column()` form; the annotations just let type
checkers see `query.latency_ms` as `float | None` rather than `Column[float]`.
"""

import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    """An ingested source document (PDF, DOCX, HTML, etc.)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        comment="pending | processing | completed | failed",
    )
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A text chunk derived from a document, with its embedding vector."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
        comment="Parent chunk ID for hierarchical chunking",
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    contextual_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="LLM-generated context prepended before embedding (Anthropic pattern)",
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    element_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="title | paragraph | table | list | image_caption",
    )
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    # Self-referential many-to-one; `backref` also creates the reverse `parent`
    # collection at runtime. Left unannotated because the backref attribute is
    # generated dynamically and has no static counterpart.
    children = relationship("Chunk", backref="parent", remote_side=[id])

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_parent_id", "parent_id"),
        Index("ix_chunks_element_type", "element_type"),
        Index("ix_chunks_created_at", "created_at"),
        # ANN index for cosine similarity search. Without it every query is a
        # sequential scan over the whole table: measured 13ms at 5.9k chunks
        # but ~230ms at 100k, where HNSW stays in single-digit ms at recall
        # 1.000 (see docs/eval-results.md).
        #
        # m=16 / ef_construction=64 are pgvector's defaults and were what the
        # scaling benchmark measured. Search-time accuracy is tuned separately
        # via hnsw.ef_search in DenseRetriever.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Query(Base):
    """A user query and its generated answer."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list | None] = mapped_column(
        JSONB, nullable=True, comment="Structured citation objects"
    )
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    retrieval_scores: Mapped[list | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Per-chunk scores: dense, sparse, rrf, rerank",
    )
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    config_snapshot: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="Retrieval/gen config used"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # Relationships
    feedback: Mapped[list["Feedback"]] = relationship(
        "Feedback", back_populates="query", cascade="all, delete-orphan"
    )


class Feedback(Base):
    """User feedback (thumbs up/down + optional correction) on a query answer."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False, comment="-1 (down) or 1 (up)")
    correction: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="User-provided correct answer"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    # Relationships
    query: Mapped["Query"] = relationship("Query", back_populates="feedback")
