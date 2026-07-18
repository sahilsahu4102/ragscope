"""
RAGScope — SQLAlchemy Models

Core data models: Document, Chunk, Query, Feedback.
All tables use UUID primary keys for distributed-safe IDs.
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """An ingested source document (PDF, DOCX, HTML, etc.)."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(500), nullable=False)
    source_url = Column(String(2000), nullable=True)
    mime_type = Column(String(100), nullable=False)
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        comment="pending | processing | completed | failed",
    )
    file_size_bytes = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True, default=dict)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """A text chunk derived from a document, with its embedding vector."""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_id = Column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True,
        comment="Parent chunk ID for hierarchical chunking",
    )
    content = Column(Text, nullable=False)
    contextual_summary = Column(
        Text, nullable=True,
        comment="LLM-generated context prepended before embedding (Anthropic pattern)",
    )
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding = Column(Vector(768), nullable=True)
    embedding_model = Column(String(100), nullable=True)
    element_type = Column(
        String(50), nullable=True,
        comment="title | paragraph | table | list | image_caption",
    )
    metadata_ = Column("metadata", JSONB, nullable=True, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="chunks")
    children = relationship("Chunk", backref="parent", remote_side=[id])

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_parent_id", "parent_id"),
    )


class Query(Base):
    """A user query and its generated answer."""

    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    citations = Column(JSONB, nullable=True, comment="Structured citation objects")
    trace_id = Column(String(64), nullable=True, index=True)
    retrieval_scores = Column(
        JSONB, nullable=True,
        comment="Per-chunk scores: dense, sparse, rrf, rerank",
    )
    latency_ms = Column(Float, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    total_cost_usd = Column(Float, nullable=True)
    config_snapshot = Column(JSONB, nullable=True, comment="Retrieval/gen config used")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    feedback = relationship("Feedback", back_populates="query", cascade="all, delete-orphan")


class Feedback(Base):
    """User feedback (thumbs up/down + optional correction) on a query answer."""

    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(
        UUID(as_uuid=True), ForeignKey("queries.id", ondelete="CASCADE"), nullable=False
    )
    rating = Column(Integer, nullable=False, comment="-1 (down) or 1 (up)")
    correction = Column(Text, nullable=True, comment="User-provided correct answer")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    query = relationship("Query", back_populates="feedback")
