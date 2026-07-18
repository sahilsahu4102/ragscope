"""
RAGScope — Pydantic Schemas

Request/response models for all API endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────

class IngestRequest(BaseModel):
    """Request to ingest a document (URL-based)."""
    source_url: Optional[str] = None
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
    chunks_created: Optional[int] = None
    error: Optional[str] = None


# ── Query ─────────────────────────────────────

class QueryRequest(BaseModel):
    """A user query to the RAG pipeline."""
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = True
    use_hybrid: bool = True
    stream: bool = False


class Citation(BaseModel):
    """A citation linking an answer claim to a source chunk."""
    chunk_id: str
    document_name: str
    content_snippet: str
    score: float
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    """Response from the RAG pipeline."""
    answer: str
    citations: list[Citation]
    trace_id: str
    latency_ms: float
    tokens_used: Optional[int] = None


# ── Retrieval Debug ───────────────────────────

class ChunkScore(BaseModel):
    """A chunk with scores from each retrieval stage."""
    chunk_id: str
    content: str
    document_name: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RetrieveRequest(BaseModel):
    """Debug retrieval request — returns chunks + scores, no generation."""
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    use_reranker: bool = True
    use_hybrid: bool = True


class RetrieveResponse(BaseModel):
    """Debug retrieval response with per-stage scores."""
    chunks: list[ChunkScore]
    query_transformed: Optional[str] = None
    latency_ms: float


# ── Feedback ──────────────────────────────────

class FeedbackRequest(BaseModel):
    """User feedback on a query answer."""
    query_id: str
    rating: int = Field(..., ge=-1, le=1, description="-1 (down), 0 (neutral), 1 (up)")
    correction: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Confirmation of feedback submission."""
    id: str
    message: str


# ── Health ────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
