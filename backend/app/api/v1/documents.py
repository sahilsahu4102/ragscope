"""
RAGScope — Documents API Router

Read and delete access to the ingested-document corpus. Powers the ingest
page's document table and the retrieval filters' document picker — without
this, the only way to see what had been ingested was to query Postgres.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import overload

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete as sql_delete
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.models import Chunk, Document
from app.schemas.schemas import DocumentDeleteResponse, DocumentResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/documents", tags=["documents"])


# Overloaded so non-nullable columns (created_at, updated_at) narrow to `str`
# for the response schema, while nullable ones stay optional.
@overload
def _iso(dt: datetime) -> str: ...
@overload
def _iso(dt: None) -> None: ...
def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _to_response(doc: Document, chunk_count: int) -> DocumentResponse:
    return DocumentResponse(
        id=str(doc.id),
        filename=doc.filename,
        mime_type=doc.mime_type,
        status=doc.status,
        file_size_bytes=doc.file_size_bytes,
        page_count=doc.page_count,
        chunk_count=chunk_count,
        error_message=doc.error_message,
        created_at=_iso(doc.created_at),
        updated_at=_iso(doc.updated_at),
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    status: str | None = Query(
        default=None, description="Filter by pending | processing | completed | failed"
    ),
    limit: int = Query(default=200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """List ingested documents, newest first, with their chunk counts.

    Uses an outer join so documents that are still processing (or failed
    before producing chunks) still show up, with chunk_count = 0.
    """
    stmt = (
        select(Document, func.count(Chunk.id).label("chunk_count"))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
        .limit(limit)
    )
    if status:
        stmt = stmt.where(Document.status == status)

    rows = (await db.execute(stmt)).all()
    return [_to_response(doc, count) for doc, count in rows]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single document with its chunk count."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document id")

    doc = (await db.execute(select(Document).where(Document.id == doc_uuid))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_count = (
        await db.execute(select(func.count(Chunk.id)).where(Chunk.document_id == doc_uuid))
    ).scalar() or 0

    return _to_response(doc, chunk_count)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document, its chunks, and its uploaded file.

    Chunks go via the FK's ON DELETE CASCADE rather than the ORM relationship —
    a large PDF can produce thousands of chunks and we don't want to load them
    all into the session just to delete them.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document id")

    doc = (await db.execute(select(Document).where(Document.id == doc_uuid))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    filename = doc.filename
    chunk_count = (
        await db.execute(select(func.count(Chunk.id)).where(Chunk.document_id == doc_uuid))
    ).scalar() or 0

    await db.execute(sql_delete(Document).where(Document.id == doc_uuid))
    await db.commit()

    # Remove the uploaded source file — it is stored as <document_id><ext>.
    try:
        for stale in Path(settings.upload_dir).glob(f"{doc_uuid}.*"):
            stale.unlink(missing_ok=True)
    except OSError as e:  # pragma: no cover — disk cleanup is best-effort
        logger.warning("Could not delete upload file", document_id=document_id, error=str(e))

    # The BM25 index is built from all chunks, so it is now stale.
    from app.retrieval.sparse import invalidate_bm25_cache

    invalidate_bm25_cache()

    logger.info(
        "Document deleted",
        document_id=document_id,
        filename=filename,
        chunks_deleted=chunk_count,
    )

    return DocumentDeleteResponse(
        id=document_id,
        chunks_deleted=chunk_count,
        message=f"Deleted '{filename}' and {chunk_count} chunks",
    )
