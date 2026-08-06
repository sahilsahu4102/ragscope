"""
RAGScope — Ingest API Router

Endpoints for document upload and ingestion job status.
"""

import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.models import Document
from app.schemas.schemas import IngestResponse, IngestStatusResponse
from app.workers.ingest_task import ingest_document_task

logger = structlog.get_logger()
router = APIRouter(prefix="/ingest", tags=["ingestion"])


def _get_upload_dir() -> Path:
    """Lazily create and return the upload directory."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document for ingestion.

    Accepts PDF files, saves to disk, and dispatches a Celery task
    for async processing (parse → chunk → embed → store).

    Returns a job_id (Celery task ID) and document_id for tracking.
    """
    # Validate file type
    allowed_types = {
        "application/pdf",
        "text/plain",
        "text/markdown",
    }
    content_type = file.content_type or "application/octet-stream"
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {allowed_types}",
        )

    # Generate IDs. The extension comes from our own UUID plus a suffix taken
    # from the upload — Path().suffix strips any directory component, so a
    # filename like "../../etc/passwd" cannot escape the upload directory.
    document_id = uuid.uuid4()
    file_ext = Path(file.filename or "document").suffix or ".pdf"
    save_path = _get_upload_dir() / f"{document_id}{file_ext}"

    # Stream to disk in chunks with a hard cap. Reading the whole upload into
    # memory first makes a large file an out-of-memory vector, and Content-Length
    # cannot be trusted because a client can lie about it.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    f.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_mb} MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error("Failed to save upload", document_id=str(document_id), error=str(e))
        # Detail is not echoed: it can contain server filesystem paths.
        raise HTTPException(status_code=500, detail="Failed to save file")

    if written == 0:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Create initial document record
    document = Document(
        id=document_id,
        filename=file.filename or "document",
        mime_type=content_type,
        status="pending",
        file_size_bytes=written,
    )
    db.add(document)
    await db.commit()

    # Dispatch Celery task
    task = ingest_document_task.delay(
        file_path=str(save_path),
        document_id=str(document_id),
    )

    # Invalidate BM25 cache so next hybrid query rebuilds with new docs
    from app.retrieval.sparse import invalidate_bm25_cache

    invalidate_bm25_cache()

    logger.info(
        "Ingestion job dispatched",
        document_id=str(document_id),
        job_id=task.id,
        filename=file.filename,
    )

    return IngestResponse(
        job_id=task.id,
        document_id=str(document_id),
        status="pending",
        message=f"Document '{file.filename}' queued for ingestion",
    )


@router.get("/{job_id}", response_model=IngestStatusResponse)
async def get_ingest_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Check the status of an ingestion job.

    Returns the Celery task status and document details.
    """
    from celery.result import AsyncResult

    result = AsyncResult(job_id)

    # Get document info from result if available
    if result.successful():
        task_result = result.result
        doc_id = task_result.get("document_id", "")
        status = task_result.get("status", "completed")

        # Count chunks created
        from app.models import Chunk

        chunk_count_result = await db.execute(
            select(Chunk).where(Chunk.document_id == uuid.UUID(doc_id))
        )
        chunks_created = len(chunk_count_result.scalars().all())

        return IngestStatusResponse(
            job_id=job_id,
            document_id=doc_id,
            status=status,
            chunks_created=chunks_created,
        )
    elif result.failed():
        return IngestStatusResponse(
            job_id=job_id,
            document_id="",
            status="failed",
            error=str(result.result),
        )
    else:
        return IngestStatusResponse(
            job_id=job_id,
            document_id="",
            status=result.status.lower(),
        )
