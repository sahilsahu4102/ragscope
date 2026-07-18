"""
RAGScope — Celery Ingestion Task

Async ingestion task dispatched via Celery + Redis.
Accepts file path and document_id, runs the full ingestion pipeline.
"""

import asyncio
import uuid
import structlog

from app.workers.celery_app import celery_app

logger = structlog.get_logger()


@celery_app.task(
    bind=True,
    name="ingest_document",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def ingest_document_task(self, file_path: str, document_id: str) -> dict:
    """
    Celery task: ingest a document through the full pipeline.
    
    This runs in a Celery worker process, so we bootstrap an async event loop.
    """
    logger.info(
        "Celery ingestion task started",
        task_id=self.request.id,
        document_id=document_id,
        file_path=file_path,
    )

    try:
        result = asyncio.run(_run_ingestion(file_path, document_id))
        return result
    except Exception as exc:
        logger.error(
            "Celery ingestion task failed",
            task_id=self.request.id,
            error=str(exc),
        )
        raise self.retry(exc=exc)


async def _run_ingestion(file_path: str, document_id: str) -> dict:
    """Run ingestion pipeline inside an async context."""
    from app.db.session import async_session
    from app.ingestion.pipeline import IngestionPipeline

    async with async_session() as db:
        pipeline = IngestionPipeline(db)
        doc = await pipeline.ingest(
            file_path=file_path,
            document_id=uuid.UUID(document_id),
        )

        return {
            "document_id": str(doc.id),
            "filename": doc.filename,
            "status": doc.status,
            "page_count": doc.page_count,
        }
