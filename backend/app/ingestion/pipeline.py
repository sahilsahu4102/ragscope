"""
RAGScope — Ingestion Pipeline Orchestrator

Coordinates the full ingestion flow: parse → chunk → embed → store in pgvector.
Wrapped in OpenInference CHAIN span for end-to-end tracing.
"""

import uuid
from pathlib import Path

import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.ingestion.chunkers.recursive_chunker import RecursiveChunker
from app.ingestion.embedders.embedder import get_embedder
from app.ingestion.parsers.pdf_parser import PDFParser
from app.models import Chunk, Document
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("ingestion")


class IngestionPipeline:
    """
    End-to-end document ingestion: parse → chunk → embed → store.

    Each step is traced with OpenInference spans for observability from day one.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.parser = PDFParser()
        self.chunker = RecursiveChunker(
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
        )
        self.embedder = get_embedder()

    async def ingest(
        self,
        file_path: str,
        document_id: uuid.UUID | None = None,
    ) -> Document:
        """
        Ingest a document end-to-end.

        Returns the Document record with all chunks created and embedded.
        """
        path = Path(file_path)
        doc_id = document_id or uuid.uuid4()

        with create_span(
            tracer,
            "ingestion_pipeline",
            "CHAIN",
            {
                "document.filename": path.name,
                "document.id": str(doc_id),
            },
        ):
            # ── 1. Adopt or create the Document record ─
            # The ingest API pre-creates a row (status="pending") before it
            # dispatches the Celery job, so inserting the same id here would
            # violate the primary key. Adopt the existing row when there is one;
            # only insert when the pipeline is driven directly (tests, scripts).
            document = await self.db.get(Document, doc_id)
            if document is None:
                document = Document(
                    id=doc_id,
                    filename=path.name,
                    mime_type=self._detect_mime(path),
                    file_size_bytes=path.stat().st_size,
                )
                self.db.add(document)
            document.status = "processing"
            # Commit, not flush — an uncommitted status is invisible to the API
            # session, so a polling client would sit on "pending" for the whole
            # run and then jump straight to "completed".
            await self.db.commit()

            try:
                # ── 2. Parse ──────────────────────────
                with create_span(
                    tracer,
                    "parse_document",
                    "CHAIN",
                    {
                        "parser.type": "pdf_layout",
                    },
                ):
                    parsed_doc = await self.parser.parse(str(path), document.mime_type)
                    document.page_count = parsed_doc.page_count

                # ── 3. Chunk ──────────────────────────
                with create_span(
                    tracer,
                    "chunk_document",
                    "CHAIN",
                    {
                        "chunker.type": "recursive",
                        "chunker.chunk_size": settings.default_chunk_size,
                        "chunker.chunk_overlap": settings.default_chunk_overlap,
                    },
                ):
                    text_chunks = self.chunker.chunk_document(parsed_doc)

                if not text_chunks:
                    document.status = "failed"
                    document.error_message = "No text chunks extracted"
                    await self.db.commit()
                    return document

                # ── 4. Embed ──────────────────────────
                with create_span(
                    tracer,
                    "embed_chunks",
                    "EMBEDDING",
                    {
                        "embedding.model_name": self.embedder.model_name(),
                        "embedding.dimension": self.embedder.dimension(),
                        "embedding.chunk_count": len(text_chunks),
                    },
                ):
                    texts_to_embed = [chunk.content for chunk in text_chunks]
                    embeddings = await self.embedder.embed(texts_to_embed)

                # ── 5. Store in pgvector ──────────────
                with create_span(
                    tracer,
                    "store_chunks",
                    "CHAIN",
                    {
                        "store.chunk_count": len(text_chunks),
                    },
                ):
                    for chunk_data, embedding in zip(text_chunks, embeddings, strict=True):
                        chunk = Chunk(
                            document_id=doc_id,
                            content=chunk_data.content,
                            chunk_index=chunk_data.chunk_index,
                            token_count=chunk_data.token_count,
                            embedding=embedding,
                            embedding_model=self.embedder.model_name(),
                            element_type=chunk_data.element_type,
                            metadata_={
                                "page_number": chunk_data.page_number,
                                "section_path": chunk_data.section_path,
                                **chunk_data.metadata,
                            },
                        )
                        self.db.add(chunk)

                document.status = "completed"
                await self.db.commit()

                logger.info(
                    "Ingestion complete",
                    document_id=str(doc_id),
                    filename=path.name,
                    chunks_created=len(text_chunks),
                    embedding_model=self.embedder.model_name(),
                )

                return document

            except Exception as e:
                logger.error("Ingestion failed", document_id=str(doc_id), error=str(e))
                # A DB-level failure leaves the session unusable, so roll back
                # before recording the status or the write fails too and the
                # document is stranded at "pending" forever. The UPDATE goes
                # through Core rather than the (now expired) ORM instance.
                await self.db.rollback()
                try:
                    await self.db.execute(
                        update(Document)
                        .where(Document.id == doc_id)
                        .values(status="failed", error_message=str(e)[:2000])
                    )
                    await self.db.commit()
                except Exception:  # pragma: no cover — best-effort status write
                    logger.warning("Could not record failure status", document_id=str(doc_id))
                raise

    @staticmethod
    def _detect_mime(path: Path) -> str:
        """Detect MIME type from file extension."""
        suffix_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".html": "text/html",
            ".htm": "text/html",
            ".txt": "text/plain",
            ".md": "text/markdown",
        }
        return suffix_map.get(path.suffix.lower(), "application/octet-stream")
