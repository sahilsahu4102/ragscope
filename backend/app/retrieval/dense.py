"""
RAGScope — Dense Retrieval

Cosine similarity search over pgvector embeddings.
Returns ranked chunks with scores, traced with OpenInference RETRIEVER span.
"""

import structlog
from uuid import UUID

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.ingestion.embedders.embedder import get_embedder
from app.observability.tracer import get_tracer, create_span

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


class DenseRetriever:
    """
    Dense vector retrieval using pgvector cosine similarity.
    
    Queries the chunks table with `<=>` (cosine distance) operator,
    returns top-k chunks ranked by similarity score.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = get_embedder()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve top-k chunks most similar to the query.
        
        Returns list of dicts with: chunk_id, content, score, document_name, metadata.
        """
        top_k = top_k or settings.retrieval_top_k

        with create_span(tracer, "dense_retrieve", "RETRIEVER", {
            "retriever.type": "dense",
            "retriever.top_k": top_k,
            "retriever.query": query,
        }):
            # 1. Embed the query
            with create_span(tracer, "embed_query", "EMBEDDING", {
                "embedding.model_name": self.embedder.model_name(),
            }):
                query_embeddings = await self.embedder.embed([query])
                query_vector = query_embeddings[0]

            # 2. pgvector cosine similarity search
            vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

            sql = text("""
                SELECT 
                    c.id,
                    c.content,
                    c.element_type,
                    c.metadata,
                    c.chunk_index,
                    c.token_count,
                    d.filename as document_name,
                    1 - (c.embedding <=> :query_vector::vector) as score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> :query_vector::vector
                LIMIT :top_k
            """)

            result = await self.db.execute(
                sql,
                {"query_vector": vector_str, "top_k": top_k},
            )
            rows = result.fetchall()

            chunks = []
            for row in rows:
                chunks.append({
                    "chunk_id": str(row.id),
                    "content": row.content,
                    "document_name": row.document_name,
                    "dense_score": float(row.score),
                    "element_type": row.element_type,
                    "chunk_index": row.chunk_index,
                    "token_count": row.token_count,
                    "metadata": row.metadata or {},
                })

            logger.info(
                "Dense retrieval complete",
                query_length=len(query),
                results=len(chunks),
                top_score=chunks[0]["dense_score"] if chunks else 0,
            )

            return chunks
