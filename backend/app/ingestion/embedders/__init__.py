"""RAGScope — Embedders package."""

from app.ingestion.embedders.embedder import (
    BaseEmbedder,
    GeminiEmbedder,
    OllamaEmbedder,
    get_embedder,
)

__all__ = ["BaseEmbedder", "GeminiEmbedder", "OllamaEmbedder", "get_embedder"]
