"""RAGScope — Embedders package."""

from app.ingestion.embedders.embedder import get_embedder, OllamaEmbedder, GeminiEmbedder, BaseEmbedder

__all__ = ["get_embedder", "OllamaEmbedder", "GeminiEmbedder", "BaseEmbedder"]
