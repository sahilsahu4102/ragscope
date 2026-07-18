"""
RAGScope — Embedding Pipeline

Swappable embedding-model registry with Ollama as default
and Gemini API as the managed alternative.
"""

import structlog
from abc import ABC, abstractmethod

import httpx
import numpy as np

from app.config import settings

logger = structlog.get_logger()


class BaseEmbedder(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, return list of vectors."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...


class OllamaEmbedder(BaseEmbedder):
    """
    Embedder using self-hosted Ollama.
    
    Default model: nomic-embed-text (768 dims, strong general-purpose).
    Fully self-hosted — no external API calls.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model or settings.ollama_embedding_model
        self._base_url = base_url or settings.ollama_base_url
        self._dim = 768  # nomic-embed-text default

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Ollama API."""
        embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, text in enumerate(texts):
                try:
                    response = await client.post(
                        f"{self._base_url}/api/embed",
                        json={"model": self._model, "input": text},
                    )
                    response.raise_for_status()
                    data = response.json()

                    # Ollama returns {"embeddings": [[...]]} for /api/embed
                    embedding = data["embeddings"][0]
                    embeddings.append(embedding)

                    if (i + 1) % 50 == 0:
                        logger.info("Embedding progress", done=i + 1, total=len(texts))

                except Exception as e:
                    logger.error("Embedding failed", text_index=i, error=str(e))
                    # Return zero vector as fallback
                    embeddings.append([0.0] * self._dim)

        logger.info(
            "Batch embedding complete",
            model=self._model,
            count=len(embeddings),
            dimension=self._dim,
        )
        return embeddings


class GeminiEmbedder(BaseEmbedder):
    """
    Embedder using Google Gemini Embedding API.
    
    Model: text-embedding-004 (~768 dims, Matryoshka support).
    Requires GEMINI_API_KEY in config.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self._model = model or settings.gemini_embedding_model
        self._api_key = api_key or settings.gemini_api_key
        self._dim = 768

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Gemini API."""
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        embeddings: list[list[float]] = []
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:embedContent?key={self._api_key}"
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                response = await client.post(
                    url,
                    json={
                        "model": f"models/{self._model}",
                        "content": {"parts": [{"text": text}]},
                    },
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["embedding"]["values"]
                embeddings.append(embedding)

        logger.info(
            "Gemini batch embedding complete",
            model=self._model,
            count=len(embeddings),
        )
        return embeddings


def get_embedder(provider: str | None = None) -> BaseEmbedder:
    """
    Embedding model registry — factory function.
    
    Switching models is one config change:
      DEFAULT_EMBEDDING_PROVIDER=ollama  → OllamaEmbedder
      DEFAULT_EMBEDDING_PROVIDER=gemini  → GeminiEmbedder
    """
    provider = provider or settings.default_embedding_provider

    match provider.lower():
        case "ollama":
            return OllamaEmbedder()
        case "gemini":
            return GeminiEmbedder()
        case _:
            logger.warning(f"Unknown embedding provider '{provider}', falling back to Ollama")
            return OllamaEmbedder()
