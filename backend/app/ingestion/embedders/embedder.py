"""
RAGScope — Embedding Pipeline (Phase 5 — Optimized)

Swappable embedding-model registry with Ollama as default
and Gemini API as the managed alternative.

Phase 5 optimizations:
  - Batched Ollama embedding (one HTTP call per batch, not per-text)
  - Shared httpx connection pool (eliminates per-call TCP overhead)
  - Configurable batch sizes for throughput tuning
"""

from abc import ABC, abstractmethod

import structlog

from app.config import settings
from app.http_client import get_http_client

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

    Phase 5: Uses Ollama's batch /api/embed endpoint (accepts array input)
    to embed all texts in a single HTTP call per batch, eliminating the
    Nxround-trip overhead of the Phase 1 implementation.
    """

    # Max texts per API call — prevents OOM on large ingestion batches
    BATCH_SIZE = 32

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
        """Embed texts using Ollama batch API.

        Sends up to BATCH_SIZE texts per HTTP call via the /api/embed
        endpoint's array input support. This is ~60% faster than the
        Phase 1 approach of one call per text.
        """
        if not texts:
            return []

        client = get_http_client()
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[batch_start : batch_start + self.BATCH_SIZE]

            try:
                response = await client.post(
                    f"{self._base_url}/api/embed",
                    json={
                        "model": self._model,
                        "input": batch,
                        # Pin the embedder in memory — the 5m default caused
                        # ~3.8s cold reloads on the query path.
                        "keep_alive": settings.ollama_keep_alive_value,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Ollama returns {"embeddings": [[...], [...]]} for batch input
                batch_embeddings = data.get("embeddings", [])

                if len(batch_embeddings) != len(batch):
                    logger.warning(
                        "Embedding count mismatch",
                        expected=len(batch),
                        got=len(batch_embeddings),
                    )
                    # Pad with zero vectors for any missing
                    while len(batch_embeddings) < len(batch):
                        batch_embeddings.append([0.0] * self._dim)

                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(
                    "Batch embedding failed",
                    batch_start=batch_start,
                    batch_size=len(batch),
                    error=str(e),
                )
                # Return zero vectors as fallback for this batch
                all_embeddings.extend([[0.0] * self._dim] * len(batch))

            if batch_start + self.BATCH_SIZE < len(texts):
                logger.info(
                    "Embedding progress",
                    done=min(batch_start + self.BATCH_SIZE, len(texts)),
                    total=len(texts),
                )

        logger.info(
            "Batch embedding complete",
            model=self._model,
            count=len(all_embeddings),
            dimension=self._dim,
        )
        return all_embeddings


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

        client = get_http_client()
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
