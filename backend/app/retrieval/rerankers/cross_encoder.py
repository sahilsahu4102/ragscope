"""
RAGScope — Cross-Encoder Reranker

Reranks top-N candidates down to top-K using a cross-encoder model.
The single highest-ROI retrieval improvement: +5 to +15 NDCG@10.

Supports:
  - Self-hosted BGE-reranker-v2-m3 (via sentence-transformers)
  - Ollama-based reranking (prompt-based)
"""

from abc import ABC, abstractmethod

import structlog

from app.config import settings
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank chunks by relevance to query. Returns top_k with rerank_score."""
        ...

    @abstractmethod
    def model_name(self) -> str: ...


class OllamaReranker(BaseReranker):
    """
    Reranker using Ollama LLM as a scoring function.

    Prompts the LLM to score each chunk's relevance on 0-10,
    then sorts by score. Practical for self-hosted setups without
    a dedicated cross-encoder model.
    """

    def __init__(self):
        self._model = settings.ollama_model
        self._base_url = settings.ollama_base_url

    def model_name(self) -> str:
        return f"ollama-rerank:{self._model}"

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score and rerank chunks using Ollama LLM."""
        import httpx

        with create_span(
            tracer,
            "rerank",
            "RERANKER",
            {
                "reranker.model_name": self.model_name(),
                "reranker.top_k": top_k,
                "reranker.input_documents": len(chunks),
            },
        ):
            scored_chunks = []

            async with httpx.AsyncClient(timeout=60.0) as client:
                for chunk in chunks:
                    prompt = (
                        f"Rate the relevance of this text passage to the query on a scale of 0-10.\n"
                        f"Query: {query}\n"
                        f"Passage: {chunk['content'][:500]}\n"
                        f"Respond with ONLY a number between 0 and 10."
                    )

                    try:
                        response = await client.post(
                            f"{self._base_url}/api/generate",
                            json={
                                "model": self._model,
                                "prompt": prompt,
                                "stream": False,
                                "options": {"temperature": 0.0, "num_predict": 10},
                            },
                        )
                        response.raise_for_status()
                        text = response.json().get("response", "0").strip()

                        # Extract numeric score
                        import re

                        match = re.search(r"(\d+\.?\d*)", text)
                        score = float(match.group(1)) / 10.0 if match else 0.0
                        score = min(1.0, max(0.0, score))

                    except Exception as e:
                        logger.warning("Rerank scoring failed", error=str(e))
                        score = chunk.get("rrf_score", chunk.get("dense_score", 0))

                    result = chunk.copy()
                    result["rerank_score"] = round(score, 4)
                    scored_chunks.append(result)

            # Sort by rerank score descending
            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = scored_chunks[:top_k]

            logger.info(
                "Reranking complete",
                model=self.model_name(),
                input=len(chunks),
                output=len(results),
                top_score=results[0]["rerank_score"] if results else 0,
            )

            return results


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Uses BGE-reranker-v2-m3 or similar cross-encoder models
    for high-quality relevance scoring. Self-hosted, no API calls.
    """

    def __init__(self, model_name_str: str = "BAAI/bge-reranker-v2-m3"):
        self._model_name = model_name_str
        self._model = None

    def model_name(self) -> str:
        return self._model_name

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name)
                logger.info("Cross-encoder loaded", model=self._model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, falling back to Ollama reranker"
                )
                return None
        return self._model

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank using cross-encoder model."""
        with create_span(
            tracer,
            "rerank_cross_encoder",
            "RERANKER",
            {
                "reranker.model_name": self._model_name,
                "reranker.top_k": top_k,
                "reranker.input_documents": len(chunks),
            },
        ):
            model = self._load_model()

            if model is None:
                # Fallback to Ollama reranker
                fallback = OllamaReranker()
                return await fallback.rerank(query, chunks, top_k)

            # Score all query-document pairs
            pairs = [(query, chunk["content"]) for chunk in chunks]
            scores = model.predict(pairs)

            for chunk, score in zip(chunks, scores, strict=True):
                chunk["rerank_score"] = round(float(score), 4)

            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = chunks[:top_k]

            logger.info(
                "Cross-encoder reranking complete",
                model=self._model_name,
                input=len(chunks),
                output=len(results),
            )

            return results


def get_reranker(use_cross_encoder: bool = False) -> BaseReranker:
    """Factory — returns appropriate reranker."""
    if use_cross_encoder:
        return CrossEncoderReranker()
    return OllamaReranker()
