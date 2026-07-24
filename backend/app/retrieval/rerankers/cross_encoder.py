"""
RAGScope — Cross-Encoder Reranker (Phase 5 — Optimized)

Reranks top-N candidates down to top-K using a cross-encoder model.
The single highest-ROI retrieval improvement: +5 to +15 NDCG@10.

Supports:
  - Self-hosted BGE-reranker-v2-m3 (via sentence-transformers)
  - Ollama-based reranking (prompt-based)

Phase 5 optimization:
  - OllamaReranker now scores ALL chunks in a single LLM call
    instead of one call per chunk (~90% latency reduction: 20 calls → 1)
  - Uses shared httpx connection pool
"""

import re
from abc import ABC, abstractmethod

import structlog

from app.config import settings
from app.http_client import get_http_client
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

    Phase 5: Scores ALL chunks in a single LLM call with structured output,
    instead of N separate calls. This reduces reranking latency by ~90%
    (e.g., 20 × 500ms = 10s → 1 × 800ms = 0.8s).
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
        """Score and rerank all chunks in a single batched LLM call."""
        if not chunks:
            return []

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
            # Build a single prompt that scores all passages at once
            passages_text = ""
            for i, chunk in enumerate(chunks):
                content = chunk["content"][:400]  # Truncate for context window
                passages_text += f"[{i}] {content}\n\n"

            prompt = (
                "You are a relevance scoring system. Rate each passage's relevance "
                "to the query on a scale of 0-10.\n\n"
                f"Query: {query}\n\n"
                f"Passages:\n{passages_text}\n"
                "Output ONLY one line per passage in the format: "
                "[index] score\n"
                "Example:\n[0] 8\n[1] 3\n[2] 9\n\n"
                "Scores:"
            )

            try:
                client = get_http_client()
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": len(chunks) * 8,  # ~8 chars per line
                        },
                    },
                )
                response.raise_for_status()
                result_text = response.json().get("response", "")

                # Parse scores from structured output
                scores = self._parse_batch_scores(result_text, len(chunks))

            except Exception as e:
                logger.warning("Batch reranking failed, using fallback scores", error=str(e))
                scores = [
                    chunk.get("rrf_score", chunk.get("dense_score", 0.5))
                    for chunk in chunks
                ]

            # Apply scores to chunks
            scored_chunks = []
            for chunk, score in zip(chunks, scores, strict=False):
                result = chunk.copy()
                result["rerank_score"] = round(float(score), 4)
                scored_chunks.append(result)

            # Sort by rerank score descending
            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = scored_chunks[:top_k]

            logger.info(
                "Reranking complete (batched)",
                model=self.model_name(),
                input=len(chunks),
                output=len(results),
                top_score=results[0]["rerank_score"] if results else 0,
            )

            return results

    @staticmethod
    def _parse_batch_scores(text: str, num_chunks: int) -> list[float]:
        """Parse structured [index] score output from the LLM."""
        scores = [0.5] * num_chunks  # default fallback

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match patterns like [0] 8, [1] 7.5, 0: 8, etc.
            match = re.match(r"\[?(\d+)\]?\s*[:=]?\s*(\d+\.?\d*)", line)
            if match:
                idx = int(match.group(1))
                score = float(match.group(2)) / 10.0  # Normalize to 0-1
                score = min(1.0, max(0.0, score))
                if 0 <= idx < num_chunks:
                    scores[idx] = score

        return scores


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
