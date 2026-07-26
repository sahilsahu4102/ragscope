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

Phase 6 (latency):
  Measured traces showed OllamaReranker at 62-67 SECONDS per query — it runs a
  full 8B generation just to produce relevance scores, and was the single largest
  component of end-to-end latency. The default is now a small cross-encoder
  (ms-marco-MiniLM-L-6-v2, 22M params) which scores 20 pairs in ~30-60ms.

  Two correctness fixes came with it:
    - The model is cached at module level. get_reranker() is called per-request,
      so a per-instance cache meant reloading the checkpoint from disk every query.
    - predict() is CPU-bound and synchronous; it now runs in a thread so it does
      not block the event loop for every other in-flight request.
"""

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any

import structlog

from app.config import settings
from app.http_client import get_http_client
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")

# ── Module-level model cache ───────────────────
# get_reranker() is called per-request; without this the checkpoint would be
# re-read from disk on every query.
_model_cache: dict[str, Any] = {}
_model_lock = asyncio.Lock()


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
    (e.g., 20 x 500ms = 10s → 1 x 800ms = 0.8s).
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

    Default checkpoint is ms-marco-MiniLM-L-6-v2 (22M params), which scores
    ~20 query-document pairs in 30-60ms on CPU. The model is held in a
    module-level cache and scored on a worker thread.
    """

    # Truncate passages before scoring. Cross-encoder cost is roughly linear in
    # sequence length, and relevance is almost always decidable from the first
    # couple hundred tokens — so this is the main latency dial on this stage.
    MAX_PASSAGE_CHARS = 800

    def __init__(self, model_name_str: str | None = None):
        self._model_name = model_name_str or settings.reranker_model

    def model_name(self) -> str:
        return self._model_name

    async def _get_model(self):
        """Load the cross-encoder once, then serve from the module cache.

        Loading happens in a thread (it does disk I/O and torch init) and is
        guarded by a lock so concurrent first-requests don't each load a copy.
        """
        cached = _model_cache.get(self._model_name)
        if cached is not None:
            return cached

        async with _model_lock:
            # Re-check: another coroutine may have loaded it while we waited.
            cached = _model_cache.get(self._model_name)
            if cached is not None:
                return cached

            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, falling back to Ollama reranker"
                )
                return None

            try:
                # ty models to_thread as taking only the callable; it is variadic.
                model = await asyncio.to_thread(CrossEncoder, self._model_name)  # ty: ignore[too-many-positional-arguments]
            except Exception as e:
                logger.warning(
                    "Cross-encoder load failed, falling back to Ollama reranker",
                    model=self._model_name,
                    error=str(e),
                )
                return None

            _model_cache[self._model_name] = model
            logger.info("Cross-encoder loaded and cached", model=self._model_name)
            return model

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank using cross-encoder model."""
        if not chunks:
            return []

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
            model = await self._get_model()

            if model is None:
                # Fallback to Ollama reranker
                fallback = OllamaReranker()
                return await fallback.rerank(query, chunks, top_k)

            pairs = [
                (query, chunk["content"][: self.MAX_PASSAGE_CHARS]) for chunk in chunks
            ]

            # predict() is synchronous CPU work — run it off the event loop so it
            # doesn't stall every other in-flight request.
            scores = await asyncio.to_thread(model.predict, pairs)

            # Copy rather than mutating the caller's dicts in place.
            scored_chunks = []
            for chunk, score in zip(chunks, scores, strict=True):
                result = chunk.copy()
                result["rerank_score"] = round(float(score), 4)
                scored_chunks.append(result)

            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            results = scored_chunks[:top_k]

            logger.info(
                "Cross-encoder reranking complete",
                model=self._model_name,
                input=len(chunks),
                output=len(results),
                top_score=results[0]["rerank_score"] if results else 0,
            )

            return results


def get_reranker(use_cross_encoder: bool | None = None) -> BaseReranker:
    """Factory — returns the configured reranker.

    Defaults to settings.reranker_backend ('cross_encoder'). Pass an explicit
    bool to override per-request, e.g. for A/B experiments.
    """
    if use_cross_encoder is None:
        use_cross_encoder = settings.reranker_backend.lower() == "cross_encoder"
    if use_cross_encoder:
        return CrossEncoderReranker()
    return OllamaReranker()
