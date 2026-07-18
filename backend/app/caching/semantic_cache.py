"""
RAGScope — Semantic Cache (Redis)

Caches RAG query results by semantic similarity.
If a new query is close enough to a cached query (cosine > threshold),
return the cached answer instead of re-running the full pipeline.

Reduces LLM costs by 40-60% on production traffic with repeated knowledge queries.
"""

import hashlib
import json
import time

import numpy as np
import redis.asyncio as redis
import structlog

from app.config import settings
from app.ingestion.embedders.embedder import get_embedder
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("caching")


class SemanticCache:
    """
    Semantic query cache backed by Redis.

    On each query:
      1. Embed the query
      2. Compare to cached query embeddings (cosine similarity)
      3. If similarity > threshold → cache HIT, return cached answer
      4. If miss → run pipeline, store result in cache

    TTL-based expiration ensures stale answers are purged.
    """

    CACHE_PREFIX = "ragscope:semantic_cache"
    # Metrics live under a separate prefix so entry globs + invalidation
    # never touch the hit/miss counters.
    HITS_KEY = "ragscope:cache_metrics:hits"
    MISSES_KEY = "ragscope:cache_metrics:misses"

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 3600,
    ):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.embedder = get_embedder()
        self._redis: redis.Redis | None = None

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                settings.redis_url,
                decode_responses=False,
            )
        return self._redis

    async def get(self, query: str) -> dict | None:
        """
        Check if a semantically similar query is cached.

        Returns cached response dict or None.
        """
        with create_span(
            tracer,
            "cache_lookup",
            "CHAIN",
            {
                "cache.type": "semantic",
                "cache.threshold": self.threshold,
            },
        ):
            try:
                r = await self._get_redis()

                # Embed the query
                query_embeddings = await self.embedder.embed([query])
                query_vec = np.array(query_embeddings[0])

                # Get all cached entries
                keys = await r.keys(f"{self.CACHE_PREFIX}:*")
                if not keys:
                    await r.incr(self.MISSES_KEY)
                    return None

                best_match: dict | None = None
                best_similarity = 0.0

                for key in keys:
                    data = await r.get(key)
                    if not data:
                        continue

                    entry = json.loads(data)
                    cached_vec = np.array(entry["query_embedding"])

                    # Cosine similarity
                    similarity = float(
                        np.dot(query_vec, cached_vec)
                        / (np.linalg.norm(query_vec) * np.linalg.norm(cached_vec) + 1e-10)
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry

                if best_match and best_similarity >= self.threshold:
                    logger.info(
                        "Semantic cache HIT",
                        similarity=round(best_similarity, 4),
                        cached_query=best_match.get("query", "")[:80],
                    )
                    await r.incr(self.HITS_KEY)
                    return {
                        "answer": best_match["answer"],
                        "citations": best_match.get("citations", []),
                        "cached": True,
                        "cache_similarity": round(best_similarity, 4),
                        "original_latency_ms": best_match.get("latency_ms", 0),
                    }

                logger.info(
                    "Semantic cache MISS",
                    best_similarity=round(best_similarity, 4),
                    threshold=self.threshold,
                )
                await r.incr(self.MISSES_KEY)
                return None

            except Exception as e:
                logger.warning("Semantic cache lookup failed", error=str(e))
                return None

    async def put(
        self,
        query: str,
        answer: str,
        citations: list[dict],
        latency_ms: float,
    ) -> None:
        """Store a query-answer pair in the semantic cache."""
        with create_span(tracer, "cache_store", "CHAIN", {}):
            try:
                r = await self._get_redis()

                # Embed and store
                query_embeddings = await self.embedder.embed([query])

                entry = {
                    "query": query,
                    "answer": answer,
                    "citations": citations,
                    "query_embedding": query_embeddings[0],
                    "latency_ms": latency_ms,
                    "cached_at": time.time(),
                }

                # Use query hash as key
                key_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
                cache_key = f"{self.CACHE_PREFIX}:{key_hash}"

                await r.setex(
                    cache_key,
                    self.ttl,
                    json.dumps(entry),
                )

                logger.info("Query cached", key=cache_key, ttl=self.ttl)

            except Exception as e:
                logger.warning("Semantic cache store failed", error=str(e))

    async def invalidate_all(self) -> int:
        """Clear all cached entries. Returns count of deleted keys."""
        try:
            r = await self._get_redis()
            keys = await r.keys(f"{self.CACHE_PREFIX}:*")
            if keys:
                deleted = await r.delete(*keys)
                logger.info("Semantic cache cleared", deleted=deleted)
                return deleted
            return 0
        except Exception as e:
            logger.warning("Cache invalidation failed", error=str(e))
            return 0

    async def stats(self) -> dict:
        """Return cache statistics including hit/miss counters and hit rate."""
        try:
            r = await self._get_redis()
            keys = await r.keys(f"{self.CACHE_PREFIX}:*")
            hits = int(await r.get(self.HITS_KEY) or 0)
            misses = int(await r.get(self.MISSES_KEY) or 0)
            total = hits + misses
            hit_rate = round(hits / total, 4) if total else 0.0
            return {
                "entries": len(keys),
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl,
            }
        except Exception:
            return {
                "entries": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl,
            }
