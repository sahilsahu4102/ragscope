"""
RAGScope — Semantic Cache (Redis) — Phase 5 Optimized

Caches RAG query results by semantic similarity.
If a new query is close enough to a cached query (cosine > threshold),
return the cached answer instead of re-running the full pipeline.

Phase 5 optimizations:
  - Batched Redis MGET instead of sequential per-key GET
  - Vectorized numpy cosine similarity (all entries at once, not loop)

Phase 7: the lookup fetched and JSON-decoded *every* cached embedding on
*every* query. Measured cost by cache size:

    entries    KEYS    MGET    parse+cosine    total
         50   0.5ms   1.1ms          4.7ms    6.3ms
        250   1.0ms   4.0ms         21.4ms   26.4ms
      1,000   2.5ms  19.3ms        102.7ms  124.5ms

Parsing dominates — 1,000 x 768 floats decoded from JSON per query. KEYS,
the obvious-looking culprit, is 2.5ms of 124.5ms.

The embeddings are now held in a process-local normalised matrix and only
refetched when a Redis version counter changes, so the steady-state lookup is
one small GET plus a numpy matmul.
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

# ── Process-local mirror of the cached embeddings ──
# Keyed by the Redis version counter: if it has not moved, the matrix is
# current and no payload needs fetching.
_local_version: int | None = None
_local_keys: list[bytes] = []
_local_matrix: np.ndarray | None = None  # L2-normalised, shape (n, dim)


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
    # Bumped on write/invalidate. Every worker watches it to know whether its
    # local embedding matrix is stale.
    VERSION_KEY = "ragscope:cache_metrics:version"

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

    async def _sync_local(self, r: redis.Redis) -> None:
        """Refresh the process-local embedding matrix if Redis has moved on.

        The version counter is bumped by put() and invalidate_all(). When it
        is unchanged the local matrix is already current, so the steady-state
        cost of a lookup is one small GET rather than fetching and decoding
        every cached embedding.
        """
        global _local_version, _local_keys, _local_matrix

        raw_version = await r.get(self.VERSION_KEY)
        version = int(raw_version) if raw_version else 0

        if version == _local_version and _local_matrix is not None:
            return

        keys = await r.keys(f"{self.CACHE_PREFIX}:*")
        if not keys:
            _local_version, _local_keys, _local_matrix = version, [], None
            return

        raw_values = await r.mget(keys)
        live_keys: list[bytes] = []
        vecs: list[list[float]] = []
        for key, raw in zip(keys, raw_values, strict=True):
            if raw is None:
                continue  # expired between KEYS and MGET
            try:
                vecs.append(json.loads(raw)["query_embedding"])
                live_keys.append(key)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        if not vecs:
            _local_version, _local_keys, _local_matrix = version, [], None
            return

        matrix = np.array(vecs, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10

        _local_version, _local_keys, _local_matrix = version, live_keys, matrix
        logger.info("Semantic cache matrix synced", entries=len(live_keys), version=version)

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

                query_embeddings = await self.embedder.embed([query])
                query_vec = np.array(query_embeddings[0], dtype=np.float32)
                query_vec /= np.linalg.norm(query_vec) + 1e-10

                await self._sync_local(r)
                if _local_matrix is None or not _local_keys:
                    await r.incr(self.MISSES_KEY)
                    return None

                similarities = _local_matrix @ query_vec
                best_idx = int(np.argmax(similarities))
                best_similarity = float(similarities[best_idx])

                if best_similarity < self.threshold:
                    logger.info(
                        "Semantic cache MISS",
                        best_similarity=round(best_similarity, 4),
                        threshold=self.threshold,
                    )
                    await r.incr(self.MISSES_KEY)
                    return None

                # The matrix can outlive its entries: TTL expiry does not bump
                # the version. Fetch only the winning key to confirm.
                raw = await r.get(_local_keys[best_idx])
                if raw is None:
                    # Expired since the last sync — force a rebuild next call
                    # rather than serving a stale answer.
                    self._invalidate_local()
                    await r.incr(self.MISSES_KEY)
                    return None

                best_match = json.loads(raw)
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

            except Exception as e:
                logger.warning("Semantic cache lookup failed", error=str(e))
                return None

    @staticmethod
    def _invalidate_local() -> None:
        global _local_version, _local_keys, _local_matrix
        _local_version, _local_keys, _local_matrix = None, [], None

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
                # Signal every worker that its local matrix is stale.
                await r.incr(self.VERSION_KEY)

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
                await r.incr(self.VERSION_KEY)
                self._invalidate_local()
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
