"""
RAGScope — Semantic Cache Tests

The cache mirrors Redis into a process-local matrix to keep lookup flat in
cache size. That mirror introduces two ways to be wrong, and neither surfaces
as an error in production:

  1. A second worker writes an entry and this process never sees it.
  2. An entry expires by TTL and this process keeps serving its answer.

Both are covered here with a fake Redis, so no service is required — matching
the rest of the suite.
"""

import json
from typing import Any, cast

import numpy as np
import pytest

import app.caching.semantic_cache as sc
from app.caching.semantic_cache import SemanticCache


class FakeRedis:
    """Minimal async Redis stand-in supporting only what the cache uses."""

    def __init__(self):
        self.store: dict[bytes, bytes] = {}

    @staticmethod
    def _b(k) -> bytes:
        return k if isinstance(k, bytes) else str(k).encode()

    async def get(self, key):
        return self.store.get(self._b(key))

    async def setex(self, key, _ttl, value):
        self.store[self._b(key)] = value.encode() if isinstance(value, str) else value

    async def incr(self, key):
        cur = int(self.store.get(self._b(key), b"0"))
        self.store[self._b(key)] = str(cur + 1).encode()
        return cur + 1

    async def keys(self, pattern):
        prefix = pattern.rstrip("*").encode()
        return [k for k in self.store if k.startswith(prefix)]

    async def mget(self, keys):
        return [self.store.get(self._b(k)) for k in keys]

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if self.store.pop(self._b(k), None) is not None:
                n += 1
        return n


class StubEmbedder:
    """Returns a fixed vector per text so similarity is deterministic."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    async def embed(self, texts):
        return [self.mapping[t] for t in texts]

    def model_name(self):
        return "stub"


def _cache(mapping) -> tuple[SemanticCache, FakeRedis]:
    """Cache wired to in-memory fakes. cast() because the attributes are typed
    for the real Redis/embedder; substituting fakes is the point here."""
    sc._local_version, sc._local_keys, sc._local_matrix = None, [], None
    c = SemanticCache(similarity_threshold=0.9)
    fake = FakeRedis()
    c._redis = cast(Any, fake)
    c.embedder = cast(Any, StubEmbedder(mapping))
    return c, fake


VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_hit_on_identical_query():
    c, _ = _cache({"q": VEC_A})
    await c.put(query="q", answer="ANSWER", citations=[], latency_ms=1.0)
    hit = await c.get("q")
    assert hit is not None
    assert hit["answer"] == "ANSWER"
    assert hit["cached"] is True


@pytest.mark.asyncio
async def test_miss_on_orthogonal_query():
    c, _ = _cache({"q": VEC_A, "other": VEC_B})
    await c.put(query="q", answer="ANSWER", citations=[], latency_ms=1.0)
    assert await c.get("other") is None


@pytest.mark.asyncio
async def test_sees_entry_written_by_another_worker():
    """A write from elsewhere bumps the version, so the local mirror resyncs.

    Without the version check this process would keep using a stale matrix and
    never return the other worker's entry.
    """
    c, r = _cache({"q": VEC_A})
    await c.put(query="q", answer="MINE", citations=[], latency_ms=1.0)
    first = await c.get("q")
    assert first is not None and first["answer"] == "MINE"

    # Simulate another process writing directly to Redis.
    entry = {
        "query": "q",
        "answer": "THEIRS",
        "citations": [],
        "query_embedding": VEC_A,
        "latency_ms": 1.0,
        "cached_at": 0,
    }
    await r.setex(f"{c.CACHE_PREFIX}:other", 3600, json.dumps(entry))
    await r.incr(c.VERSION_KEY)

    hit = await c.get("q")
    assert hit is not None
    assert hit["answer"] in {"MINE", "THEIRS"}
    assert len(sc._local_keys) == 2, "local mirror should have resynced to 2 entries"


@pytest.mark.asyncio
async def test_expired_entry_is_not_served():
    """TTL expiry does not bump the version, so the mirror still lists the key.

    The winning key is confirmed with a GET before its answer is returned; if
    it has expired the lookup must miss rather than serve a stale answer.

    Asserting only `is None` is not enough: removing the guard makes
    json.loads(None) raise, which the broad except-clause turns into None too,
    so the test would pass against broken code. The guard's distinguishing
    side effects are that it records a miss and drops the local mirror, so
    those are what get asserted.
    """
    c, r = _cache({"q": VEC_A})
    await c.put(query="q", answer="STALE", citations=[], latency_ms=1.0)
    first = await c.get("q")
    assert first is not None and first["answer"] == "STALE"

    # Expire the entry without touching the version counter.
    for key in list(r.store):
        if key.startswith(c.CACHE_PREFIX.encode()):
            del r.store[key]

    misses_before = int(r.store.get(c.MISSES_KEY.encode(), b"0"))

    assert await c.get("q") is None, "expired entry must not be served"

    misses_after = int(r.store.get(c.MISSES_KEY.encode(), b"0"))
    assert misses_after == misses_before + 1, "expiry must be recorded as a miss, not an error"
    assert sc._local_matrix is None, "expiry must drop the stale local mirror"


@pytest.mark.asyncio
async def test_lookup_does_not_refetch_when_version_unchanged():
    """Steady state must not re-read every payload — that was the 124ms cost."""
    c, r = _cache({"q": VEC_A})
    await c.put(query="q", answer="ANSWER", citations=[], latency_ms=1.0)
    await c.get("q")  # first call syncs

    calls = {"n": 0}
    original = r.mget

    async def counting_mget(keys):
        calls["n"] += 1
        return await original(keys)

    r.mget = cast(Any, counting_mget)
    await c.get("q")
    assert calls["n"] == 0, "unchanged version should not trigger an MGET"


@pytest.mark.asyncio
async def test_invalidate_all_clears_local_mirror():
    c, _ = _cache({"q": VEC_A})
    await c.put(query="q", answer="ANSWER", citations=[], latency_ms=1.0)
    await c.get("q")
    assert sc._local_matrix is not None

    await c.invalidate_all()
    assert sc._local_matrix is None
    assert await c.get("q") is None


@pytest.mark.asyncio
async def test_similarity_is_cosine_on_normalised_vectors():
    """Entries are stored unnormalised; the mirror normalises them, so a
    scaled duplicate must still match at similarity 1."""
    c, _ = _cache({"q": VEC_A, "scaled": [5.0, 0.0, 0.0]})
    await c.put(query="q", answer="ANSWER", citations=[], latency_ms=1.0)
    hit = await c.get("scaled")
    assert hit is not None
    assert hit["cache_similarity"] == pytest.approx(1.0, abs=1e-4)


def test_local_mirror_starts_empty():
    sc._local_version, sc._local_keys, sc._local_matrix = None, [], None
    assert sc._local_matrix is None
    assert np.asarray(sc._local_keys).size == 0
