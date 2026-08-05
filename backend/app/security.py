"""
RAGScope — Rate limiting and API-key auth

Both are FastAPI dependencies so they can be attached per-endpoint rather than
applied globally: the expensive paths (query, ingest, eval) need protecting,
the health probes must stay open for orchestrators.

Rate limiting is backed by Redis rather than process memory. With more than one
worker an in-memory counter lets a client through N times per worker, which is
not a limit. Redis also survives restarts, so a burst cannot be reset by
bouncing the app.
"""

from __future__ import annotations

import time

import redis.asyncio as redis
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from app.config import settings

logger = structlog.get_logger()

_redis: redis.Redis | None = None


async def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def client_identity(request: Request) -> str:
    """Identify the caller for rate-limiting purposes.

    X-Forwarded-For is only trusted when trust_proxy_headers is set, because
    the header is client-supplied: honouring it on a directly-exposed service
    lets anyone reset their own limit by spoofing it. Behind a load balancer
    that rewrites the header, trusting it is required — otherwise every request
    appears to come from the proxy and all clients share one bucket.
    """
    if settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Left-most entry is the original client.
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Fixed-window rate limiter.

    A fixed window can allow up to 2x the limit across a window boundary. That
    is accepted deliberately: it costs one Redis INCR, whereas a sliding window
    needs a sorted set per client. The purpose here is preventing cost blowouts
    and abuse, not precise traffic shaping.
    """

    def __init__(self, limit: int, window_seconds: int, scope: str):
        self.limit = limit
        self.window = window_seconds
        self.scope = scope

    async def __call__(self, request: Request) -> None:
        if not settings.rate_limit_enabled:
            return

        identity = client_identity(request)
        bucket = int(time.time()) // self.window
        key = f"ragscope:ratelimit:{self.scope}:{identity}:{bucket}"

        try:
            r = await _get_redis()
            count = await r.incr(key)
            if count == 1:
                # Expire slightly past the window so the key cannot outlive it.
                await r.expire(key, self.window + 1)
        except Exception as e:
            # Fail open. A Redis outage should not take down the API; the
            # alternative is refusing all traffic because the limiter is down.
            logger.warning("Rate limit check failed, allowing request", error=str(e))
            return

        if count > self.limit:
            retry_after = self.window - (int(time.time()) % self.window)
            logger.warning(
                "Rate limit exceeded",
                scope=self.scope,
                identity=identity,
                count=count,
                limit=self.limit,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded for '{self.scope}': "
                    f"{self.limit} requests per {self.window}s"
                ),
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )


# Query endpoints are the expensive ones — each costs an LLM call.
limit_query = RateLimiter(limit=settings.rate_limit_query_per_min, window_seconds=60, scope="query")
# Ingestion and evaluation are far heavier again (whole-corpus work).
limit_heavy = RateLimiter(
    limit=settings.rate_limit_heavy_per_hour, window_seconds=3600, scope="heavy"
)


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """Reject requests without a valid API key, when one is configured.

    Disabled by default (empty api_key) so local development needs no setup.
    Set RAGSCOPE_API_KEY to require it — do that before exposing the service.
    """
    if not settings.api_key:
        return

    if not api_key or not _constant_time_eq(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key",
        )


def _constant_time_eq(a: str, b: str) -> bool:
    """Compare without leaking length/prefix information through timing."""
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())
