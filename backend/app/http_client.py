"""
RAGScope — Shared HTTP Client (Phase 5)

Long-lived httpx.AsyncClient with connection pooling and keep-alive.
Eliminates the overhead of creating a new TCP connection + TLS handshake
per Ollama call (~50ms saved per request on average).

Usage:
    from app.http_client import get_http_client
    client = get_http_client()
    response = await client.post(...)
"""

from __future__ import annotations

import httpx
import structlog

logger = structlog.get_logger()

# Module-level singleton — created on first access, reused across requests.
_client: httpx.AsyncClient | None = None


def get_http_client(timeout: float = 120.0) -> httpx.AsyncClient:
    """Return the shared async HTTP client.

    Uses connection pooling (max 20 connections, max 10 keepalive)
    to reuse TCP connections across Ollama/API calls.
    """
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30.0,
            ),
            http2=False,  # Ollama doesn't support HTTP/2
        )
        logger.info("HTTP client pool created", max_connections=20, keepalive=10)
    return _client


async def close_http_client() -> None:
    """Gracefully close the shared client (call on app shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
        logger.info("HTTP client pool closed")
