"""
RAGScope — FastAPI Application Entrypoint (Phase 5)

Factory pattern with lifespan events, CORS, versioned routers,
structured logging, health probes, and OpenTelemetry bootstrap.

Phase 5: Added GZip compression, request timing middleware,
proper readiness probe (DB + Redis), and HTTP client lifecycle.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.api.v1 import router as v1_router
from app.config import settings
from app.db.session import engine, init_db
from app.observability.logging import setup_logging
from app.observability.tracer import setup_tracing


class TimingMiddleware(BaseHTTPMiddleware):
    """Add X-Response-Time header to every response for latency debugging."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        return response


async def _warm_reranker(logger) -> None:
    """Load the cross-encoder checkpoint ahead of the first request."""
    try:
        from app.retrieval.rerankers import get_reranker

        reranker = get_reranker()
        get_model = getattr(reranker, "_get_model", None)
        if get_model is None:
            return  # Ollama backend — nothing to preload.
        start = time.perf_counter()
        if await get_model() is not None:
            logger.info(
                "Reranker warmed",
                model=reranker.model_name(),
                load_ms=round((time.perf_counter() - start) * 1000, 1),
            )
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("Reranker warmup failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown hooks."""
    logger = structlog.get_logger()

    # ── Startup ───────────────────────────────
    logger.info("Starting RAGScope", env=settings.app_env)
    setup_logging(settings.log_level)
    setup_tracing(
        settings.app_name,
        sampling_rate=settings.trace_sampling_rate,
        console_export=settings.trace_console_export,
        otlp_endpoint=settings.otlp_endpoint,
    )
    await init_db()

    # Warm the reranker in the background so the first query doesn't pay the
    # checkpoint load. Fire-and-forget: a failure here degrades to the Ollama
    # fallback rather than blocking startup.
    warmup_task = asyncio.create_task(_warm_reranker(logger))

    logger.info("RAGScope ready", api_version="v1", version="1.0.0")

    yield

    warmup_task.cancel()

    # ── Shutdown ──────────────────────────────
    logger.info("Shutting down RAGScope")
    from app.http_client import close_http_client

    await close_http_client()
    await engine.dispose()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Self-hosted, production-grade RAG platform with "
            "built-in evaluation & observability layer"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (order matters — outermost first) ──
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health Probes ─────────────────────────
    @app.get("/healthz", tags=["infra"])
    async def healthz():
        """Liveness probe — is the process alive?"""
        return {"status": "alive"}

    @app.get("/readyz", tags=["infra"])
    async def readyz():
        """Readiness probe — can we serve traffic? Checks DB + Redis."""
        checks: dict = {}
        ok = True

        # Check PostgreSQL
        try:
            from sqlalchemy import text

            from app.db.session import async_session

            async with async_session() as session:
                await session.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
        except Exception as e:
            checks["postgres"] = f"error: {e}"
            ok = False

        # Check Redis
        try:
            import redis.asyncio as redis_mod

            r = redis_mod.from_url(settings.redis_url, decode_responses=True)
            await r.ping()
            await r.aclose()
            checks["redis"] = "ok"
        except Exception as e:
            checks["redis"] = f"error: {e}"
            ok = False

        return {
            "status": "ready" if ok else "degraded",
            "checks": checks,
        }

    # ── Versioned API Router ──────────────────
    app.include_router(v1_router, prefix="/api/v1")

    return app


app = create_app()
