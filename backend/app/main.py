"""
RAGScope — FastAPI Application Entrypoint

Factory pattern with lifespan events, CORS, versioned routers,
structured logging, health probes, and OpenTelemetry bootstrap.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router as v1_router
from app.config import settings
from app.db.session import engine, init_db
from app.observability.logging import setup_logging
from app.observability.tracer import setup_tracing


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
    logger.info("RAGScope ready", api_version="v1")

    yield

    # ── Shutdown ──────────────────────────────
    logger.info("Shutting down RAGScope")
    await engine.dispose()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Self-hosted, production-grade RAG platform with "
            "built-in evaluation & observability layer"
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────
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
        """Readiness probe — can we serve traffic?"""
        # TODO: check DB and Redis connectivity
        return {"status": "ready"}

    # ── Versioned API Router ──────────────────
    app.include_router(v1_router, prefix="/api/v1")

    return app


app = create_app()
