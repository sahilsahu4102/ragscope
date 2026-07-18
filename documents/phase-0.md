# Phase 0 — Scaffold & DevOps Foundation

**Duration:** 2-3 days  
**Git Tag:** `v0.1.0-scaffold`  
**Status:** ✅ Complete

## What Was Built

### Infrastructure
- **Docker Compose** — 6-service stack: PostgreSQL 16 + pgvector, Redis 7, Ollama (self-hosted LLM), FastAPI backend, Celery worker, Next.js frontend
- **Makefile** — Common operations: `make dev`, `make test`, `make lint`, `make migrate`, `make pull-models`
- **GitHub Actions CI** — Lint (ruff) + test pipeline with Postgres/Redis services
- **Eval regression gate** — Placeholder workflow (implemented in Phase 3)

### Backend Scaffold
- **FastAPI** app factory with lifespan events, CORS, versioned API (`/api/v1/`)
- **Health probes** — `/healthz` (liveness) and `/readyz` (readiness)
- **Pydantic Settings** — Centralized config from `.env` with validation
- **SQLAlchemy async** — Engine, session factory, pgvector extension registration
- **Data models** — `Document`, `Chunk` (with pgvector embedding + parent_id for hierarchical), `Query`, `Feedback`
- **Pydantic schemas** — Request/response models for all planned endpoints
- **Celery** — App configured with Redis broker for async tasks
- **Structured logging** — structlog → JSON (prod) / pretty-print (dev)
- **OpenTelemetry** — Tracer provider with OpenInference span helpers (wired from day one)
- **Module stubs** — `ingestion/`, `retrieval/`, `generation/`, `eval/`, `caching/`, `guardrails/`

### Documentation
- **README.md** — Architecture diagram, tech stack, quickstart, benchmark table
- **ADR-001** — Monorepo structure & technology choices
- **`.env.example`** — All config vars with comments

### Tests
- Smoke tests — Health probes, API root, OpenAPI docs accessibility

## Key Design Decisions
1. **API versioning from day one** (`/api/v1/`) — production pattern
2. **OpenInference tracing wired in Phase 0** — not bolted on later
3. **Dual embedding registry** — Ollama (nomic-embed-text) default + Gemini API option
4. **Fully self-hosted** — Ollama for LLM inference, no external API dependencies required

## Files Created

```
ragscope/
├── docker-compose.yml
├── Makefile
├── .env.example
├── .gitignore
├── README.md
├── .github/workflows/
│   ├── ci.yml
│   └── evals.yml
├── docs/
│   ├── adr/001-monorepo-structure.md
│   └── phases/phase-0.md
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── __init__.py
│   │   ├── api/v1/__init__.py
│   │   ├── models/models.py
│   │   ├── schemas/schemas.py
│   │   ├── db/session.py
│   │   ├── observability/{tracer,logging}.py
│   │   ├── workers/celery_app.py
│   │   └── {ingestion,retrieval,generation,eval,caching,guardrails}/__init__.py
│   └── tests/test_smoke.py
├── frontend/  (Next.js — initialized next)
└── eval-datasets/
```

## Next Phase
**Phase 1 — MVP Vertical Slice**: Ingest PDF → chunk → embed → dense retrieve → generate with citations → chat UI
