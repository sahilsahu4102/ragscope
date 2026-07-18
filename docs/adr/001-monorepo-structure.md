# ADR-001: Monorepo Structure & Technology Choices

**Status:** Accepted  
**Date:** 2026-07-18  
**Author:** Sahil Sahu

## Context

RAGScope is a self-hosted, production-grade RAG platform with an integrated evaluation and observability layer. We need to choose an architecture that supports:
- Rapid iteration across backend (Python) and frontend (TypeScript)
- Clear service boundaries for ingestion, retrieval, generation, eval, and observability
- Docker-based development with zero external dependencies
- Phase-wise git commits that tell a coherent story

## Decision

### Monorepo over Polyrepo
- Single repo with `backend/` and `frontend/` top-level directories
- Shared `docker-compose.yml` at root for the full stack
- Simpler CI/CD, atomic commits across backend + frontend changes

### Technology Stack
| Layer | Choice | Rationale |
|-------|--------|-----------|
| Backend | FastAPI (Python 3.11) | Async-native, OpenAPI auto-gen, strong typing |
| Frontend | Next.js 15 (App Router) | SSR, React Server Components, portfolio signal |
| Database | PostgreSQL 16 + pgvector | Single DB for relational + vector; ACID; SQL filtering |
| Cache/Queue | Redis 7 | Semantic cache + Celery broker in one service |
| Task Queue | Celery | Mature, Flower monitoring, async ingestion |
| LLM | Ollama (self-hosted) | Full self-hosted narrative; Llama 3.1 / Qwen3 |
| Embeddings | Dual: Ollama + Gemini API | Swappable registry; benchmark both |
| Tracing | OpenTelemetry + OpenInference | Industry standard; wired from Phase 0 |
| CI/CD | GitHub Actions | Lint, test, eval regression gates |

### Backend Module Structure
Service-modular: `ingestion/`, `retrieval/`, `generation/`, `eval/`, `observability/`, `caching/`, `guardrails/`. Each module has clear interfaces so they can be built and tested independently.

### API Versioning
All endpoints under `/api/v1/` from day one — production API design pattern.

## Consequences

- **Positive:** Single `docker compose up` spins up entire stack. Clear module boundaries. Phase-wise commits are coherent.
- **Negative:** Monorepo CI is slightly more complex (path-filtered workflows). pgvector has a ~10M vector practical limit (acceptable for this project).

## Alternatives Considered

- **Qdrant** for vector store: Better filtered search at scale, but adds another service. pgvector is sufficient for <10M vectors and reduces operational complexity.
- **Polyrepo**: Rejected — atomic cross-stack commits are more valuable for this project's phase-wise development model.
- **LangServe**: Rejected — too opinionated, limits custom observability layer.
