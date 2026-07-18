# 🔬 RAGScope

**Self-hosted, production-grade RAG platform with built-in evaluation & observability layer**

> Not another PDF chatbot — RAGScope rebuilds the core mechanics of LangSmith, Arize Phoenix, and Langfuse in one self-hosted platform, with hybrid retrieval, cross-encoder reranking, LLM-as-judge evaluation, OpenTelemetry tracing, and CI regression gates.

---

## Architecture

```mermaid
graph TB
    subgraph Frontend ["Next.js Frontend"]
        PG[Playground/Chat]
        RI[Retrieval Inspector]
        ED[Eval Dashboard]
        EC[Experiment Comparison]
        TV[Trace Viewer]
        CA[Cost/Latency Analytics]
    end

    subgraph Backend ["FastAPI Backend"]
        API[API v1]
        ING[Ingestion Pipeline]
        RET[Retrieval Engine]
        GEN[Generation Service]
        EVL[Eval Harness]
        OBS[Observability Layer]
        GRD[Guardrails]
        CACHE[Semantic Cache]
    end

    subgraph Infra ["Infrastructure"]
        PG_DB[(PostgreSQL + pgvector)]
        REDIS[(Redis)]
        OLLAMA[Ollama - Self-hosted LLM]
        CELERY[Celery Workers]
    end

    Frontend --> API
    API --> ING
    API --> RET
    API --> GEN
    API --> EVL
    API --> OBS

    ING --> |Parse/Chunk/Embed| PG_DB
    ING --> |Async Jobs| CELERY
    CELERY --> REDIS

    RET --> |Dense + BM25 + RRF| PG_DB
    RET --> |Rerank| OLLAMA
    RET --> |Cache Check| CACHE
    CACHE --> REDIS

    GEN --> |LLM Inference| OLLAMA
    GEN --> |Grounded + Citations| API

    EVL --> |Metrics + LLM Judge| PG_DB
    OBS --> |OpenTelemetry + OpenInference| PG_DB

    GRD --> |PII/Injection/Hallucination| API
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Backend** | FastAPI (Python 3.11) | Async-native, auto OpenAPI docs |
| **Frontend** | Next.js 15 (App Router) | SSR, React Server Components |
| **Database** | PostgreSQL 16 + pgvector | Relational + vector in one DB |
| **LLM** | Ollama (Llama 3.1 / Qwen3) | Fully self-hosted |
| **Embeddings** | Dual: Ollama + Gemini API | Swappable registry, benchmark both |
| **Cache/Queue** | Redis 7 + Celery | Semantic cache + async ingestion |
| **Tracing** | OpenTelemetry + OpenInference | Industry-standard distributed tracing |
| **CI/CD** | GitHub Actions | Lint, test, eval regression gates |
| **Deploy** | Docker Compose (dev) → GCP Cloud Run (prod) | Local-first, cloud-ready |

## Benchmark Results

> Updated after each phase. All metrics measured on the project's golden evaluation dataset.

| Metric | Phase 1 (Baseline) | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|--------|-------------------|---------|---------|---------|---------|
| NDCG@10 | — | — | — | — | — |
| Faithfulness | — | — | — | — | — |
| Context Recall | — | — | — | — | — |
| Hallucination Rate | — | — | — | — | — |
| p95 Latency (ms) | — | — | — | — | — |
| Cost/Query (USD) | — | — | — | — | — |
| Cache Hit Rate | — | — | — | — | — |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/sahilsahu4102/ragscope.git
cd ragscope

# 2. Configure
cp .env.example .env

# 3. Start all services
make dev
# or: docker compose up --build -d

# 4. Pull LLM models (first time only)
make pull-models

# 5. Run database migrations
make migrate

# 6. Open
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
# Health: http://localhost:8000/healthz
```

## Project Structure

```
ragscope/
├── docker-compose.yml          # 6-service stack
├── Makefile                    # Dev shortcuts
├── .env.example                # Config template
├── .github/workflows/          # CI + eval regression gate
├── documents/                  # Phase-by-phase documentation
├── docs/adr/                   # Architecture Decision Records
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── config.py           # Pydantic Settings
│   │   ├── api/v1/             # Versioned endpoints
│   │   ├── ingestion/          # Parse, chunk, embed
│   │   ├── retrieval/          # Dense, BM25, RRF, rerank
│   │   ├── generation/         # LLM, prompts, citations
│   │   ├── eval/               # Metrics, LLM judge, datasets
│   │   ├── observability/      # OTel, OpenInference, cost
│   │   ├── caching/            # Semantic cache
│   │   ├── guardrails/         # PII, injection, hallucination
│   │   ├── models/             # SQLAlchemy models
│   │   ├── schemas/            # Pydantic schemas
│   │   ├── db/                 # Session, migrations
│   │   └── workers/            # Celery tasks
│   └── tests/
├── frontend/                   # Next.js dashboard
└── eval-datasets/              # Versioned golden sets (JSONL)
```

## Roadmap

- [x] **Phase 0** — Scaffold & DevOps Foundation
- [ ] **Phase 1** — MVP Vertical Slice (ingest → retrieve → generate → chat)
- [ ] **Phase 2** — Advanced Retrieval (hybrid/RRF, reranking, caching)
- [ ] **Phase 3** — Eval Harness (metrics, LLM judge, CI gates)
- [ ] **Phase 4** — Observability & Experiments (traces, A/B, analytics)
- [ ] **Phase 5** — Deploy, Guardrails, Benchmark Blog

## License

MIT
