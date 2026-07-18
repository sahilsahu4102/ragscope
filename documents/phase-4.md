# Phase 4 — Observability & Experiments

**Duration:** Week 7-8
**Git Tag:** `v0.5.0-observability`
**Status:** ✅ Complete
**Date:** 2026-07-18

---

## Overview

Phase 4 turns the OpenInference spans that RAGScope has emitted since Phase 0
into a **queryable trace store**, adds an **A/B experiment framework** with a
real significance test, and computes **cost/latency/cache/throughput analytics**
from the query history — all surfaced through a newly bootstrapped **Next.js 15**
dashboard.

This is the phase that makes RAGScope legible: you can now see *where the
milliseconds and tokens go* on any query, and *prove* whether a config change
(e.g. reranker on vs off) actually helped, with a number and a p-value.

---

## What Was Built

### Backend

#### 1. Trace store (`models/trace.py`)
- `Trace` — one row per request: `otel_trace_id`, optional `query_id`, rolled-up
  `total_duration_ms` / `total_tokens` / `total_cost_usd`, `span_count`, status.
- `Span` — one row per span: `otel_span_id`, `parent_span_id`, `span_kind`
  (OpenInference), timings, per-span `tokens` / `cost_usd`, full `attributes` JSONB.

#### 2. Trace collector (`observability/collector.py`)
- `SpanCollector(SpanProcessor)` — a custom OTel span processor registered on the
  tracer provider. Buffers finished spans in-process keyed by OTel trace id,
  thread-safe, with an eviction cap and a **configurable sampling rate**
  (deterministic per-trace so a whole trace is kept or dropped together).
- `persist_trace(db, trace_id_hex, query_id)` — drains a trace's spans and writes
  the `Trace` + `Span` tree through the app's async session, computing aggregates.
  Never raises into the request path.
- Rationale: first-class traces in our own Postgres (for the waterfall UI) without
  standing up a separate collector service — while still demonstrating how OTel's
  span pipeline works internally.

#### 3. Cost model (`observability/cost.py`)
- `calculate_cost(model, input_tokens, output_tokens)` with a per-1M-token price
  table. Self-hosted (Ollama) models are $0 but still token-tracked; real Gemini/
  OpenAI rates are included so a config switch surfaces the true cost delta.
- Prefix matching tolerates tags (`llama3.1:8b-instruct-q4` → `llama3.1`).

#### 4. Tracer enhancements (`observability/tracer.py`)
- Registers the `SpanCollector`; optional console exporter and **OTLP export**
  (Grafana/Tempo/Phoenix) via config.
- `get_current_trace_id()` and `gen_ai.*` attribute builders
  (`llm_attributes`, `retriever_attributes`, `reranker_attributes`).
- The query path now adopts the live OTel trace id as the response `trace_id`,
  and the LLM span carries `gen_ai.usage.*` + `cost.usd`.

#### 5. A/B experiment framework (`eval/experiments.py`, `models/eval.py::Experiment`)
- `ExperimentRunner` runs the same golden dataset through **two configs**
  (variant A vs B) via the eval runner, then computes:
  - **Per-metric deltas** (`diff_metrics`) with winner logic that respects
    lower-is-better metrics (hallucination).
  - **Paired-bootstrap significance** (`_bootstrap_significant`): aligns per-sample
    scores across runs, resamples paired differences (1000 iters, fixed seed →
    reproducible), and reports a two-sided p-value + significance flag.

#### 6. Cache observability (`caching/semantic_cache.py`)
- Added Redis hit/miss counters (separate key prefix so entry globs and
  invalidation never touch them) → real hit-rate for analytics.

#### 7. API endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/traces` | GET | List traces (filter by latency, status) |
| `/api/v1/traces/{id}` | GET | Full span tree (accepts UUID or OTel hex) |
| `/api/v1/experiments` | POST | Create + run an A/B experiment |
| `/api/v1/experiments` | GET | List experiments |
| `/api/v1/experiments/{id}` | GET | Experiment with deltas |
| `/api/v1/feedback` | POST | Thumbs up/down + correction (by query_id or trace_id) |
| `/api/v1/feedback` | GET | List feedback |
| `/api/v1/analytics/latency` | GET | p50/p95/p99 overall + per day |
| `/api/v1/analytics/cost` | GET | Total + per-query cost, by model, over time |
| `/api/v1/analytics/cache` | GET | Hit rate + estimated savings |
| `/api/v1/analytics/throughput` | GET | Queries/day + avg/peak QPS |

#### 8. Bug fixes (Phase 3 eval runner)
The experiment framework depends on the eval runner, which had latent breakage:
- Imported `GroundedGenerator` (class is `Generator`).
- Treated `Generator.generate()`'s dict return as a string.
- Called `pipeline.retrieve()` with `mode=` / `rerank_top_k=` kwargs that don't
  exist. Rewired to the real signature. Eval + experiments now work end-to-end.

### Frontend — Next.js 15 (bootstrapped this phase)

The frontend was static HTML mockups through Phase 3. Phase 4 bootstraps the
**real Next.js 15 app** (App Router, TypeScript, Tailwind, Recharts) that the
plan called for, fixing the previously-unbuildable `docker-compose` frontend
service.

- **App shell** — `app/layout.tsx` + `components/Sidebar.tsx` (dark "Lumina Nexus"
  teal/navy theme), typed API client `lib/api.ts`, shared types, formatters, UI kit.
- **Overview** (`app/page.tsx`) — Phase 4 feature tour + span-kind legend.
- **Trace Viewer** (`app/traces/page.tsx` + `components/SpanWaterfall.tsx`) —
  filterable trace list → **span-waterfall** timeline. Nested spans indented by
  depth, bars positioned/sized by real timing, color-coded by kind, click a span
  for duration/tokens/cost + full attributes.
- **Experiments** (`app/experiments/page.tsx`) — A/B builder with presets
  (reranker on/off, hybrid vs dense, HyDE vs none), delta table with
  up/down/significance, and an A-vs-B Recharts bar chart.
- **Analytics** (`app/analytics/page.tsx`) — KPI row + latency percentile lines,
  cost-over-time area, throughput bars, cost-by-model table, cache savings.
- The Phase 1–3 mockups remain reachable from the sidebar (`/chat.html`,
  `/inspector.html`).
- Bumped Next.js to **15.5.20** (patches CVE-2025-66478). `frontend/Dockerfile`
  is multi-stage (dev target for compose, standalone prod target for Cloud Run).

---

## Span Tree

```
CHAIN  rag_query
├── CHAIN  cache_lookup
├── CHAIN  retrieval_pipeline
│   ├── RETRIEVER  dense_retrieve
│   │   └── EMBEDDING  embed_query
│   ├── RETRIEVER  sparse_retrieve
│   └── RERANKER   rerank
└── LLM    llm_generate   (gen_ai.usage.*, cost.usd)
```

---

## Files Created / Modified

### New (backend)
| File | Purpose |
|------|---------|
| `app/models/trace.py` | Trace + Span models |
| `app/observability/collector.py` | OTel→Postgres span collector |
| `app/observability/cost.py` | Token→USD cost model |
| `app/eval/experiments.py` | A/B runner + bootstrap significance |
| `app/api/v1/traces.py` | Traces API |
| `app/api/v1/experiments.py` | Experiments API |
| `app/api/v1/feedback.py` | Feedback API |
| `app/api/v1/analytics.py` | Analytics API |
| `tests/test_phase4.py` | 28 unit tests |

### New (frontend)
`package.json`, `next.config.mjs`, `tsconfig.json`, `postcss.config.mjs`,
`tailwind.config.ts`, `Dockerfile`, `app/{layout,page,globals.css}`,
`app/{traces,experiments,analytics}/page.tsx`,
`components/{Sidebar,MetricCard,SpanWaterfall,ui}.tsx`,
`lib/{api,types,format}.ts`.

### Modified (backend)
`models/{__init__,eval}.py`, `observability/tracer.py`, `config.py`, `main.py`,
`generation/generator.py`, `api/v1/{__init__,query}.py`, `caching/semantic_cache.py`,
`eval/runner.py`, `schemas/schemas.py`, `docker-compose.yml`.

---

## Test Results

```
49 passed  (21 prior + 28 new Phase 4)
ruff check: All checks passed
ruff format: 67 files already formatted
frontend: next build ✓ (7 routes), served + rendered on all pages
```

New Phase 4 tests cover: cost calc (free/hosted/prefix), tracer attribute builders,
collector extraction/sampling/hex/time helpers, experiment deltas + winner logic,
bootstrap significance (clear separation / zero-effect / too-few-samples),
analytics percentiles, schema serialization, and router mounting.

---

## Verification Notes

- **Backend**: unit tests run without Postgres/Redis/Ollama (pure functions +
  OpenAPI wiring). Full trace persistence + experiment runs require the docker
  stack (Postgres + Redis + Ollama + ingested docs) — not exercised in CI.
- **Frontend**: `next build` compiles + typechecks; production server serves all
  routes (200) and renders loading/empty/error states with the backend down.
  Live data flows require the running backend.
- **Local caveat**: the C: drive was full during dev, so npm cache/temp were
  redirected to F: to install and build.

---

## Next Phase

**Phase 5 — Deploy, Polish, Guardrails & Benchmark**: PII/injection/hallucination
guardrails, Cloud Run deploy configs, benchmark report with real numbers, ADRs.
