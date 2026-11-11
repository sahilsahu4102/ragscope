# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: a recruiter or engineer looking at Sahil Kumar Sahu's flagship project. Today they see it as screenshots and a demo video (README, CV, portfolio); later they may open a live demo that Sahil deploys. They decide in a few seconds whether the evaluation and observability work behind it is real or decoration.

Secondary, inside the tool: the person running the stack (Sahil, or a reviewer who clones it) who asks questions against ingested documents, inspects retrieval, reads query traces, runs A/B experiments between two pipeline settings, and checks latency and cost.

## Product Purpose

RAGScope is a self-hosted retrieval-augmented generation platform with its own evaluation and observability layer. The web UI is the console for it: upload documents, ask grounded questions with citations, look at every retrieved chunk with its dense, BM25, fusion and rerank scores, read the span timeline of any query (retriever, reranker, model), compare two configurations on a golden set with significance tests, and watch latency, cost and cache behaviour.

Success: a viewer sees that every claim on screen is a measurement the system made, and can follow any answer back to its chunks, scores, spans and timings.

## Positioning

The console shows its working. Every number on a screen is read from the running API (a trace, an experiment, an analytics query), and any answer can be traced back to its evidence. A generic chat or admin dashboard does not expose retrieval stages, span timings and paired-bootstrap comparisons side by side.

## Operating Context

- Local-first. The stack is Docker Compose (Postgres with pgvector, Redis, Ollama, the FastAPI backend, a Celery worker, this Next.js frontend). It needs the whole stack running; on the owner's 16 GB machine the first request after start can be slow.
- Not deployed yet. A live demo is planned later, so first-open states must handle an unreachable or still-starting backend and an empty database honestly.
- Screenshots and video are the main way it is seen, so every screen must read well frozen, with real data on it.

## Capabilities and Constraints

- Pages today: Overview, Documents (upload, ingestion status, delete), Playground (grounded answers, citations, latency and token counts, thumbs feedback), Retrieval Inspector (retrieval without generation, per-stage scores, metadata filters), Trace Viewer (span waterfall with duration, tokens and cost), Experiments (A/B on a golden dataset with bootstrap significance), Analytics (p50/p95/p99 latency, cost by model, cache hit rate, throughput).
- The Overview shows live status and latest activity from the API (health and readiness, documents ingested, recent traces and experiments). Nothing on it is hard-coded.
- Data comes only from the API under `/api/v1` (plus `/healthz` and `/readyz`). The UI must never invent a figure, a sample query or a benchmark. Measured benchmark results live in the README and `docs/`, not in the UI.
- Version and "Phase 4" labels in the old UI were stale (the project is well past Phase 4); show no version or phase text that is not read from the running system.
- Constraints: Next.js 15 App Router, React 19, Tailwind 3, Recharts. `cd frontend && npm run build` must stay green (CI runs it). A dependency change needs the lockfile checked with the CI's npm.
- Uncommitted deployment work (Dockerfiles, Makefile, Caddyfile, `deployment/`) exists in the repo working tree and is not part of the redesign: never stage it.

## Brand Commitments

Name: RAGScope. Honest framing: a self-hosted evaluation and observability platform for RAG. No claim of users, traffic or production scale.

## Evidence on Hand

- Real measurements exist in `docs/eval-results.md` and the README (for example MultiHop-RAG retrieval, latency, the reranker A/B), produced by the system itself.
- The database holds ingested documents and traces from the owner's runs; screenshots must be taken from real data.
- No user study, no deployment, no traffic figures. Do not invent any.

## Product Principles

1. Show the working: every figure traces to a trace, a run or a query.
2. Read from the system, never from the page: no hard-coded stats, versions or sample content.
3. Retrieval stages, scores and timings are the content; give them room and a consistent notation.
4. Waiting and failure are states with words: a stack that is starting, a model that is slow, an empty database.
5. Scanning beats decoration: dense, aligned, tabular, readable when frozen in a screenshot.

## Accessibility & Inclusion

Target WCAG 2.2 AA: contrast (state and span kind never by colour alone), visible keyboard focus, labelled controls, reduced motion respected, no horizontal scroll at 390 px.
