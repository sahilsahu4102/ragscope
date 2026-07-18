# Phase 3 — Evaluation Harness

**Git tag:** `v0.4.0-eval-harness`  
**Status:** ✅ Complete  
**Date:** 2026-07-18

---

## Overview

Phase 3 implements RAGScope's **Evaluation Harness** — the differentiator that transforms this from a "PDF chatbot" into a production-grade platform with measurable quality guarantees.

The eval layer is what separates RAGScope from tutorial projects: every accuracy claim carries a before/after number with a baseline, computed via automated metrics and LLM-as-Judge scoring.

---

## What Was Built

### 1. Database Models (`backend/app/models/eval.py`)

| Model | Purpose |
|-------|---------|
| `Dataset` | Versioned golden QA sets (JSONL samples with question, gold_answer, gold_contexts) |
| `EvalRun` | A complete evaluation run with config snapshot + aggregate metrics |
| `EvalSample` | Per-question results: generated answer, retrieved chunks, per-sample metrics, judge reasoning |

### 2. Retrieval Metrics (`backend/app/eval/metrics/retrieval_metrics.py`)

Measures how well the retriever finds the right documents:

| Metric | Description |
|--------|-------------|
| **Context Precision** | Fraction of retrieved chunks that are relevant |
| **Context Recall** | Fraction of gold contexts found by retriever |
| **Hit Rate** | Did at least one relevant chunk appear? (binary) |
| **MRR** | Reciprocal rank of first relevant chunk |
| **NDCG@k** | Ranking quality — rewards relevant docs at higher positions |
| **Precision@k** | Fraction of top-k that are relevant |
| **Recall@k** | Fraction of relevant docs found in top-k |

### 3. Generation Metrics (`backend/app/eval/metrics/generation_metrics.py`)

Measures answer quality:

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevance** | Does the answer address the question? |
| **Answer Correctness** | Similarity to gold reference answer |
| **Hallucination Score** | Fraction of answer not grounded (= 1 - faithfulness) |

### 4. LLM-as-Judge (`backend/app/eval/llm_judge.py`)

Custom LLM judge with bias mitigation — implements techniques from "Judging the Judges" (Shi et al., IJCNLP 2025):

- **Position bias mitigation**: Randomizes context paragraph order, averages across N permutations
- **Verbosity bias correction**: Penalizes answers 3x+ longer than context suggests
- **Reference-guided scoring**: Uses gold answer when available for calibrated assessment
- **Calibration warnings**: Flags high variance across permutations (>1.0) for human review
- **Analytic rubric**: Structured 1-5 scale with explicit criteria per level

### 5. Dataset Generator (`backend/app/eval/dataset_generator.py`)

Generates synthetic golden QA sets from ingested documents:
- **Factual questions** (direct answer in one chunk)
- **Reasoning questions** (requires combining multiple chunks)
- **Analytical questions** (requires interpretation/synthesis)
- Maps source chunks to gold contexts for metric computation

### 6. Eval Runner (`backend/app/eval/runner.py`)

Orchestrates the full evaluation pipeline:

```
Load dataset → For each sample:
  1. Retrieve (hybrid search + rerank)
  2. Generate (grounded answer)
  3. Compute retrieval metrics (vs gold contexts)
  4. Compute generation metrics (vs gold answer)
  5. LLM Judge scoring (faithfulness + relevance)
  6. Store per-sample results
→ Aggregate all metrics → Store eval run
```

Key features:
- Config snapshot for every run (embedding model, chunk size, reranker, top_k, etc.)
- Supports config overrides for A/B testing
- Per-sample and aggregate metrics stored in database

### 7. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/datasets` | GET | List all datasets |
| `/api/v1/datasets/{id}` | GET | Get dataset with samples |
| `/api/v1/datasets/generate` | POST | Generate dataset from documents |
| `/api/v1/datasets/upload` | POST | Upload pre-built dataset |
| `/api/v1/evaluate` | POST | Trigger eval run |
| `/api/v1/evaluate` | GET | List all eval runs |
| `/api/v1/evaluate/{run_id}` | GET | Get detailed results with per-sample metrics |
| `/api/v1/evaluate/{run_id}/gate` | GET | CI regression gate check |

### 8. CI Regression Gate (`.github/workflows/evals.yml`)

- Runs on PRs touching retrieval/generation/eval code
- Executes eval metric unit tests (16 tests)
- Placeholder for full golden dataset regression run
- Threshold assertions: faithfulness ≥ 0.80, context_recall ≥ 0.70

### 9. Tests (`backend/tests/test_eval.py`)

16 comprehensive tests covering:
- Retrieval metrics (recall, hit rate, MRR, NDCG, precision/recall@k)
- Generation metrics (faithfulness, correctness, hallucination)
- LLM judge score parsing and clamping
- Regression gate schema validation

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Eval Runner                       │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │  Dataset  │──▶│ Pipeline │──▶│ Metric Engine  │  │
│  │  Loader   │   │ (R+G)   │   │                │  │
│  └──────────┘   └──────────┘   │ ┌────────────┐ │  │
│                                │ │ Retrieval   │ │  │
│  ┌──────────┐                  │ │ Metrics     │ │  │
│  │  Config   │                 │ ├────────────┤ │  │
│  │ Snapshot  │                 │ │ Generation  │ │  │
│  └──────────┘                  │ │ Metrics     │ │  │
│                                │ ├────────────┤ │  │
│  ┌──────────┐                  │ │ LLM Judge   │ │  │
│  │ Results   │◀────────────────│ │ (bias-aware)│ │  │
│  │ Store     │                 │ └────────────┘ │  │
│  └──────────┘                  └────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Files Created/Modified

### New Files
| File | Lines | Purpose |
|------|-------|---------|
| `backend/app/models/eval.py` | 141 | Dataset, EvalRun, EvalSample DB models |
| `backend/app/eval/metrics/retrieval_metrics.py` | 195 | 7 retrieval quality metrics |
| `backend/app/eval/metrics/generation_metrics.py` | 130 | 4 generation quality metrics |
| `backend/app/eval/llm_judge.py` | 258 | LLM-as-Judge with bias mitigation |
| `backend/app/eval/dataset_generator.py` | 156 | Synthetic QA dataset generation |
| `backend/app/eval/runner.py` | 262 | Full eval run orchestrator |
| `backend/app/api/v1/datasets.py` | 130 | Dataset CRUD API endpoints |
| `backend/app/api/v1/evaluate.py` | 195 | Eval run trigger + regression gate API |
| `backend/tests/test_eval.py` | 168 | 16 unit tests for eval system |

### Modified Files
| File | Change |
|------|--------|
| `backend/app/models/__init__.py` | Register eval models |
| `backend/app/api/v1/__init__.py` | Mount dataset + evaluate routers |
| `backend/app/schemas/schemas.py` | Add eval request/response schemas |
| `backend/app/config.py` | Add eval threshold settings |
| `.github/workflows/evals.yml` | Real CI workflow with tests |

---

## Test Results

```
16 passed in 2.07s

tests/test_eval.py::test_retrieval_metrics_perfect_recall PASSED
tests/test_eval.py::test_retrieval_metrics_zero_recall PASSED
tests/test_eval.py::test_retrieval_metrics_hit_rate_hit PASSED
tests/test_eval.py::test_retrieval_metrics_hit_rate_miss PASSED
tests/test_eval.py::test_retrieval_metrics_mrr PASSED
tests/test_eval.py::test_retrieval_metrics_ndcg PASSED
tests/test_eval.py::test_retrieval_metrics_compute_all PASSED
tests/test_eval.py::test_generation_metrics_faithfulness_grounded PASSED
tests/test_eval.py::test_generation_metrics_faithfulness_hallucinated PASSED
tests/test_eval.py::test_generation_metrics_answer_correctness PASSED
tests/test_eval.py::test_generation_metrics_hallucination_score PASSED
tests/test_eval.py::test_generation_metrics_compute_all PASSED
tests/test_eval.py::test_llm_judge_parse_score PASSED
tests/test_eval.py::test_llm_judge_parse_score_clamped PASSED
tests/test_eval.py::test_regression_gate_schema PASSED
```

---

## Resume Bullet (fill in measured numbers after running on real data)

> "Designed the platform's evaluation + observability layer — custom position-bias-corrected LLM-as-judge with analytic rubrics, 7 retrieval metrics (NDCG, MRR, context precision/recall), CI regression gates — cutting measured hallucination rate from X% to Y% on a Z-sample weekly eval with no helpfulness regression."
