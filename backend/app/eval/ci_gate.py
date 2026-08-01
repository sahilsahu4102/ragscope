"""
RAGScope — CI Regression Gate

Runs an evaluation against a committed golden dataset and fails the build if
any gated metric falls below its threshold.

`.github/workflows/evals.yml` referenced this module before it existed, so the
"regression gate" step was commented out and the workflow only ever ran unit
tests. This is the actual gate.

Usage:
    python -m app.eval.ci_gate \
        --dataset eval-datasets/golden-v1.jsonl \
        --faithfulness-threshold 0.80 \
        --context-recall-threshold 0.70 \
        --ndcg-threshold 0.60

Exit codes:
    0 — every gated metric met its threshold
    1 — at least one metric regressed below threshold
    2 — the run could not be completed (bad dataset, pipeline error)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import structlog

from app.db.session import async_session
from app.eval.runner import EvalRunner
from app.models import Dataset

logger = structlog.get_logger()

# Metric key -> CLI flag. ndcg is stored as "ndcg@{k}", so it is resolved by
# prefix rather than exact match.
GATED_METRICS = {
    "faithfulness": "faithfulness_threshold",
    "context_recall": "context_recall_threshold",
    "ndcg": "ndcg_threshold",
}


def load_jsonl(path: Path) -> list[dict]:
    """Load a golden dataset from JSONL (one sample object per line)."""
    samples: list[dict] = []
    with io.open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {e}") from e
            if "question" not in sample:
                raise ValueError(f"{path}:{lineno} has no 'question' field")
            samples.append(sample)
    if not samples:
        raise ValueError(f"{path} contains no samples")
    return samples


def resolve_metric(metrics: dict, key: str) -> tuple[str, float] | None:
    """Find a metric by exact key, else by prefix (handles 'ndcg@10')."""
    if key in metrics:
        return key, float(metrics[key])
    for name, value in metrics.items():
        if name.startswith(key):
            return name, float(value)
    return None


async def run_gate(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"::error::Dataset not found: {dataset_path}")
        return 2

    try:
        samples = load_jsonl(dataset_path)
    except ValueError as e:
        print(f"::error::{e}")
        return 2

    print(f"Loaded {len(samples)} samples from {dataset_path}")

    async with async_session() as session:
        # Ephemeral dataset row — the gate evaluates the committed file, so the
        # source of truth stays in git rather than in whatever is in the DB.
        dataset = Dataset(
            name=f"ci-gate:{dataset_path.stem}",
            version="ci",
            description=f"Ephemeral CI dataset from {dataset_path}",
            sample_count=len(samples),
            samples=samples,
        )
        session.add(dataset)
        await session.flush()

        runner = EvalRunner(session, use_llm_judge=args.use_llm_judge)
        try:
            eval_run = await runner.run(
                dataset=dataset,
                run_name=f"ci-gate:{dataset_path.stem}",
                config_overrides={
                    "use_hybrid": args.use_hybrid,
                    "use_reranker": args.use_reranker,
                    "reranker_backend": args.reranker_backend,
                },
            )
        except Exception as e:
            print(f"::error::Eval run failed: {e}")
            return 2

        metrics = eval_run.metrics or {}
        await session.rollback()  # Don't leave the ephemeral dataset behind.

    if not metrics:
        print("::error::Eval run produced no metrics")
        return 2

    print("\n--- Metrics ---")
    for name in sorted(metrics):
        print(f"  {name}: {metrics[name]}")

    failures: list[str] = []
    checked = 0

    print("\n--- Gate ---")
    for metric_key, arg_name in GATED_METRICS.items():
        threshold = getattr(args, arg_name, None)
        if threshold is None:
            continue

        found = resolve_metric(metrics, metric_key)
        if found is None:
            # A threshold was requested for a metric the run didn't produce.
            # Treat as failure rather than silently passing.
            msg = f"{metric_key}: NOT PRODUCED (threshold {threshold})"
            print(f"  FAIL  {msg}")
            failures.append(msg)
            continue

        name, value = found
        checked += 1
        if value < threshold:
            msg = f"{name}: {value:.4f} < {threshold}"
            print(f"  FAIL  {msg}")
            failures.append(msg)
        else:
            print(f"  PASS  {name}: {value:.4f} >= {threshold}")

    if failures:
        print(f"\n::error::Eval regression gate failed ({len(failures)} metric(s) below threshold)")
        for msg in failures:
            print(f"::error::  {msg}")
        return 1

    print(f"\nEval regression gate passed ({checked} metric(s) checked).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGScope eval regression gate")
    parser.add_argument("--dataset", required=True, help="Path to golden dataset JSONL")
    parser.add_argument("--faithfulness-threshold", type=float, default=None)
    parser.add_argument("--context-recall-threshold", type=float, default=None)
    parser.add_argument("--ndcg-threshold", type=float, default=None)
    parser.add_argument("--use-hybrid", action="store_true", default=True)
    parser.add_argument("--use-reranker", action="store_true", default=True)
    parser.add_argument(
        "--reranker-backend",
        default=None,
        help="'cross_encoder' or 'ollama'. Default: settings.reranker_backend",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        default=False,
        help="Enable LLM-as-judge scoring (slower; needs a reachable model)",
    )
    args = parser.parse_args()

    if all(
        getattr(args, a) is None
        for a in ("faithfulness_threshold", "context_recall_threshold", "ndcg_threshold")
    ):
        print("::error::No thresholds supplied — the gate would pass vacuously.")
        sys.exit(2)

    sys.exit(asyncio.run(run_gate(args)))


if __name__ == "__main__":
    main()
