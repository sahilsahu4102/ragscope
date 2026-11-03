"""
RAGScope — Retrieval-only A/B

Compares two retrieval configurations on stored golden datasets with the eval
harness's retrieval metrics and the same paired bootstrap as the experiments
API, without generating answers. Retrieval metrics do not depend on the answer,
and generation is what made a full experiment (POST /experiments) take the
better part of an hour per dataset on a 16 GB laptop and run it out of memory.

    python -m app.scripts.retrieval_ab --datasets multihop-v1 golden-v2 --b retrieval_strategy=agentic
    python -m app.scripts.retrieval_ab --datasets multihop-v1 --a query_transform=decompose \\
        --b retrieval_strategy=agentic

Variant settings match EvalRunner's defaults (hybrid, reranker on, top_k and
rrf_k from settings); --a/--b override query_transform and retrieval_strategy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from sqlalchemy import select

from app.config import settings
from app.db.session import async_session
from app.eval.experiments import _bootstrap_significant
from app.eval.metrics.retrieval_metrics import compute_all_retrieval_metrics
from app.models import Dataset
from app.retrieval.pipeline import RetrievalPipeline

METRICS = ("ndcg@10", "mrr", "recall@10", "full_support@10", "hit_rate")


def _variant(pairs: list[str]) -> dict:
    return dict(pair.split("=", 1) for pair in pairs)


async def _load(name: str) -> list[dict]:
    async with async_session() as db:
        dataset = (
            (
                await db.execute(
                    select(Dataset).where(Dataset.name == name).order_by(Dataset.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
    if dataset is None:
        raise SystemExit(f"no dataset named {name!r}")
    return dataset.samples or []


async def _run(samples: list[dict], variant: dict) -> tuple[list[dict], list[float]]:
    metrics, latencies = [], []
    async with async_session() as db:
        pipeline = RetrievalPipeline(db)
        for sample in samples:
            start = time.perf_counter()
            chunks = await pipeline.retrieve(
                query=sample["question"],
                top_k=settings.retrieval_top_k,
                use_hybrid=True,
                use_reranker=True,
                query_transform=variant.get("query_transform", "none"),
                rrf_k=settings.rrf_k,
                strategy=variant.get("retrieval_strategy", "standard"),
            )
            latencies.append((time.perf_counter() - start) * 1000)
            metrics.append(
                compute_all_retrieval_metrics(chunks, sample.get("gold_contexts", []), k=10)
            )
    return metrics, latencies


async def compare(name: str, a: dict, b: dict) -> dict:
    samples = await _load(name)
    metrics_a, latency_a = await _run(samples, a)
    metrics_b, latency_b = await _run(samples, b)

    result: dict = {"dataset": name, "n": len(samples), "a": a, "b": b, "metrics": {}}
    for metric in METRICS:
        values_a = [m[metric] for m in metrics_a]
        values_b = [m[metric] for m in metrics_b]
        _, p_value = _bootstrap_significant(values_a, values_b)
        result["metrics"][metric] = {
            "a": round(statistics.fmean(values_a), 4),
            "b": round(statistics.fmean(values_b), 4),
            "p": p_value,
        }
    result["retrieval_ms_p50"] = {
        "a": round(statistics.median(latency_a)),
        "b": round(statistics.median(latency_b)),
    }
    result["per_sample"] = {
        "a": [m["full_support@10"] for m in metrics_a],
        "b": [m["full_support@10"] for m in metrics_b],
    }
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", required=True, help="names as stored in Postgres")
    parser.add_argument("--a", nargs="*", default=[], metavar="KEY=VALUE", help="variant A")
    parser.add_argument("--b", nargs="*", default=[], metavar="KEY=VALUE", help="variant B")
    args = parser.parse_args()

    for name in args.datasets:
        result = await compare(name, _variant(args.a), _variant(args.b))
        print(
            f"\n{name} (n={result['n']})  A={result['a'] or 'standard'}  B={result['b'] or 'standard'}"
        )
        print("| metric | A | B | delta | p |\n|---|---|---|---|---|")
        for metric, m in result["metrics"].items():
            print(f"| {metric} | {m['a']:.4f} | {m['b']:.4f} | {m['b'] - m['a']:+.4f} | {m['p']} |")
        lat = result["retrieval_ms_p50"]
        print(f"| retrieval p50 (ms) | {lat['a']} | {lat['b']} | | |")
        print("RESULT_JSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
