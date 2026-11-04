"""
RAGScope — End-to-end answer evaluation

Runs each question the way /query does: retrieve top_k = --gen-top-k chunks
(so the reranker scores top_k x 4 candidates, as in /query), then generate
from all of them. Until 2026-11-04 it retrieved 20 chunks (80 candidates) and
generated from the first 3, so its answer scores and its retrieval latency
described a configuration /query never runs. Each answer is scored three ways:
  - heuristics (app/eval/metrics/generation_metrics.py), including fact recall
    against the question's required facts,
  - a hosted judge (app/eval/answer_judge.py) for correctness and faithfulness,
  - latency per question: retrieval, time to first token, total.

    python -m app.scripts.answer_eval run /eval-datasets/golden-v3-test.jsonl --label baseline \\
        --out /eval-datasets/results/baseline.jsonl
    python -m app.scripts.answer_eval compare /eval-datasets/results/baseline.jsonl \\
        /eval-datasets/results/after.jsonl
    python -m app.scripts.answer_eval judge /eval-datasets/results/after.jsonl

`judge` grades the rows the judge skipped (--all: every row), from the context
stored with each row.

`run` prints means with 95% bootstrap intervals and latency percentiles;
`compare` pairs two runs by question and adds paired-bootstrap p-values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

from app.config import settings
from app.db.session import async_session
from app.eval.answer_judge import AnswerJudge
from app.eval.experiments import _bootstrap_significant
from app.eval.metrics.generation_metrics import compute_all_generation_metrics
from app.eval.metrics.retrieval_metrics import full_support_at_k, recall_at_k
from app.generation.generator import Generator, format_context
from app.retrieval.pipeline import RetrievalPipeline

QUALITY = (
    "judge_correctness",
    "judge_faithfulness",
    "fact_recall_all",
    "answer_fact_recall",
    "answer_correctness",
    "faithfulness",
    "abstained",
    "context_full_support",
    "context_recall",
)
LATENCY = ("retrieval_ms", "ttft_ms", "total_ms", "output_tokens")
JUDGE_BATCH = 4


def _load(path: str) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def _bootstrap(values: list[float], stat, iters: int = 2000) -> tuple[float, float]:
    rng = random.Random(42)
    n = len(values)
    samples = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return samples[int(0.025 * iters)], samples[int(0.975 * iters) - 1]


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


async def run(args) -> None:
    samples = [s for path in args.datasets for s in _load(path)][: args.limit or None]
    judge = None
    if not args.no_judge:
        try:
            judge = AnswerJudge()
        except ValueError as e:
            print(f"no judge: {e}")
    generator = Generator()
    results: list[dict] = []

    async with async_session() as db:
        pipeline = RetrievalPipeline(db)

        async def answer(question: str) -> tuple[list[dict], dict, float]:
            start = time.perf_counter()
            chunks = await pipeline.retrieve(
                query=question,
                top_k=args.gen_top_k,
                use_hybrid=True,
                use_reranker=True,
                query_transform=args.query_transform,
                rrf_k=settings.rrf_k,
                strategy=args.strategy,
            )
            retrieval_ms = (time.perf_counter() - start) * 1000
            generated = await generator.generate(question, chunks)
            return chunks, generated, retrieval_ms

        for sample in samples[: args.warmup]:  # load models; not recorded
            await answer(sample["question"])

        for i, sample in enumerate(samples):
            chunks, generated, retrieval_ms = await answer(sample["question"])
            golds = sample.get("gold_contexts", [])
            facts = sample.get("answer_facts")
            row = {
                "question": sample["question"],
                "gold_answer": sample.get("gold_answer", ""),
                "answer": generated["answer"],
                "label": args.label,
                "strategy": args.strategy,
                "retrieval_ms": round(retrieval_ms, 1),
                "ttft_ms": round(retrieval_ms + generated["prefill_ms"], 1),
                "total_ms": round(retrieval_ms + generated["latency_ms"], 1),
                "output_tokens": generated["output_tokens"],
                "context_recall": round(recall_at_k(chunks, golds, args.gen_top_k), 4),
                "context_full_support": full_support_at_k(chunks, golds, args.gen_top_k),
                **compute_all_generation_metrics(
                    sample["question"],
                    generated["answer"],
                    generator.seen_texts(chunks),
                    sample.get("gold_answer"),
                    facts,
                ),
            }
            if facts:
                row["fact_recall_all"] = 1.0 if row["answer_fact_recall"] == 1.0 else 0.0
            row["context"] = format_context(chunks, generator.max_chunk_chars)
            results.append(row)
            print(f"[{i + 1}/{len(samples)}] {row['total_ms']:.0f} ms", flush=True)

    _save(args.out, results)  # generation results survive a judge failure
    if judge:
        # Judged after generation, so the hosted calls never overlap the timed local ones.
        await _judge_rows(judge, results)
        await judge.aclose()
        _save(args.out, results)
    summarize(results, args.label)
    print(f"wrote {len(results)} rows to {args.out}")


async def _judge_rows(judge: AnswerJudge, rows: list[dict]) -> None:
    """Grade rows several per request: free tiers cap requests, not tokens."""
    for start in range(0, len(rows), JUDGE_BATCH):
        batch = rows[start : start + JUDGE_BATCH]
        verdicts = await judge.judge_batch(
            [
                {
                    "question": r["question"],
                    "gold": r["gold_answer"],
                    "context": r["context"],
                    "answer": r["answer"],
                }
                for r in batch
            ]
        )
        for row, verdict in zip(batch, verdicts, strict=True):
            row.pop("judge_error", None)
            row.update(verdict)
        print(f"judged {min(start + JUDGE_BATCH, len(rows))}/{len(rows)}", flush=True)


async def rejudge(args) -> None:
    rows = _load(args.results)
    todo = [r for r in rows if r.get("context") and (args.all or "judge_correctness" not in r)]
    judge = AnswerJudge()
    await _judge_rows(judge, todo)
    await judge.aclose()
    _save(args.results, rows)
    summarize(rows, args.results)


def _save(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict], label: str) -> None:
    print(f"\n{label} (n={len(rows)})\n| metric | mean | 95% CI | n |\n|---|---|---|---|")
    for metric in QUALITY:
        values = [r[metric] for r in rows if isinstance(r.get(metric), (int, float))]
        if values:
            low, high = _bootstrap(values, statistics.fmean)
            print(
                f"| {metric} | {statistics.fmean(values):.3f} | {low:.3f} to {high:.3f} | {len(values)} |"
            )
    errors = sum(1 for r in rows if "judge_error" in r)
    if errors:
        print(f"judge errors: {errors}")
    print("\n| latency | p50 | p50 95% CI | p95 | p95 95% CI |\n|---|---|---|---|---|")
    for metric in LATENCY:
        values = [r[metric] for r in rows]
        p50_ci = _bootstrap(values, lambda v: _pct(v, 0.5))
        p95_ci = _bootstrap(values, lambda v: _pct(v, 0.95))
        print(
            f"| {metric} | {_pct(values, 0.5):.0f} | {p50_ci[0]:.0f} to {p50_ci[1]:.0f} "
            f"| {_pct(values, 0.95):.0f} | {p95_ci[0]:.0f} to {p95_ci[1]:.0f} |"
        )


def compare(args) -> None:
    a = {r["question"]: r for r in _load(args.a)}
    b = {r["question"]: r for r in _load(args.b)}
    shared = [q for q in a if q in b]
    print(f"\n{args.a} vs {args.b} (paired n={len(shared)})")
    print("| metric | A | B | delta | p |\n|---|---|---|---|---|")
    for metric in QUALITY + LATENCY:
        pairs = [
            (a[q][metric], b[q][metric])
            for q in shared
            if isinstance(a[q].get(metric), (int, float))
            and isinstance(b[q].get(metric), (int, float))
        ]
        if not pairs:
            continue
        va, vb = [p[0] for p in pairs], [p[1] for p in pairs]
        _, p_value = _bootstrap_significant(va, vb)
        mean_a, mean_b = statistics.fmean(va), statistics.fmean(vb)
        print(f"| {metric} | {mean_a:.3f} | {mean_b:.3f} | {mean_b - mean_a:+.3f} | {p_value} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("datasets", nargs="+", help="JSONL golden sets")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--strategy", default="standard", choices=["standard", "agentic", "graph"])
    r.add_argument("--query-transform", default="none")
    r.add_argument("--gen-top-k", type=int, default=settings.generation_top_k)
    r.add_argument("--warmup", type=int, default=2)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--no-judge", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    j = sub.add_parser("judge")
    j.add_argument("results")
    j.add_argument("--all", action="store_true", help="re-grade every row, not only skipped ones")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args))
    elif args.command == "judge":
        asyncio.run(rejudge(args))
    else:
        compare(args)


if __name__ == "__main__":
    main()
