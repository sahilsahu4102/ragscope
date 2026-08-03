"""
RAGScope — Latency configuration sweep

Measures TTFT and total latency across pipeline configurations and writes a
CSV, so the choice of configuration is made from data rather than intuition.

Two things this deliberately does:

  - Measures TTFT from the streaming endpoint, not just total time. TTFT is
    what a user perceives, and it is set by retrieval + prefill; total time is
    dominated by how many tokens the model decides to emit.

  - Records answer length alongside latency. A configuration can look fast
    purely because it produced a shorter answer, which is not the same as
    being faster. Comparing ms/token separates the two.

Latency alone cannot pick a winner — a fast configuration that retrieves the
wrong chunks is worse than a slow correct one. Feed the shortlist from here
into the eval harness (docs/eval-results.md) before choosing.

Usage:
    python -m app.scripts.latency_sweep --repeats 3 --out /app/sweep.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import statistics
import time

import httpx

BASE = "http://localhost:8000"
# Questions come from the API rather than the JSONL on disk: eval-datasets/
# lives at the repo root and only ./backend is mounted into the container.
DATASET_NAME = "golden-v2"


async def load_questions(client: httpx.AsyncClient, limit: int) -> list[str]:
    listing = (await client.get(f"{BASE}/api/v1/datasets")).json()
    match = next((d for d in listing if d["name"] == DATASET_NAME), None)
    if match is None:
        raise SystemExit(f"dataset {DATASET_NAME!r} not found — upload it first")

    detail = (await client.get(f"{BASE}/api/v1/datasets/{match['id']}")).json()
    return [s["question"] for s in detail["samples"]][:limit]


# Each entry is a named configuration. Only fields that differ from the
# defaults are set, so the table stays readable.
CONFIGS: list[dict] = [
    {"name": "baseline (top_k=5, rerank, hybrid)"},
    {"name": "no reranker", "use_reranker": False},
    {"name": "no hybrid (dense only)", "use_hybrid": False},
    {"name": "dense only, no rerank", "use_hybrid": False, "use_reranker": False},
    {"name": "top_k=3", "top_k": 3},
    {"name": "top_k=8", "top_k": 8},
    {"name": "chunk<=400 chars", "max_chunk_chars": 400},
    {"name": "chunk<=800 chars", "max_chunk_chars": 800},
    {"name": "num_predict=256", "num_predict": 256},
    {"name": "num_predict=512", "num_predict": 512},
    {"name": "top_k=3 + chunk<=400", "top_k": 3, "max_chunk_chars": 400},
    {
        "name": "top_k=3 + chunk<=400 + np=256",
        "top_k": 3,
        "max_chunk_chars": 400,
        "num_predict": 256,
    },
    {
        "name": "lean: tk3 + 400ch + np256 + no rerank",
        "top_k": 3,
        "max_chunk_chars": 400,
        "num_predict": 256,
        "use_reranker": False,
    },
    {"name": "1b model", "model": "llama3.2:1b"},
    {
        "name": "1b + tk3 + 400ch + np256",
        "model": "llama3.2:1b",
        "top_k": 3,
        "max_chunk_chars": 400,
        "num_predict": 256,
    },
]

BASE_PAYLOAD = {
    "top_k": 5,
    "use_hybrid": True,
    "use_reranker": True,
    "use_cache": False,
    "query_transform": "none",
}


async def measure_stream(client: httpx.AsyncClient, payload: dict) -> tuple[float, float, int]:
    """Return (ttft_ms, total_ms, answer_chars) for one streaming request."""
    start = time.perf_counter()
    ttft = None
    chars = 0
    async with client.stream("POST", f"{BASE}/api/v1/query/stream", json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body.strip() == "[DONE]":
                break
            try:
                tok = json.loads(body).get("token", "")
            except json.JSONDecodeError:
                continue
            if tok:
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000
                chars += len(tok)
    total = (time.perf_counter() - start) * 1000
    return (ttft if ttft is not None else total), total, chars


async def run(repeats: int, num_questions: int, out_path: str) -> None:
    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=600.0) as client:
        questions = await load_questions(client, num_questions)
        print(f"loaded {len(questions)} questions from {DATASET_NAME}")

        # Warm the model + reranker so the first config is not penalised.
        await measure_stream(client, {**BASE_PAYLOAD, "question": questions[0]})

        for cfg in CONFIGS:
            name = cfg["name"]
            payload_base = {**BASE_PAYLOAD, **{k: v for k, v in cfg.items() if k != "name"}}
            print(f"\n=== {name} ===")

            # Warm this specific config (model swaps need a load).
            try:
                await measure_stream(client, {**payload_base, "question": questions[0]})
            except Exception as e:
                print(f"  warmup failed: {e}")
                continue

            ttfts: list[float] = []
            totals: list[float] = []
            chars: list[int] = []
            failures = 0

            for q in questions * repeats:
                try:
                    t, tot, c = await measure_stream(client, {**payload_base, "question": q})
                    ttfts.append(t)
                    totals.append(tot)
                    chars.append(c)
                except Exception as e:
                    failures += 1
                    print(f"  request failed: {str(e)[:90]}")

            if not ttfts:
                print("  no successful requests")
                continue

            med = statistics.median
            row = {
                "config": name,
                "ttft_p50_ms": round(med(ttfts), 1),
                "total_p50_ms": round(med(totals), 1),
                "answer_chars_p50": int(med(chars)),
                # Separates "fast" from "produced less output".
                "ms_per_char": round(med(totals) / max(med(chars), 1), 2),
                "n": len(ttfts),
                "failures": failures,
                "top_k": payload_base["top_k"],
                "use_hybrid": payload_base["use_hybrid"],
                "use_reranker": payload_base["use_reranker"],
                "num_predict": payload_base.get("num_predict", "default"),
                "max_chunk_chars": payload_base.get("max_chunk_chars", "default"),
                "model": payload_base.get("model", "default"),
            }
            rows.append(row)
            print(
                f"  TTFT p50 {row['ttft_p50_ms']:8.1f} ms | "
                f"total p50 {row['total_p50_ms']:8.1f} ms | "
                f"answer {row['answer_chars_p50']:5} chars | "
                f"{row['ms_per_char']:6.2f} ms/char"
            )

    if not rows:
        print("no results")
        return

    # Print the table before writing the file. A failed write must not lose
    # results that took 20 minutes to produce — this ran once and produced no
    # CSV and no table, because the write raised before anything was printed.
    print("\n| config | TTFT p50 | total p50 | answer chars | ms/char |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x["ttft_p50_ms"]):
        print(
            f"| {r['config']} | {r['ttft_p50_ms']} ms | {r['total_p50_ms']} ms | "
            f"{r['answer_chars_p50']} | {r['ms_per_char']} |"
        )

    try:
        with io.open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_path} ({len(rows)} configs)")
    except OSError as e:
        print(f"\nCSV write to {out_path} failed ({e}) — table above is the result")


def main() -> None:
    ap = argparse.ArgumentParser(description="Latency configuration sweep")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--out", type=str, default="/app/sweep.csv")
    args = ap.parse_args()
    asyncio.run(run(args.repeats, args.questions, args.out))


if __name__ == "__main__":
    main()
