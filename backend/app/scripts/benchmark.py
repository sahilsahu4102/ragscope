"""
RAGScope — Benchmarking Script (Phase 5)

Measures end-to-end RAG pipeline performance:
  - Full pipeline latency (p50/p95/p99)
  - Retrieval latency by mode (dense, hybrid, hybrid+rerank)
  - Cache performance (hit rate, latency delta)
  - Throughput (queries per second)

Usage:
    python -m app.scripts.benchmark [--queries 20] [--warmup 3] [--base-url http://localhost:8000]

Output: Markdown table suitable for README.md
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

# ── Benchmark Questions ──────────────────────────
# Sourced from eval-datasets/golden-v2.jsonl so the benchmark queries are
# actually answerable from the ingested corpus. The original list asked about
# RAG concepts (BM25, RRF, semantic caching) that appear nowhere in the
# indexed documents, so every measurement ran on near-miss retrievals.
DEFAULT_QUESTIONS = [
    "What is the range of per-GPU model state size saved during Llama 3 checkpointing?",
    "In the Transformer's position-wise feed-forward network, how do the linear transformations vary across positions compared to across layers?",
    "Why does the synchronous nature of Llama 3's 16K-GPU training make it less fault-tolerant?",
    "Why are scaling law experiments necessary for training models on large GPUs?",
    "What is the main challenge in learning long-range dependencies in sequence transduction tasks?",
    "How are the image adapters pre-trained?",
    "Why did the authors choose to use sinusoidal positional embeddings instead of learned positional embeddings?",
    "How does the iterative process of using feedback from incorrect attempts and correcting them help improve Llama 3's ability to reason accurately?",
    "How does the Llama 3 model handle code-switched speech?",
    "What is multitask prompted training, and how does it enable zero-shot task generalization?",
    "How many parameters does the image encoder have with the additional layers?",
    "Why did the authors choose not to freeze the language RM part during training?",
    "How does the fundamental constraint of sequential computation impact the efficiency of computational models?",
    "What is the primary benefit of using Llama 405B with Llama Guard compared to competing systems?",
    "How does the reward model contribute to the safety of the Llama 3 model?",
    "What is the purpose of using an attention mask in sequence processing?",
    "How do prompt-based system guards work to improve LLM safety and control user requests?",
    "What percentage of prompt injection attacks against Llama 3 405B were successful?",
    "Why do pre-trained models perform better on paraphrase detection than post-trained models?",
    "Why did the Llama 3 authors modify their pipeline parallelism schedule to allow setting N flexibly?",
]


async def _query(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    use_hybrid: bool = False,
    use_reranker: bool = False,
    use_cache: bool = True,
) -> dict:
    """Send a single query and return timing + metadata."""
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url}/api/v1/query",
            json={
                "question": question,
                "top_k": 5,
                "use_hybrid": use_hybrid,
                "use_reranker": use_reranker,
                "use_cache": use_cache,
                "stream": False,
                "query_transform": "none",
            },
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "elapsed_ms": elapsed_ms,
                "api_latency_ms": data.get("latency_ms", elapsed_ms),
                "cached": data.get("cached", False),
                "citations": len(data.get("citations", [])),
                "tokens": data.get("tokens_used", 0),
            }
        else:
            return {
                "success": False,
                "elapsed_ms": elapsed_ms,
                "error": response.text[:200],
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "success": False,
            "elapsed_ms": elapsed_ms,
            "error": str(e)[:200],
        }


def _percentiles(values: list[float]) -> dict:
    """Compute p50, p95, p99 percentiles."""
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "p50": round(sorted_vals[int(n * 0.50)], 1),
        "p95": round(sorted_vals[int(n * 0.95)], 1),
        "p99": round(sorted_vals[int(n * 0.99)], 1),
        "mean": round(statistics.mean(sorted_vals), 1),
        "min": round(sorted_vals[0], 1),
        "max": round(sorted_vals[-1], 1),
    }


async def run_benchmark(
    base_url: str = "http://localhost:8000",
    num_queries: int = 20,
    warmup: int = 3,
    skip_throughput: bool = False,
) -> dict:
    """Run the full benchmark suite."""
    questions = DEFAULT_QUESTIONS[:num_queries]
    results: dict = {}

    async with httpx.AsyncClient(timeout=300.0) as client:
        # ── Health check ─────────────────────────
        print("🔍 Checking API health...")
        try:
            health = await client.get(f"{base_url}/healthz")
            if health.status_code != 200:
                print(f"❌ API not healthy: {health.text}")
                return {}
            print("✅ API is healthy")
        except Exception as e:
            print(f"❌ Cannot reach API at {base_url}: {e}")
            return {}

        # ── Check if documents are ingested ──────
        try:
            retrieve_test = await client.post(
                f"{base_url}/api/v1/query",
                json={
                    "question": "test",
                    "top_k": 1,
                    "use_cache": False,
                    "stream": False,
                },
            )
            if retrieve_test.status_code == 404:
                print("⚠️  No documents ingested — retrieval benchmarks will fail.")
                print("   Run: make seed or upload documents first.")
        except Exception:
            pass

        # ── Warmup ───────────────────────────────
        print(f"\n🔄 Warming up ({warmup} queries)...")
        for i in range(min(warmup, len(questions))):
            await _query(client, base_url, questions[i], use_cache=False)
            print(f"   Warmup {i + 1}/{warmup}")

        # ── 1. Dense-only retrieval ──────────────
        print("\n📊 Benchmark 1: Dense-only retrieval...")
        dense_latencies = []
        for i, q in enumerate(questions):
            result = await _query(
                client, base_url, q,
                use_hybrid=False, use_reranker=False, use_cache=False,
            )
            if result["success"]:
                dense_latencies.append(result["api_latency_ms"])
            print(f"   [{i + 1}/{len(questions)}] {'✅' if result['success'] else '❌'} {result.get('api_latency_ms', result['elapsed_ms']):.0f}ms")
        results["dense"] = _percentiles(dense_latencies)
        results["dense"]["count"] = len(dense_latencies)

        # ── 2. Hybrid (dense + BM25 + RRF) ──────
        print("\n📊 Benchmark 2: Hybrid retrieval (dense + BM25 + RRF)...")
        hybrid_latencies = []
        for i, q in enumerate(questions):
            result = await _query(
                client, base_url, q,
                use_hybrid=True, use_reranker=False, use_cache=False,
            )
            if result["success"]:
                hybrid_latencies.append(result["api_latency_ms"])
            print(f"   [{i + 1}/{len(questions)}] {'✅' if result['success'] else '❌'} {result.get('api_latency_ms', result['elapsed_ms']):.0f}ms")
        results["hybrid"] = _percentiles(hybrid_latencies)
        results["hybrid"]["count"] = len(hybrid_latencies)

        # ── 3. Hybrid + Rerank ───────────────────
        print("\n📊 Benchmark 3: Hybrid + Rerank...")
        rerank_latencies = []
        for i, q in enumerate(questions):
            result = await _query(
                client, base_url, q,
                use_hybrid=True, use_reranker=True, use_cache=False,
            )
            if result["success"]:
                rerank_latencies.append(result["api_latency_ms"])
            print(f"   [{i + 1}/{len(questions)}] {'✅' if result['success'] else '❌'} {result.get('api_latency_ms', result['elapsed_ms']):.0f}ms")
        results["rerank"] = _percentiles(rerank_latencies)
        results["rerank"]["count"] = len(rerank_latencies)

        # ── 4. Cache performance ─────────────────
        print("\n📊 Benchmark 4: Semantic cache performance...")
        # Clear cache first
        try:
            await client.delete(f"{base_url}/api/v1/query/cache")
        except Exception:
            pass

        # First pass: cache MISS (populate cache)
        miss_latencies = []
        for q in questions[:5]:
            result = await _query(
                client, base_url, q,
                use_hybrid=False, use_reranker=False, use_cache=True,
            )
            if result["success"]:
                miss_latencies.append(result["api_latency_ms"])

        # Second pass: cache HIT (should be much faster)
        hit_latencies = []
        for q in questions[:5]:
            result = await _query(
                client, base_url, q,
                use_hybrid=False, use_reranker=False, use_cache=True,
            )
            if result["success"] and result.get("cached"):
                hit_latencies.append(result["api_latency_ms"])

        results["cache_miss"] = _percentiles(miss_latencies) if miss_latencies else {}
        results["cache_hit"] = _percentiles(hit_latencies) if hit_latencies else {}
        results["cache_hit"]["speedup"] = (
            round(results["cache_miss"].get("p50", 1) / max(results["cache_hit"].get("p50", 1), 0.1), 1)
            if miss_latencies and hit_latencies
            else 0
        )

        # ── 5. Throughput (concurrent queries) ───
        # 10 simultaneous generations against a single 4GB-VRAM GPU is the
        # heaviest thing this suite does. Skippable so latency percentiles can
        # be collected without it.
        if skip_throughput:
            print("\n⏭  Benchmark 5: Throughput — skipped (--skip-throughput)")
        else:
            print("\n📊 Benchmark 5: Throughput (concurrent)...")
            throughput_start = time.perf_counter()
            concurrent_results = await asyncio.gather(
                *[
                    _query(client, base_url, q, use_cache=False)
                    for q in questions[:10]
                ]
            )
            throughput_elapsed = time.perf_counter() - throughput_start
            successful = sum(1 for r in concurrent_results if r["success"])
            qps = round(successful / max(throughput_elapsed, 0.01), 2)
            results["throughput"] = {
                "concurrent_queries": 10,
                "successful": successful,
                "total_time_s": round(throughput_elapsed, 2),
                "qps": qps,
            }

    return results


def format_results(results: dict) -> str:
    """Format benchmark results as a markdown table."""
    if not results:
        return "❌ No benchmark results — API unreachable or no documents ingested."

    lines = [
        "# RAGScope Benchmark Results",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Latency by Retrieval Mode",
        "",
        "| Mode | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Queries |",
        "|------|----------|----------|----------|-----------|---------|",
    ]

    for mode, label in [("dense", "Dense only"), ("hybrid", "Hybrid (RRF)"), ("rerank", "Hybrid + Rerank")]:
        d = results.get(mode, {})
        if d:
            lines.append(
                f"| {label} | {d.get('p50', '—')} | {d.get('p95', '—')} | "
                f"{d.get('p99', '—')} | {d.get('mean', '—')} | {d.get('count', 0)} |"
            )

    lines.extend([
        "",
        "## Cache Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])

    if results.get("cache_miss"):
        lines.append(f"| Cache Miss p50 | {results['cache_miss'].get('p50', '—')} ms |")
    if results.get("cache_hit"):
        lines.append(f"| Cache Hit p50 | {results['cache_hit'].get('p50', '—')} ms |")
        lines.append(f"| Speedup | {results['cache_hit'].get('speedup', '—')}x |")

    lines.extend([
        "",
        "## Throughput",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])

    if results.get("throughput"):
        t = results["throughput"]
        lines.append(f"| Concurrent queries | {t.get('concurrent_queries', 0)} |")
        lines.append(f"| Successful | {t.get('successful', 0)} |")
        lines.append(f"| QPS | {t.get('qps', 0)} |")
        lines.append(f"| Total time | {t.get('total_time_s', 0)}s |")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="RAGScope Benchmark Suite")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries per benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup queries")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", type=str, default=None, help="Output file path for results")
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="Skip the 10-way concurrent phase (heaviest load on a single GPU)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🔬 RAGScope Benchmark Suite (Phase 5)")
    print("=" * 60)
    print(f"  API: {args.base_url}")
    print(f"  Queries: {args.queries} | Warmup: {args.warmup}")
    print("=" * 60)

    results = await run_benchmark(
        base_url=args.base_url,
        num_queries=args.queries,
        warmup=args.warmup,
        skip_throughput=args.skip_throughput,
    )

    report = format_results(results)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\n📄 Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
