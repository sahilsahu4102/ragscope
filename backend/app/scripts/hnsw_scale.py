"""
RAGScope — pgvector index scaling benchmark

Answers "does retrieval hold up as the corpus grows, and does HNSW earn its
keep?" with measurements rather than assertion.

The real corpus is only a few thousand chunks, which is too small to show
anything: a sequential scan over it takes ~15-25ms, so an ANN index looks
pointless. To measure the scaling curve honestly, this builds a scratch table
seeded with the real embeddings and grown with perturbed copies of them.

Why perturbed real vectors rather than random ones: uniformly random vectors
sit nowhere near the manifold real embeddings occupy, and ANN recall depends
heavily on that structure. Perturbing real vectors keeps the neighbourhood
geometry roughly realistic, so recall numbers mean something.

What this does and does not claim:
  - index latency and recall at scale: measured, and valid
  - retrieval *quality* at scale: NOT measured here. Synthetic chunks have no
    meaningful text, so NDCG/recall against gold contexts stays on the real
    corpus (see docs/eval-results.md).

Recall is measured against exact search on the same table, which is the
standard definition: |ANN top-k INTERSECT exact top-k| / k.

Usage:
    python -m app.scripts.hnsw_scale --sizes 2000 10000 50000
"""

from __future__ import annotations

import argparse
import asyncio
import io
import statistics
import time

import numpy as np
from sqlalchemy import text

from app.db.session import async_session

SCALE_TABLE = "chunks_scale"
DIM = 768
N_QUERIES = 20
TOP_K = 10


async def fetch_real_embeddings(session) -> np.ndarray:
    """Pull the real corpus embeddings as the seed for synthetic expansion."""
    rows = (
        await session.execute(text("SELECT embedding FROM chunks WHERE embedding IS NOT NULL"))
    ).fetchall()
    if not rows:
        raise SystemExit("no embeddings in chunks — ingest documents first")

    vecs = np.array(
        [np.fromstring(r[0].strip("[]"), sep=",", dtype=np.float32) for r in rows],
        dtype=np.float32,
    )
    print(f"seed: {vecs.shape[0]} real embeddings, dim={vecs.shape[1]}")
    return vecs


def synth_batch(seed: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n vectors formed by perturbing randomly chosen real ones, re-normalised."""
    idx = rng.integers(0, seed.shape[0], size=n)
    base = seed[idx]
    noise = rng.normal(0.0, 0.05, size=base.shape).astype(np.float32)
    out = base + noise
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-8)


async def build_table(session, seed: np.ndarray, target: int, rng) -> None:
    """(Re)create the scratch table and fill it to `target` rows via COPY."""
    await session.execute(text(f"DROP TABLE IF EXISTS {SCALE_TABLE}"))
    await session.execute(
        text(f"CREATE TABLE {SCALE_TABLE} (id bigserial PRIMARY KEY, embedding vector({DIM}))")
    )
    await session.commit()

    raw = await session.connection()
    driver = await raw.get_raw_connection()
    asyncpg_conn = driver.driver_connection

    batch_rows = 5000
    written = 0
    t0 = time.perf_counter()
    while written < target:
        n = min(batch_rows, target - written)
        vecs = synth_batch(seed, n, rng)
        # asyncpg's COPY wants bytes. Text format (the default) is used rather
        # than CSV because a vector literal is full of commas.
        buf = io.BytesIO()
        for v in vecs:
            buf.write(("[" + ",".join(f"{x:.6f}" for x in v) + "]\n").encode())
        buf.seek(0)
        await asyncpg_conn.copy_to_table(SCALE_TABLE, source=buf, columns=["embedding"])
        written += n
    await session.commit()
    print(f"  populated {target} rows in {time.perf_counter() - t0:.1f}s")


async def search(session, qvec: np.ndarray, use_index: bool, ef: int | None) -> list[int]:
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in qvec) + "]"
    if use_index:
        if ef:
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef}"))
        await session.execute(text("SET LOCAL enable_seqscan = on"))
    else:
        # Force exact search for ground truth.
        await session.execute(text("SET LOCAL enable_indexscan = off"))
        await session.execute(text("SET LOCAL enable_bitmapscan = off"))

    rows = (
        await session.execute(
            text(
                f"SELECT id FROM {SCALE_TABLE} "
                f"ORDER BY embedding <=> CAST(:q AS vector) LIMIT {TOP_K}"
            ),
            {"q": vec_literal},
        )
    ).fetchall()
    return [r[0] for r in rows]


async def timed(session, queries, use_index, ef) -> tuple[float, float, list[list[int]]]:
    """Return (p50_ms, mean_ms, results) over the query set."""
    lat: list[float] = []
    results: list[list[int]] = []
    for q in queries:
        t0 = time.perf_counter()
        ids = await search(session, q, use_index, ef)
        lat.append((time.perf_counter() - t0) * 1000)
        results.append(ids)
    lat.sort()
    return lat[len(lat) // 2], statistics.mean(lat), results


def recall_at_k(exact: list[list[int]], approx: list[list[int]]) -> float:
    scores = [len(set(e) & set(a)) / max(len(e), 1) for e, a in zip(exact, approx, strict=True)]
    return sum(scores) / len(scores)


async def run(sizes: list[int]) -> None:
    rng = np.random.default_rng(42)
    report: list[dict] = []

    async with async_session() as session:
        seed = await fetch_real_embeddings(session)
        queries = [seed[i] for i in rng.integers(0, seed.shape[0], size=N_QUERIES)]

        for size in sizes:
            print(f"\n=== corpus size: {size:,} ===")
            await build_table(session, seed, size, rng)

            # Exact / sequential scan, no index present.
            exact_p50, exact_mean, exact_ids = await timed(session, queries, False, None)
            print(f"  exact (seq scan)   p50={exact_p50:8.1f} ms  mean={exact_mean:8.1f} ms")

            # Build HNSW.
            t0 = time.perf_counter()
            await session.execute(
                text(
                    f"CREATE INDEX ON {SCALE_TABLE} USING hnsw (embedding vector_cosine_ops) "
                    f"WITH (m = 16, ef_construction = 64)"
                )
            )
            await session.commit()
            build_s = time.perf_counter() - t0
            print(f"  hnsw build         {build_s:.1f}s")

            row = {"size": size, "exact_p50": exact_p50, "build_s": build_s}
            for ef in (40, 100):
                p50, mean, ids = await timed(session, queries, True, ef)
                rec = recall_at_k(exact_ids, ids)
                speedup = exact_p50 / p50 if p50 else 0
                print(
                    f"  hnsw ef={ef:<4}       p50={p50:8.1f} ms  mean={mean:8.1f} ms  "
                    f"recall@{TOP_K}={rec:.3f}  speedup={speedup:.1f}x"
                )
                row[f"hnsw_ef{ef}_p50"] = p50
                row[f"hnsw_ef{ef}_recall"] = rec
                row[f"hnsw_ef{ef}_speedup"] = speedup
            report.append(row)

        await session.execute(text(f"DROP TABLE IF EXISTS {SCALE_TABLE}"))
        await session.commit()

    print("\n\n| corpus | exact p50 | hnsw ef=40 | recall@10 | speedup | build |")
    print("|---|---|---|---|---|---|")
    for r in report:
        print(
            f"| {r['size']:,} | {r['exact_p50']:.1f} ms | {r['hnsw_ef40_p50']:.1f} ms | "
            f"{r['hnsw_ef40_recall']:.3f} | {r['hnsw_ef40_speedup']:.1f}x | {r['build_s']:.1f}s |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="pgvector HNSW scaling benchmark")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 10000, 50000])
    args = ap.parse_args()
    asyncio.run(run(args.sizes))


if __name__ == "__main__":
    main()
