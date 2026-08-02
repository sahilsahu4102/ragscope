"""
RAGScope — Sparse retrieval scaling: in-process BM25 vs Postgres FTS

At 5,869 chunks the in-process BM25 index is *faster* than a GIN-indexed
tsvector (13.8ms vs 27.6ms p50), which is the opposite of the assumption that
motivated adding FTS. numpy scoring a few thousand documents in RAM beats an
index lookup plus ts_rank_cd plus a sort.

BM25 is still O(N) in the corpus and FTS is not, so there is a crossover. This
finds it rather than assuming it, because "replace BM25 with FTS" is only
correct on the far side of that point.

Both backends are measured over the same synthetic text corpus, built by
sampling real chunk content so term statistics stay realistic.

Usage:
    python -m app.scripts.sparse_scale --sizes 5000 25000 100000
"""

from __future__ import annotations

import argparse
import asyncio
import io
import random
import time

from rank_bm25 import BM25Okapi
from sqlalchemy import text

from app.db.session import async_session

TABLE = "chunks_text_scale"
TOP_K = 20

QUERIES = [
    "Why does the synchronous nature of Llama 3 16K-GPU training make it less fault-tolerant?",
    "What is the range of per-GPU model state size saved during checkpointing?",
    "What effect does reducing the attention key size have on model quality?",
    "How does the model handle code-switched speech?",
    "What percentage of prompt injection attacks were successful?",
]


async def real_contents(session) -> list[str]:
    rows = (
        await session.execute(text("SELECT content FROM chunks WHERE content IS NOT NULL"))
    ).fetchall()
    if not rows:
        raise SystemExit("no chunks — ingest documents first")
    return [r[0] for r in rows]


async def build_table(session, corpus: list[str], target: int, rng: random.Random) -> list[str]:
    """Fill a scratch table to `target` rows by sampling real chunk text."""
    await session.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
    await session.execute(
        text(f"""
            CREATE TABLE {TABLE} (
                id bigserial PRIMARY KEY,
                content text NOT NULL,
                content_tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
            )
        """)
    )
    await session.commit()

    rows = [corpus[rng.randrange(len(corpus))] for _ in range(target)]

    raw = await session.connection()
    driver = await raw.get_raw_connection()
    conn = driver.driver_connection

    buf = io.BytesIO()
    for c in rows:
        # COPY text format: escape backslash, tab and newline.
        safe = c.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ").replace("\r", " ")
        buf.write((safe + "\n").encode())
    buf.seek(0)
    await conn.copy_to_table(TABLE, source=buf, columns=["content"])

    await session.execute(text(f"CREATE INDEX ON {TABLE} USING gin (content_tsv)"))
    await session.execute(text(f"ANALYZE {TABLE}"))
    await session.commit()
    return rows


async def time_fts(session) -> float:
    sql = text(f"""
        WITH q AS (
            SELECT to_tsquery(
                'english',
                NULLIF(replace(plainto_tsquery('english', :query)::text, '&', '|'), '')
            ) AS tsq
        )
        SELECT c.id, ts_rank_cd(c.content_tsv, q.tsq) AS score
        FROM {TABLE} c CROSS JOIN q
        WHERE q.tsq IS NOT NULL AND c.content_tsv @@ q.tsq
        ORDER BY score DESC
        LIMIT {TOP_K}
    """)
    await session.execute(sql, {"query": QUERIES[0]})  # warm
    lat = []
    for q in QUERIES * 3:
        t0 = time.perf_counter()
        await session.execute(sql, {"query": q})
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return lat[len(lat) // 2]


def time_bm25(rows: list[str]) -> tuple[float, float]:
    """Return (build_s, query_p50_ms). Build cost is paid per worker process."""
    t0 = time.perf_counter()
    index = BM25Okapi([r.lower().split() for r in rows])
    build_s = time.perf_counter() - t0

    index.get_scores(QUERIES[0].lower().split())  # warm
    lat = []
    for q in QUERIES * 3:
        t0 = time.perf_counter()
        index.get_scores(q.lower().split())
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    return build_s, lat[len(lat) // 2]


async def run(sizes: list[int]) -> None:
    rng = random.Random(42)
    report = []

    async with async_session() as session:
        corpus = await real_contents(session)
        print(f"seed: {len(corpus)} real chunks\n")

        for size in sizes:
            print(f"=== {size:,} rows ===")
            rows = await build_table(session, corpus, size, rng)

            fts_p50 = await time_fts(session)
            bm_build, bm_p50 = time_bm25(rows)

            print(f"  postgres_fts  p50={fts_p50:8.1f} ms")
            print(f"  bm25          p50={bm_p50:8.1f} ms   (index build {bm_build:.1f}s)")
            winner = "fts" if fts_p50 < bm_p50 else "bm25"
            print(f"  faster: {winner}\n")
            report.append((size, fts_p50, bm_p50, bm_build, winner))

        await session.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        await session.commit()

    print("| rows | postgres_fts p50 | bm25 p50 | bm25 build | faster |")
    print("|---|---|---|---|---|")
    for size, f, b, bb, w in report:
        print(f"| {size:,} | {f:.1f} ms | {b:.1f} ms | {bb:.1f}s | {w} |")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sparse backend scaling comparison")
    ap.add_argument("--sizes", type=int, nargs="+", default=[5000, 25000, 100000])
    args = ap.parse_args()
    asyncio.run(run(args.sizes))


if __name__ == "__main__":
    main()
