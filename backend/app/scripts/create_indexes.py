"""
RAGScope — Index creation for existing databases

Creates every declared index idempotently, reporting what was already present.

Two reasons this exists rather than relying on create_all():

  - pgvector's HNSW options (USING hnsw, vector_cosine_ops, m /
    ef_construction) do not round-trip reliably through create_all() on a
    table that already exists, so the ANN index needs explicit DDL.
  - Building an HNSW index over a large table is slow (measured: 350s for
    100k x 768-dim vectors) and worth running deliberately, with the option
    of CONCURRENTLY, rather than silently during application startup.

Safe to re-run: existing indexes are skipped, not rebuilt.

Usage:
    python -m app.scripts.create_indexes [--concurrently]
"""

from __future__ import annotations

import argparse
import asyncio
import time

from sqlalchemy import text

from app.db.session import async_session, engine

# (name, DDL target). Kept as raw DDL because pgvector's HNSW options do not
# round-trip cleanly through create_all on an existing table.
INDEXES: list[tuple[str, str]] = [
    ("ix_chunks_document_id", "chunks (document_id)"),
    ("ix_chunks_parent_id", "chunks (parent_id)"),
    ("ix_chunks_element_type", "chunks (element_type)"),
    ("ix_chunks_created_at", "chunks (created_at)"),
    (
        "ix_chunks_embedding_hnsw",
        "chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
    ),
    ("ix_chunks_content_tsv", "chunks USING gin (content_tsv)"),
]

# Generated columns that create_all() will not add to an existing table.
# Must be applied before the indexes that depend on them.
COLUMNS: list[str] = [
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector "
    "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED",
]


async def existing_indexes(session, table: str) -> set[str]:
    rows = (
        await session.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = :t"),
            {"t": table},
        )
    ).fetchall()
    return {r[0] for r in rows}


async def main_async(concurrently: bool) -> None:
    async with async_session() as session:
        present = await existing_indexes(session, "chunks")
        print(f"existing indexes on chunks: {sorted(present) or 'none'}")

    # Columns first — the GIN index depends on content_tsv existing.
    for ddl in COLUMNS:
        print(f"  column {ddl.split()[5]} ...", end="", flush=True)
        t0 = time.perf_counter()
        async with engine.begin() as conn:
            await conn.execute(text(ddl))
        print(f" done in {time.perf_counter() - t0:.1f}s")

    for name, body in INDEXES:
        if name in present:
            print(f"  skip   {name} (already present)")
            continue

        # CREATE INDEX CONCURRENTLY cannot run inside a transaction block, so
        # it needs an autocommit connection.
        conc = "CONCURRENTLY " if concurrently else ""
        ddl = f"CREATE INDEX {conc}IF NOT EXISTS {name} ON {body}"
        print(f"  create {name} ...", end="", flush=True)
        t0 = time.perf_counter()
        async with engine.connect() as conn:
            await conn.execution_options(isolation_level="AUTOCOMMIT")
            await conn.execute(text(ddl))
        print(f" done in {time.perf_counter() - t0:.1f}s")

    async with async_session() as session:
        final = await existing_indexes(session, "chunks")
        print(f"\nindexes on chunks now: {sorted(final)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create declared indexes on an existing DB")
    ap.add_argument(
        "--concurrently",
        action="store_true",
        help="Build without locking writes (slower; requires autocommit)",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args.concurrently))


if __name__ == "__main__":
    main()
