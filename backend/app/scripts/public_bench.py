"""
RAGScope — Public benchmarks: NanoBEIR and MultiHop-RAG

Runs the production retrieval pipeline on public data, so its numbers can sit
next to published ones instead of only next to RAGScope's own golden sets.

Each benchmark gets its own Postgres database, and the script refuses any
database whose name does not start with "bench_", so a benchmark corpus is
never mixed into the main one:

    docker compose exec postgres createdb -U ragscope bench_nanonfcorpus
    docker compose run --rm -e POSTGRES_DB=bench_nanonfcorpus backend \\
        python -m app.scripts.public_bench nanobeir NanoNFCorpus
    docker compose exec postgres createdb -U ragscope bench_multihoprag
    docker compose run --rm -e POSTGRES_DB=bench_multihoprag backend \\
        python -m app.scripts.public_bench multihop-rag --limit 250 --modes production agentic

The first run downloads the data (Hugging Face dataset viewer API, cached
under /eval-datasets/public/) and ingests it with the production embedder;
later runs reuse the database.

NanoBEIR (zeta-alpha-ai/Nano*, 50 queries each, binary qrels): NDCG@10 for
BM25, dense, hybrid (RRF), and hybrid + cross-encoder at production depth
(top_k=10, so 40 candidates) and at 100 candidates. `reference` runs the
published reranking protocol instead (BM25 top 100 plus missed positives, no
retrieval), to check the reranker against the published numbers.

MultiHop-RAG (yixuantt/MultiHopRAG, 609 news articles, 2,255 non-null
queries): articles are split into ~256-token sentence windows, as in the
benchmark's own setup, and retrieval is scored with the logic of its
retrieval_evaluate.py: a chunk is relevant when an evidence sentence, with
spaces and newlines removed, is a substring of it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import time
from pathlib import Path

import httpx
from sqlalchemy import func, select

from app.config import settings
from app.db.session import async_session, init_db
from app.eval.experiments import _bootstrap_significant
from app.ingestion.embedders.embedder import get_embedder
from app.models import Chunk, Document
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.rerankers import get_reranker
from app.retrieval.sparse import SparseRetriever
from app.scripts.answer_eval import _bootstrap

ROWS_API = "https://datasets-server.huggingface.co/rows"
CACHE = Path("/eval-datasets/public")
RESULTS = Path("/eval-datasets/results/public")

# name -> pipeline.retrieve keyword arguments (bm25 bypasses the pipeline)
MODES = {
    "bm25": None,
    "dense": {"top_k": 10, "use_hybrid": False, "use_reranker": False},
    "hybrid": {"top_k": 10, "use_hybrid": True, "use_reranker": False},
    "production": {"top_k": 10, "use_hybrid": True, "use_reranker": True},
    "rerank100": {"top_k": 25, "use_hybrid": True, "use_reranker": True},
    "agentic": {"top_k": 10, "use_hybrid": True, "use_reranker": True, "strategy": "agentic"},
}

WORDS_PER_CHUNK = 190  # ~256 tokens of English news text
OVERLAP_WORDS = 60


# ── Data ──────────────────────────────────────────


def fetch_rows(dataset: str, config: str, split: str = "train") -> list[dict]:
    """Every row of one config, via the dataset viewer API (no extra dependencies)."""
    name = config if split == "train" else f"{config}-{split}"
    path = CACHE / dataset.replace("/", "__") / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    with httpx.Client(timeout=60) as client:
        while True:
            params = {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": len(rows),
                "length": 100,
            }
            for attempt in range(5):
                response = client.get(ROWS_API, params=params)
                # The viewer API also returns transient 502s mid-download.
                if response.status_code != 429 and response.status_code < 500:
                    break
                time.sleep(2**attempt)
            response.raise_for_status()
            data = response.json()
            rows += [r["row"] for r in data["rows"]]
            if not data["rows"] or len(rows) >= data["num_rows_total"]:
                break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def sentence_windows(text: str) -> list[str]:
    """Pack whole sentences into ~WORDS_PER_CHUNK-word windows that overlap by
    ~OVERLAP_WORDS, so an evidence sentence always sits whole in some chunk."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    windows, start = [], 0
    while start < len(sentences):
        end, words = start, 0
        while end < len(sentences) and (
            words == 0 or words + len(sentences[end].split()) <= WORDS_PER_CHUNK
        ):
            words += len(sentences[end].split())
            end += 1
        windows.append(" ".join(sentences[start:end]))
        if end >= len(sentences):
            break
        back, kept = end, 0  # step back over the last ~OVERLAP_WORDS words
        while back - 1 > start and kept + len(sentences[back - 1].split()) <= OVERLAP_WORDS:
            back -= 1
            kept += len(sentences[back].split())
        start = back
    return windows


# ── Database ──────────────────────────────────────


def _guard() -> None:
    if not settings.postgres_db.startswith("bench_"):
        raise SystemExit(
            f"POSTGRES_DB is {settings.postgres_db!r}; benchmarks only run in a database "
            "named bench_*, so they cannot mix into the main corpus"
        )


async def ingest(name: str, docs: list[tuple[str, str]]) -> None:
    """Store (doc_id, text) pairs as one chunk each, embedded by the production embedder."""
    await init_db()
    async with async_session() as db:
        stored = (await db.execute(select(func.count(Chunk.id)))).scalar()
        if stored == len(docs):
            print("already ingested", flush=True)
            return
        if stored:
            raise SystemExit(f"{stored} of {len(docs)} chunks stored; drop the database and rerun")
        document = Document(filename=name, mime_type="text/plain", status="completed")
        db.add(document)
        await db.flush()
        embedder = get_embedder()
        for start in range(0, len(docs), 256):
            batch = docs[start : start + 256]
            vectors = await embedder.embed([text for _, text in batch])
            # The embedder returns zero vectors when Ollama fails; never store them.
            if not all(any(v) for v in vectors):
                raise SystemExit("embedding failed (zero vectors); drop the database and rerun")
            db.add_all(
                Chunk(
                    document_id=document.id,
                    content=text,
                    chunk_index=start + i,
                    embedding=vector,
                    embedding_model=embedder.model_name(),
                    element_type="paragraph",
                    metadata_={"doc_id": doc_id},
                )
                for i, ((doc_id, text), vector) in enumerate(zip(batch, vectors, strict=True))
            )
            await db.commit()
            print(f"ingested {min(start + 256, len(docs))}/{len(docs)}", flush=True)


async def _doc_ids() -> dict[str, str]:
    async with async_session() as db:
        rows = (await db.execute(select(Chunk.id, Chunk.metadata_))).fetchall()
    return {str(r.id): r.metadata_["doc_id"] for r in rows}


async def retrieve(mode: str, query: str) -> list[dict]:
    async with async_session() as db:
        if MODES[mode] is None:
            return await SparseRetriever(db).retrieve(query=query, top_k=10)
        return await RetrievalPipeline(db).retrieve(
            query=query, rrf_k=settings.rrf_k, **MODES[mode]
        )


# ── Metrics ───────────────────────────────────────


def ndcg_at_10(ranked: list[str], relevant: set[str]) -> float:
    """Binary-gain NDCG@10 over document ids, each document counted once."""
    seen: list[str] = []
    for doc in ranked:
        if doc not in seen:
            seen.append(doc)
    dcg = sum(1 / math.log2(rank + 1) for rank, doc in enumerate(seen[:10], 1) if doc in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
    return dcg / ideal if ideal else 0.0


def multihop_rag_metrics(retrieved: list[str], gold: list[str]) -> dict[str, float]:
    """Per-query Hits@10, Hits@4, MAP@10 and MRR@10, following MultiHop-RAG's
    retrieval_evaluate.py, plus full support (every evidence sentence in the top 10)."""
    gold = [g.replace(" ", "").replace("\n", "") for g in gold]
    retrieved = [r.replace(" ", "").replace("\n", "") for r in retrieved]
    found: list[str] = []
    first = None
    precision_sum = 0.0
    hits4 = False
    for rank, text in enumerate(retrieved[:10], 1):
        if any(g in text for g in gold):
            first = first or rank
            hits4 = hits4 or rank <= 4
            new = [g for g in gold if g in text and g not in found]
            found += new
            precision_sum += len(new) / rank
    return {
        "hits@10": float(first is not None),
        "hits@4": float(hits4),
        "map@10": precision_sum / min(len(gold), 10),
        "mrr@10": 1 / first if first else 0.0,
        "full_support@10": float(len(set(found)) == len(set(gold))),
    }


def _summary(values: list[float]) -> dict:
    low, high = _bootstrap(values, statistics.fmean)
    return {"mean": round(statistics.fmean(values), 4), "ci95": [round(low, 4), round(high, 4)]}


async def _run(modes: list[str], items: list[tuple[str, object]], score) -> dict:
    results = {}
    for mode in modes:
        rows, latencies = [], []
        for query, gold in items:
            start = time.perf_counter()
            chunks = await retrieve(mode, query)
            latencies.append((time.perf_counter() - start) * 1000)
            rows.append(score(chunks, gold))
            if len(rows) % 50 == 0:
                print(f"{mode}: {len(rows)}/{len(items)}", flush=True)
        results[mode] = _summarize(mode, rows, latencies)
    return results


def _summarize(mode: str, rows: list[dict], latencies: list[float]) -> dict:
    result = {
        "n": len(rows),
        "latency_ms_p50": round(statistics.median(latencies), 1),
        **{metric: _summary([r[metric] for r in rows]) for metric in rows[0]},
    }
    print(mode, json.dumps(result), flush=True)
    # Kept out of the printed summary; `compare` pairs these by position.
    result["per_query"] = {m: [round(r[m], 4) for r in rows] for m in rows[0]}
    return result


def compare(path_a: str, path_b: str) -> None:
    """Paired bootstrap between two result files, for every mode and metric they share."""
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    print("| mode | metric | A | B | delta | p |\n|---|---|---|---|---|---|")
    for mode in [m for m in a if m in b]:
        for metric, values_a in a[mode]["per_query"].items():
            values_b = b[mode]["per_query"][metric]
            _, p = _bootstrap_significant(values_a, values_b)
            mean_a, mean_b = statistics.fmean(values_a), statistics.fmean(values_b)
            print(
                f"| {mode} | {metric} | {mean_a:.4f} | {mean_b:.4f} | {mean_b - mean_a:+.4f} | {p} |"
            )


# ── Benchmarks ────────────────────────────────────


async def nanobeir(args) -> dict:
    repo = f"zeta-alpha-ai/{args.dataset}"
    corpus = fetch_rows(repo, "corpus")
    docs = [
        (r["_id"], f"{r.get('title') or ''}\n{r['text']}".strip())
        for r in corpus
        if (r.get("text") or "").strip()
    ]
    await ingest(args.dataset, docs)
    queries = {r["_id"]: r["text"] for r in fetch_rows(repo, "queries")}
    relevant: dict[str, set[str]] = {}
    for r in fetch_rows(repo, "qrels"):
        relevant.setdefault(r["query-id"], set()).add(r["corpus-id"])
    doc_of = await _doc_ids()
    items = [(queries[q], rel) for q, rel in relevant.items() if q in queries]
    modes = args.modes or ["bm25", "dense", "hybrid", "production", "rerank100"]

    def score(chunks: list[dict], rel: set[str]) -> dict:
        return {"ndcg@10": ndcg_at_10([doc_of[str(c["chunk_id"])] for c in chunks], rel)}

    return await _run(modes, items, score)


async def reference(args) -> dict:
    """The published protocol with RAGScope's reranker and no retrieval.

    sentence-transformers' CrossEncoderNanoBEIREvaluator reranks BM25's top 100
    (precomputed in sentence-transformers/NanoBEIR-en) plus every relevant
    document BM25 missed, on document text without titles. Matching its
    published numbers shows the reranker is wired as the reference assumes;
    the nanobeir modes show what RAGScope gets when it finds its own candidates.
    """
    repo, split = "sentence-transformers/NanoBEIR-en", args.dataset
    text = {r["_id"]: r["text"] for r in fetch_rows(repo, "corpus", split)}
    queries = {r["_id"]: r["text"] for r in fetch_rows(repo, "queries", split)}
    relevant: dict[str, list[str]] = {}
    for r in fetch_rows(repo, "qrels", split):
        relevant.setdefault(r["query-id"], []).append(r["corpus-id"])
    reranker = get_reranker(use_cross_encoder=True)
    rows, latencies = [], []
    for r in fetch_rows(repo, "bm25", split):
        positives = relevant[r["query-id"]]
        ids = positives + [i for i in r["corpus-ids"][:100] if i not in positives]
        start = time.perf_counter()
        ranked = await reranker.rerank(
            queries[r["query-id"]], [{"chunk_id": i, "content": text[i]} for i in ids], top_k=10
        )
        latencies.append((time.perf_counter() - start) * 1000)
        rows.append({"ndcg@10": ndcg_at_10([c["chunk_id"] for c in ranked], set(positives))})
    return {"reference": _summarize("reference", rows, latencies)}


async def multihop_rag(args) -> dict:
    repo = "yixuantt/MultiHopRAG"
    docs = []
    for i, article in enumerate(fetch_rows(repo, "corpus")):
        header = f"{article['title']} ({article['source']}, {str(article['published_at'])[:10]})"
        docs += [(f"a{i}", f"{header}\n{w}") for w in sentence_windows(article["body"] or "")]
    await ingest("MultiHopRAG", docs)
    queries = [q for q in fetch_rows(repo, "MultiHopRAG") if q["question_type"] != "null_query"]
    items = [(q["query"], [e["fact"] for e in q["evidence_list"]]) for q in queries][
        : args.limit or None
    ]
    modes = args.modes or ["dense", "production"]

    def score(chunks: list[dict], gold: list[str]) -> dict:
        return multihop_rag_metrics([c["content"] for c in chunks], gold)

    return await _run(modes, items, score)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="bench", required=True)
    nb = sub.add_parser("nanobeir")
    nb.add_argument("dataset", help="e.g. NanoNFCorpus, NanoMSMARCO, NanoNQ")
    ref = sub.add_parser("reference", help="the published NanoBEIR reranking protocol")
    ref.add_argument("dataset", help="e.g. NanoNFCorpus, NanoMSMARCO, NanoNQ")
    mh = sub.add_parser("multihop-rag")
    mh.add_argument("--limit", type=int, default=0, help="first N non-null queries")
    for p in (nb, mh):
        p.add_argument("--modes", nargs="*", choices=list(MODES))
    for p in (nb, ref, mh):
        p.add_argument("--tag", default="", help="added to the result file name, e.g. base")
    cmp = sub.add_parser("compare", help="paired bootstrap between two result files")
    cmp.add_argument("a")
    cmp.add_argument("b")
    args = parser.parse_args()
    if args.bench == "compare":
        compare(args.a, args.b)
        return
    _guard()

    run = {"nanobeir": nanobeir, "reference": reference, "multihop-rag": multihop_rag}
    results = await run[args.bench](args)
    name = f"MultiHopRAG-{args.limit or 'all'}" if args.bench == "multihop-rag" else args.dataset
    name += f"-{args.tag}" if args.tag else ""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}-{'-'.join(results)}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
