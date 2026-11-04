"""
RAGScope — Golden-set builder with fairness checks

An LLM drafts the questions; code decides which ones are fair to score. A
question is kept only if:
  - each required fact appears in its gold chunk,
  - no other chunk states that fact (near-duplicates of the gold chunk, which
    the retrieval metrics already count as relevant, are allowed), so a
    retriever that finds the fact elsewhere is never marked wrong,
  - the question does not contain its own answer,
  - for two-hop questions, no single chunk states both facts.

    python -m app.scripts.build_golden single --n 100 --out /eval-datasets/golden-v3
    python -m app.scripts.build_golden multi --source /eval-datasets/golden-v3-test.jsonl \\
        --n 48 --out /eval-datasets/multihop-v2-test

    python -m app.scripts.build_golden check /eval-datasets/golden-v3-*.jsonl

`check` re-runs the checks on finished files, so hand edits get the same
scrutiny. `single` writes <out>-dev.jsonl and <out>-test.jsonl (--dev-fraction per
document); `multi` writes <out>.jsonl. Drafting uses settings.judge_model
(a "gemini-*" model needs GEMINI_API_KEY, anything else OPENROUTER_API_KEY).
Several passages go into each request, because free tiers cap requests, not
tokens. Whatever was accepted is written even if a run stops partway.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

import httpx
from sqlalchemy import select

from app.config import settings
from app.db.session import async_session
from app.eval.answer_judge import llm_chat, parse_items
from app.eval.metrics.generation_metrics import fact_present
from app.models import Chunk, Document

DOCUMENTS = {
    "llama3_paper.pdf": "the Llama 3 paper (Meta, 2024)",
    "attention_is_all_you_need.pdf": "the paper 'Attention Is All You Need' (the original Transformer)",
    "ipcc_ar6_syr_longer_report.pdf": "the IPCC AR6 Synthesis Report",
    "apple_10k_2023.pdf": "Apple's 2023 Form 10-K annual report",
}
PASSAGES_PER_CALL = 8
PAIRS_PER_CALL = 6

SINGLE_PROMPT = """You are building a test set for a question-answering system over four documents: the Llama 3 paper, "Attention Is All You Need", the IPCC AR6 Synthesis Report and Apple's 2023 Form 10-K.

For EACH numbered passage below, write ONE question that the passage answers with a specific fact, for a reader who has not seen the passage, or skip the passage.
- The question must stand alone: name its subject ("In Llama 3 pre-training, ...", "According to Apple's 2023 10-K, ...").
- The answer must be a specific stated fact: a number with its unit, a name, a method or a short phrase. Not an opinion or a summary.
- Do not put the answer in the question.
- "facts": the 1 or 2 key facts a correct answer must state. For each, list 1-3 short strings (1-4 words) copied exactly from the passage, plus common rewordings of a number (e.g. ["0.4 seconds", "0.4 s"]). Prefer a number together with its unit, or a distinctive name, over a bare number or a common word, so the fact belongs to this passage only.
- Ask about a detail specific to the passage, not a headline fact the document repeats in many places (for Llama 3: 405B parameters, 15.6T tokens, 128K context; for Apple: total net sales). Questions whose answer appears elsewhere in the document are thrown away.
- Skip a passage that has no self-contained fact worth asking about (a heading, a citation list, a table fragment, a cut-off sentence).

{passages}

Reply with JSON only:
{{"items": [{{"passage": 1, "question": "...", "answer": "<one sentence>", "facts": [["...", "..."]]}}, {{"passage": 2, "skip": "<why>"}}]}}"""

PASSAGE = "### Passage {n} (from {document})\n{text}"

MULTI_PROMPT = """You are building two-hop questions for a question-answering system.

For EACH numbered pair of facts below, write ONE natural question that needs BOTH facts to be answered completely, in the style given for that pair, or skip the pair.
- The question must stand alone and require stating both facts (for a comparison, ask for both values and which is larger).
- Do not put either answer in the question.
- Skip a pair whose facts cannot be combined into a sensible question in that style.

{pairs}

Reply with JSON only:
{{"items": [{{"pair": 1, "question": "...", "answer": "<one or two sentences stating both facts>"}}, {{"pair": 2, "skip": "<why>"}}]}}"""

PAIR = """### Pair {n} (style: {style})
Fact A, from {doc_a}: Q: {q_a} A: {a_a}
Fact B, from {doc_b}: Q: {q_b} A: {a_b}"""

STYLES = {
    "comparison": "a comparison of the two values",
    "conjunction": "one question with two related parts",
    "bridge": "refer to one fact's subject only through a detail from the other, so the reader must resolve it",
}

_JUNK = re.compile(
    r"et al\.|arXiv|doi\.org|https?://|\[\d+\]\s*[A-Z][a-z]+, [A-Z]\.", re.IGNORECASE
)


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def fact_holders(fact: list[str], chunks: list[dict]) -> list[str]:
    """Ids of chunks that state a fact (any of its wordings)."""
    return [c["id"] for c in chunks if fact_present(c["content"], fact)]


def check_single(item: dict, gold: dict, chunks: list[dict]) -> str | None:
    """Why a drafted single-hop question is unfair, or None if it can be kept.

    A fact's first wording must be copied from the passage, and that wording
    must not appear in any other chunk. Later wordings ("2.45%" for "2.450%")
    only widen what counts as a correct answer, so they are not checked for
    uniqueness, but none of them may appear in the question.
    """
    question, facts = str(item.get("question", "")).strip(), item.get("facts") or []
    if not question or not item.get("answer") or not 1 <= len(facts) <= 2:
        return "malformed"
    near_dupes = {c["id"] for c in chunks if _jaccard(c["content"], gold["content"]) >= 0.5}
    for fact in facts:
        if not isinstance(fact, list) or not all(
            isinstance(w, str) and len(w.strip()) >= 2 for w in fact
        ):
            return "malformed fact"
        if not fact_present(gold["content"], fact[:1]):
            return "fact not in its passage"
        if set(fact_holders(fact[:1], chunks)) - near_dupes:
            return "fact also stated elsewhere"
        if fact_present(question, fact):
            return "question contains its answer"
    return None


def check_multi(question: str, facts_a: list, facts_b: list, chunks: list[dict]) -> str | None:
    if not question:
        return "malformed"
    for fact in facts_a + facts_b:
        if fact_present(question, fact):
            return "question contains an answer"
    both = set.intersection(*(set(fact_holders(f[:1], chunks)) for f in facts_a + facts_b))
    return "one chunk states both facts" if both else None


def check_record(record: dict, chunks: list[dict]) -> str | None:
    """Re-run the fairness checks on a finished golden-set row (after hand edits)."""
    by_text = {c["content"]: c for c in chunks}
    golds = [by_text.get(g) for g in record.get("gold_contexts", [])]
    if not golds or None in golds:
        return "gold context not in the corpus"
    facts = record.get("answer_facts") or []
    item = {"question": record.get("question", ""), "answer": record.get("gold_answer", "")}
    if len(golds) == 1:
        return check_single({**item, "facts": facts}, golds[0], chunks)
    parts: list[list] = [[] for _ in golds]
    for fact in facts:
        owners = [i for i, g in enumerate(golds) if fact_present(g["content"], fact[:1])]
        if not owners:
            return f"fact {fact[0]!r} in none of its passages"
        parts[owners[0]].append(fact)
    for gold, part in zip(golds, parts, strict=True):
        if not part:
            return "a passage contributes no fact"
        problem = check_single({**item, "facts": part}, gold, chunks)
        if problem:
            return problem
    return check_multi(item["question"], parts[0], parts[1], chunks)


async def check_files(paths: list[str]) -> None:
    chunks = await _load_chunks()
    bad = 0
    for path in paths:
        rows = [
            json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
        ]
        for i, row in enumerate(rows):
            problem = check_record(row, chunks)
            if problem:
                bad += 1
                print(f"{Path(path).name}:{i}: {problem} | {row['question'][:90]}", flush=True)
        print(f"{Path(path).name}: {len(rows)} rows checked", flush=True)
    if bad:
        raise SystemExit(f"{bad} rows fail the checks")


def _items(reply: str, key: str) -> dict[int, dict]:
    """The reply's items keyed by their passage/pair number; {} if unusable."""
    try:
        items = parse_items(reply, "items")
    except ValueError:
        return {}
    keyed = {}
    for item in items:
        try:
            keyed[int(item.get(key, 0))] = item
        except (TypeError, ValueError):
            continue
    return keyed


def _verdict(items: dict[int, dict], n: int, reasons: Counter) -> dict | None:
    """Item n, or None after recording why there is nothing to check."""
    if n not in items:
        reasons["missing from reply"] += 1
        return None
    if "skip" in items[n]:
        reasons["skipped by writer"] += 1
        return None
    return items[n]


async def _load_chunks() -> list[dict]:
    async with async_session() as db:
        rows = (
            await db.execute(
                select(Chunk.id, Chunk.content, Chunk.element_type, Document.filename).join(
                    Document, Chunk.document_id == Document.id
                )
            )
        ).fetchall()
    return [
        {"id": str(r.id), "content": r.content, "type": r.element_type, "doc": r.filename}
        for r in rows
    ]


async def build_single(args, chunks: list[dict], llm) -> None:
    counts = Counter(c["content"] for c in chunks)
    rng = random.Random(args.seed)
    per_doc = args.n // len(DOCUMENTS)
    kept: list[dict] = []
    try:
        for doc, label in DOCUMENTS.items():
            pool = [
                c
                for c in chunks
                if c["doc"] == doc
                and c["type"] == "paragraph"
                and 150 <= len(c["content"]) <= 511
                and counts[c["content"]] == 1
                and not _JUNK.search(c["content"])
            ]
            rng.shuffle(pool)
            accepted: list[dict] = []
            reasons: Counter = Counter()
            for start in range(0, len(pool), PASSAGES_PER_CALL):
                if len(accepted) >= per_doc:
                    break
                batch = pool[start : start + PASSAGES_PER_CALL]
                prompt = SINGLE_PROMPT.format(
                    passages="\n\n".join(
                        PASSAGE.format(n=i, document=label, text=c["content"])
                        for i, c in enumerate(batch, 1)
                    )
                )
                items = _items(await llm(prompt), "passage")
                for i, chunk in enumerate(batch, 1):
                    item = _verdict(items, i, reasons)
                    if item is None:
                        continue
                    problem = check_single(item, chunk, chunks)
                    if problem:
                        reasons[problem] += 1
                        continue
                    accepted.append(
                        {
                            "question": str(item["question"]).strip(),
                            "gold_answer": str(item["answer"]).strip(),
                            "gold_contexts": [chunk["content"]],
                            "answer_facts": item["facts"],
                            "question_type": "factual",
                            "metadata": {
                                "source_chunk_ids": [chunk["id"]],
                                "source_documents": [doc],
                                "written_by": args.model,
                                "built": date.today().isoformat(),
                            },
                        }
                    )
                    if len(accepted) >= per_doc:
                        break
            print(f"{doc}: kept {len(accepted)}; {dict(reasons)}", flush=True)
            kept += accepted
    finally:
        rng.shuffle(kept)
        dev, test = [], []
        for doc in DOCUMENTS:
            mine = [k for k in kept if k["metadata"]["source_documents"] == [doc]]
            cut = round(len(mine) * args.dev_fraction)
            dev += mine[:cut]
            test += mine[cut:]
        _write(f"{args.out}-dev.jsonl", dev)
        _write(f"{args.out}-test.jsonl", test)


async def build_multi(args, chunks: list[dict], llm) -> None:
    source = [
        json.loads(line)
        for line in Path(args.source).read_text(encoding="utf-8").splitlines()
        if line
    ]
    rng = random.Random(args.seed)
    by_doc: dict[str, list[dict]] = {}
    for s in source:
        by_doc.setdefault(s["metadata"]["source_documents"][0], []).append(s)

    # Half within one document, half across two documents.
    pairs: list[tuple[dict, dict, str]] = []
    seen: set[tuple[str, str]] = set()
    styles = list(STYLES)
    for i in range(args.n * 4):
        if i % 2 == 0:
            doc = rng.choice([d for d in by_doc if len(by_doc[d]) >= 2])
            a, b = rng.sample(by_doc[doc], 2)
        else:
            da, db = rng.sample([d for d in by_doc if by_doc[d]], 2)
            a, b = rng.choice(by_doc[da]), rng.choice(by_doc[db])
        key = tuple(sorted((a["question"], b["question"])))
        if key not in seen and a["gold_contexts"] != b["gold_contexts"]:
            seen.add(key)
            pairs.append((a, b, styles[i % len(styles)]))

    kept: list[dict] = []
    reasons: Counter = Counter()
    try:
        for start in range(0, len(pairs), PAIRS_PER_CALL):
            if len(kept) >= args.n:
                break
            batch = pairs[start : start + PAIRS_PER_CALL]
            prompt = MULTI_PROMPT.format(
                pairs="\n\n".join(
                    PAIR.format(
                        n=i,
                        style=STYLES[style],
                        doc_a=DOCUMENTS[a["metadata"]["source_documents"][0]],
                        q_a=a["question"],
                        a_a=a["gold_answer"],
                        doc_b=DOCUMENTS[b["metadata"]["source_documents"][0]],
                        q_b=b["question"],
                        a_b=b["gold_answer"],
                    )
                    for i, (a, b, style) in enumerate(batch, 1)
                )
            )
            items = _items(await llm(prompt), "pair")
            for i, (a, b, style) in enumerate(batch, 1):
                item = _verdict(items, i, reasons)
                if item is None:
                    continue
                question = str(item.get("question", "")).strip()
                problem = check_multi(question, a["answer_facts"], b["answer_facts"], chunks)
                if problem:
                    reasons[problem] += 1
                    continue
                kept.append(
                    {
                        "question": question,
                        "gold_answer": str(item.get("answer", "")).strip(),
                        "gold_contexts": a["gold_contexts"] + b["gold_contexts"],
                        "answer_facts": a["answer_facts"] + b["answer_facts"],
                        "question_type": "multi-hop",
                        "metadata": {
                            "source_chunk_ids": a["metadata"]["source_chunk_ids"]
                            + b["metadata"]["source_chunk_ids"],
                            "source_documents": a["metadata"]["source_documents"]
                            + b["metadata"]["source_documents"],
                            "hops": 2,
                            "kind": style,
                            "written_by": args.model,
                            "built": date.today().isoformat(),
                        },
                    }
                )
                if len(kept) >= args.n:
                    break
    finally:
        print(f"multi: kept {len(kept)}; {dict(reasons)}", flush=True)
        _write(f"{args.out}.jsonl", kept)


def _write(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} to {path}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kind", choices=["single", "multi", "check"])
    parser.add_argument("files", nargs="*", help="golden-set JSONL files to re-check (check)")
    parser.add_argument("--n", type=int)
    parser.add_argument("--out", help="output path without the -dev/-test.jsonl suffix")
    parser.add_argument("--source", help="single-hop JSONL to pair facts from (multi)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--model", default=settings.judge_model)
    args = parser.parse_args()
    if args.kind == "check":
        await check_files(args.files)
        return
    if not args.n or not args.out:
        parser.error("--n and --out are required to build a set")

    chunks = await _load_chunks()
    async with httpx.AsyncClient(timeout=120.0) as client:

        async def llm(prompt: str) -> str:
            return await llm_chat(client, prompt, args.model, 2048)

        if args.kind == "single":
            await build_single(args, chunks, llm)
        else:
            await build_multi(args, chunks, llm)


if __name__ == "__main__":
    asyncio.run(main())
