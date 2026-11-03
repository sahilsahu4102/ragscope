"""
RAGScope — Multi-hop retrieval

Single-pass retrieval embeds the whole question once. That works when one chunk
holds the answer. A question that needs two facts from two places ("how many
layers does the original Transformer have, and how many does Llama 3 405B
have?") makes one query vector straddle both, and the reranker then scores
every candidate against the combined question, where neither half wins.

This retriever gives each part of the question its own retrieval:

  1. decompose: one LLM call splits the question into sub-questions
  2. retrieve: each sub-question runs the normal hybrid + rerank pipeline
     against itself, concurrently
  3. follow up (agentic mode): the LLM sees the best evidence so far and either
     replies DONE or writes one search query for what is missing, up to
     max_followups times
  4. merge: the original question's ranking keeps its top PROTECTED chunks;
     each other hop's best chunk follows, in hop order; the rest of the
     original ranking fills the remaining slots.

The first version fused the hops with RRF and measured significantly worse
than plain retrieval (docs/eval-results.md, Phase 8). RRF rewards agreement, and
sub-questions about one topic agree on generic chunks: "Figure 1: The
Transformer - model architecture" sat mid-list for three sub-questions and
out-scored the one chunk that answered each of them. The merge takes a single
chunk per hop, so nothing accumulates.

query_transform="decompose" is this retriever with max_followups=0.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Protocol

import structlog

from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")

_FOLLOWUP_PROMPT = """You are searching a document collection to answer a question.

Question: {question}

Evidence found so far:
{evidence}

If the evidence answers every part of the question, reply with exactly: DONE
Otherwise reply with ONE short search query for the missing information and nothing else."""

# The follow-up prompt shows each hop's best chunks, trimmed, so a 3B model sees
# the evidence without a 4k-token prompt.
_EVIDENCE_PER_HOP = 2
_EVIDENCE_CHARS = 300

# The original question's best chunks are not displaced. Its MRR on the
# single-hop golden set is 0.89, so the answer is almost always in the top 2.
PROTECTED = 2


class _Transformer(Protocol):
    async def decompose(self, query: str) -> list[str]: ...

    async def _llm_call(self, prompt: str) -> str: ...


def _norm(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", query.lower()))


def _dedupe(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    kept = []
    for q in queries:
        key = _norm(q)
        if key and key not in seen:
            seen.add(key)
            kept.append(q)
    return kept


def parse_followup(text: str, asked: list[str]) -> str | None:
    """The follow-up query from the LLM's reply, or None when it is done.

    A repeat of a query already asked also counts as done: retrieving it again
    would return the same list and only add latency.
    """
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = re.sub(r"^(?:search\s+)?query\s*:\s*", "", line, flags=re.IGNORECASE)
    line = line.strip(" \"'`*")
    if not line or line.upper().startswith("DONE"):
        return None
    if _norm(line) in {_norm(q) for q in asked}:
        return None
    return line


def merge_hops(ranked: list[list[dict]], top_k: int) -> list[dict]:
    """Original ranking's top PROTECTED, then each later hop's best chunk, then the rest.

    `ranked[0]` is the original question's list. A hop whose best chunk is
    already placed adds nothing; it does not reach further down its list.
    """
    original, hops = ranked[0], ranked[1:]
    merged = original[:PROTECTED]
    placed = {c["chunk_id"] for c in merged}
    for hop in hops:
        if hop and hop[0]["chunk_id"] not in placed:
            merged.append(hop[0])
            placed.add(hop[0]["chunk_id"])
    merged += [c for c in original[PROTECTED:] if c["chunk_id"] not in placed]
    return merged[:top_k]


def _evidence(ranked_lists: list[list[dict]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for ranked in ranked_lists:
        for chunk in ranked[:_EVIDENCE_PER_HOP]:
            if chunk["chunk_id"] in seen:
                continue
            seen.add(chunk["chunk_id"])
            text = " ".join(chunk["content"].split())[:_EVIDENCE_CHARS]
            lines.append(f"[{len(lines) + 1}] {text}")
    return "\n".join(lines) or "(nothing found)"


class MultiHopRetriever:
    """Decompose, retrieve per hop, optionally follow up, then merge.

    `retrieve_one` runs single-query retrieval and must be safe to call
    concurrently; the pipeline passes one that opens its own DB session.
    """

    def __init__(
        self,
        retrieve_one: Callable[[str], Awaitable[list[dict]]],
        transformer: _Transformer,
        max_followups: int = 1,
    ):
        self.retrieve_one = retrieve_one
        self.transformer = transformer
        self.max_followups = max_followups

    async def retrieve(self, question: str, top_k: int) -> list[dict]:
        with create_span(
            tracer,
            "multihop_retrieval",
            "AGENT" if self.max_followups else "CHAIN",
            {"multihop.max_followups": self.max_followups},
        ) as span:
            sub_queries = await self.transformer.decompose(question)
            hops = _dedupe([question, *sub_queries])
            ranked = list(await asyncio.gather(*(self.retrieve_one(q) for q in hops)))

            followups = 0
            while followups < self.max_followups:
                reply = await self.transformer._llm_call(
                    _FOLLOWUP_PROMPT.format(question=question, evidence=_evidence(ranked))
                )
                query = parse_followup(reply, hops)
                if query is None:
                    break
                hops.append(query)
                ranked.append(await self.retrieve_one(query))
                followups += 1

            merged = merge_hops(ranked, top_k)

            span.set_attribute("multihop.hops", len(hops))
            span.set_attribute("multihop.followups", followups)
            logger.info(
                "Multi-hop retrieval complete",
                hops=len(hops),
                followups=followups,
                returned=len(merged),
            )
            return merged
