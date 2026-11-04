"""
RAGScope — Entity-graph retrieval

A GraphRAG-style retriever that needs no LLM at index time. The corpus becomes a
bipartite graph: chunks on one side, named entities on the other, an edge
wherever a chunk mentions an entity. Entities are what the text itself marks
as names: acronyms (RLHF, IPCC), letter-digit tokens (H100, 405B) and words
capitalised mid-sentence somewhere (Cupertino, Teton). The mid-sentence rule
keeps sentence-initial words like "However" out. Entities in more than 5% of
chunks are dropped: a hub that touches everything carries no signal and drowns
the walk.

At query time the question's entities seed a personalized PageRank. Mass flows
from those entities to the chunks that mention them, on to the other entities
those chunks mention, and back, restarting at the seeds with probability
RESTART. A chunk can therefore score without sharing a word with the question
if it shares rare entities with chunks that do: the bridge a multi-hop
question needs, which dense and BM25 retrieval cannot see. Edges are weighted
by entity IDF so a rare shared entity counts for more than a common one.

The graph is built in-process from the chunks table and cached until the chunk
count changes or ingestion invalidates it, the same contract as the BM25 index.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]")
_SENTENCE_END = frozenset(".!?:;•\n")
_STOPWORDS = frozenset(
    "the a an and or of in on for to with by from as at is are be this that these those "
    "it its we our they their table figure fig section appendix see".split()
)
MAX_ENTITY_DF = 0.05


def _share(weight: np.ndarray, total: np.ndarray) -> np.ndarray:
    """weight / total, with 0 where total is 0. In a corpus of one or two chunks an
    entity can sit in every chunk: its idf is 0, and 0/0 put NaN into the walk."""
    return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)


RESTART = 0.5
ITERATIONS = 20


def candidate_entities(text: str) -> dict[str, bool]:
    """Entity key -> whether this text proves the key is a name.

    Acronyms and letter-digit tokens are names wherever they appear. A plain
    capitalised word only proves it when capitalised mid-sentence; at the start
    of a sentence it may just be "However". Keys are lowercased so a question
    typed in lowercase still matches.
    """
    found: dict[str, bool] = {}
    for match in _TOKEN.finditer(text):
        token = match.group()
        key = token.lower()
        if len(key) < 2 or key in _STOPWORDS:
            continue
        has_digit = any(c.isdigit() for c in token)
        has_alpha = any(c.isalpha() for c in token)
        if not has_alpha:
            continue
        if has_digit or any(c.isupper() for c in token[1:]):
            found[key] = True
        elif token[0].isupper():
            before = text[: match.start()].rstrip()[-1:]
            found[key] = found.get(key, False) or (bool(before) and before not in _SENTENCE_END)
    return found


@dataclass
class EntityGraph:
    chunks: list[dict]
    vocab: dict[str, int]
    idf: np.ndarray
    edge_chunk: np.ndarray
    edge_entity: np.ndarray
    to_chunk: np.ndarray  # P(entity -> chunk) for each edge
    to_entity: np.ndarray  # P(chunk -> entity) for each edge

    @classmethod
    def build(cls, chunks: list[dict]) -> EntityGraph:
        per_chunk = [candidate_entities(c["content"]) for c in chunks]
        proven = {k for found in per_chunk for k, ok in found.items() if ok}
        df = Counter(k for found in per_chunk for k in found if k in proven)
        cap = max(2, int(MAX_ENTITY_DF * len(chunks)))
        vocab = {k: i for i, k in enumerate(sorted(k for k, n in df.items() if n <= cap))}
        idf = np.array([math.log(len(chunks) / df[k]) for k in vocab], dtype=np.float64)

        pairs = [(c, vocab[k]) for c, found in enumerate(per_chunk) for k in found if k in vocab]
        edge_chunk = np.array([c for c, _ in pairs], dtype=np.int64)
        edge_entity = np.array([e for _, e in pairs], dtype=np.int64)
        weight = idf[edge_entity] if pairs else np.zeros(0)
        chunk_total = np.bincount(edge_chunk, weights=weight, minlength=len(chunks))
        entity_total = np.bincount(edge_entity, weights=weight, minlength=len(vocab))
        return cls(
            chunks=chunks,
            vocab=vocab,
            idf=idf,
            edge_chunk=edge_chunk,
            edge_entity=edge_entity,
            to_chunk=_share(weight, entity_total[edge_entity]),
            to_entity=_share(weight, chunk_total[edge_chunk]),
        )

    def query_entities(self, question: str) -> list[int]:
        keys = {m.group().lower() for m in _TOKEN.finditer(question)}
        return sorted(self.vocab[k] for k in keys if k in self.vocab)

    def rank(self, seeds: list[int], top_k: int) -> list[dict]:
        """Personalized PageRank from the seed entities; chunks ranked by mass."""
        if not seeds:
            return []
        restart = np.zeros(len(self.vocab))
        restart[seeds] = self.idf[seeds]
        restart /= restart.sum()

        n_chunks = len(self.chunks)
        entity_mass = restart
        for _ in range(ITERATIONS):
            chunk_mass = np.bincount(
                self.edge_chunk,
                weights=entity_mass[self.edge_entity] * self.to_chunk,
                minlength=n_chunks,
            )
            flowed = np.bincount(
                self.edge_entity,
                weights=chunk_mass[self.edge_chunk] * self.to_entity,
                minlength=len(self.vocab),
            )
            entity_mass = RESTART * restart + (1 - RESTART) * flowed

        scores = np.bincount(
            self.edge_chunk,
            weights=entity_mass[self.edge_entity] * self.to_chunk,
            minlength=n_chunks,
        )
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            {**self.chunks[i], "graph_score": round(float(scores[i]), 6)}
            for i in order
            if scores[i] > 0
        ]


# ── Module-level graph cache ───────────────────
_cached_graph: EntityGraph | None = None
_cached_chunk_count: int = 0


def invalidate_graph_cache() -> None:
    """Call after ingestion or deletion so the next graph query rebuilds."""
    global _cached_graph, _cached_chunk_count
    _cached_graph = None
    _cached_chunk_count = 0


class GraphRetriever:
    """Ranks chunks by personalized PageRank from the question's entities."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _graph(self) -> EntityGraph | None:
        global _cached_graph, _cached_chunk_count

        count = (await self.db.execute(select(func.count(Chunk.id)))).scalar() or 0
        if _cached_graph is not None and _cached_chunk_count == count:
            return _cached_graph

        rows = (
            await self.db.execute(
                select(
                    Chunk.id,
                    Chunk.content,
                    Chunk.element_type,
                    Chunk.chunk_index,
                    Chunk.token_count,
                    Chunk.metadata_,
                    Document.filename,
                ).join(Document, Chunk.document_id == Document.id)
            )
        ).fetchall()
        if not rows:
            return None

        graph = EntityGraph.build(
            [
                {
                    "chunk_id": str(r.id),
                    "content": r.content,
                    "document_name": r.filename,
                    "element_type": r.element_type,
                    "chunk_index": r.chunk_index,
                    "token_count": r.token_count,
                    "metadata": r.metadata_ or {},
                }
                for r in rows
            ]
        )
        _cached_graph, _cached_chunk_count = graph, count
        logger.info(
            "Entity graph built and cached",
            chunks=len(graph.chunks),
            entities=len(graph.vocab),
            edges=len(graph.edge_chunk),
        )
        return graph

    async def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        with create_span(
            tracer,
            "graph_retrieve",
            "RETRIEVER",
            {"retriever.type": "entity_graph", "retriever.top_k": top_k},
        ) as span:
            graph = await self._graph()
            if graph is None:
                return []
            seeds = graph.query_entities(query)
            span.set_attribute("graph.query_entities", len(seeds))
            if not seeds:
                return []
            return graph.rank(seeds, top_k)
