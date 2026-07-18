"""
RAGScope — Retrieval Metrics

Measures how well the retriever finds the right documents.
All metrics computed against gold reference contexts.

Metrics:
- Context Precision: fraction of retrieved chunks that are relevant
- Context Recall: fraction of relevant chunks that were retrieved
- Hit Rate: did at least one relevant chunk appear in top-k?
- MRR: reciprocal rank of the first relevant chunk
- NDCG@k: normalized discounted cumulative gain
- Precision@k, Recall@k
"""

from __future__ import annotations

import math

import structlog

from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("eval")


def _compute_relevance_labels(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
    similarity_threshold: float = 0.5,
) -> list[int]:
    """
    Compute binary relevance labels for retrieved chunks.

    A chunk is relevant if it has significant token overlap with any gold context.
    Uses Jaccard similarity on word-level tokens.
    """
    labels = []
    gold_token_sets = [set(ctx.lower().split()) for ctx in gold_contexts]

    for chunk in retrieved_chunks:
        chunk_tokens = set(chunk.get("content", "").lower().split())
        is_relevant = 0

        for gold_tokens in gold_token_sets:
            if not gold_tokens or not chunk_tokens:
                continue
            intersection = chunk_tokens & gold_tokens
            union = chunk_tokens | gold_tokens
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= similarity_threshold:
                is_relevant = 1
                break

        labels.append(is_relevant)

    return labels


def context_precision(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
) -> float:
    """
    Context Precision: what fraction of retrieved chunks are relevant?

    precision = |relevant ∩ retrieved| / |retrieved|
    """
    if not retrieved_chunks:
        return 0.0

    labels = _compute_relevance_labels(retrieved_chunks, gold_contexts)
    relevant_count = sum(labels)
    return relevant_count / len(labels)


def context_recall(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
) -> float:
    """
    Context Recall: what fraction of gold contexts were found?

    recall = |gold contexts covered| / |total gold contexts|
    """
    if not gold_contexts:
        return 1.0  # No gold contexts = trivially complete
    if not retrieved_chunks:
        return 0.0

    retrieved_texts = " ".join(c.get("content", "") for c in retrieved_chunks).lower()
    covered = 0

    for gold in gold_contexts:
        gold_tokens = set(gold.lower().split())
        retrieved_tokens = set(retrieved_texts.split())
        if not gold_tokens:
            covered += 1
            continue
        overlap = gold_tokens & retrieved_tokens
        coverage = len(overlap) / len(gold_tokens) if gold_tokens else 0
        if coverage >= 0.5:
            covered += 1

    return covered / len(gold_contexts)


def hit_rate(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
) -> float:
    """
    Hit Rate: did at least one relevant chunk appear in top-k?

    Returns 1.0 if yes, 0.0 if no.
    """
    labels = _compute_relevance_labels(retrieved_chunks, gold_contexts)
    return 1.0 if any(label == 1 for label in labels) else 0.0


def mean_reciprocal_rank(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
) -> float:
    """
    MRR: 1 / (rank of first relevant chunk).

    Returns 0.0 if no relevant chunk found.
    """
    labels = _compute_relevance_labels(retrieved_chunks, gold_contexts)
    for rank, label in enumerate(labels, start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
    k: int = 10,
) -> float:
    """
    NDCG@k: normalized discounted cumulative gain.

    Measures ranking quality — rewards relevant docs at higher positions.
    """
    labels = _compute_relevance_labels(retrieved_chunks[:k], gold_contexts)

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(labels))

    # Ideal DCG (all relevant first)
    ideal_labels = sorted(labels, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_labels))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def precision_at_k(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
    k: int = 10,
) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    labels = _compute_relevance_labels(retrieved_chunks[:k], gold_contexts)
    return sum(labels) / k if k > 0 else 0.0


def recall_at_k(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
    k: int = 10,
) -> float:
    """Recall@k: fraction of relevant docs found in top-k."""
    if not gold_contexts:
        return 1.0
    labels = _compute_relevance_labels(retrieved_chunks[:k], gold_contexts)
    total_relevant = len(gold_contexts)
    return sum(labels) / total_relevant if total_relevant > 0 else 0.0


def compute_all_retrieval_metrics(
    retrieved_chunks: list[dict],
    gold_contexts: list[str],
    k: int = 10,
) -> dict[str, float]:
    """Compute all retrieval metrics in one pass."""
    with create_span(
        tracer,
        "compute_retrieval_metrics",
        "EVALUATOR",
        {"eval.metric_type": "retrieval", "eval.k": k},
    ):
        metrics = {
            "context_precision": round(context_precision(retrieved_chunks, gold_contexts), 4),
            "context_recall": round(context_recall(retrieved_chunks, gold_contexts), 4),
            "hit_rate": round(hit_rate(retrieved_chunks, gold_contexts), 4),
            "mrr": round(mean_reciprocal_rank(retrieved_chunks, gold_contexts), 4),
            f"ndcg@{k}": round(ndcg_at_k(retrieved_chunks, gold_contexts, k), 4),
            f"precision@{k}": round(precision_at_k(retrieved_chunks, gold_contexts, k), 4),
            f"recall@{k}": round(recall_at_k(retrieved_chunks, gold_contexts, k), 4),
        }

        logger.info("Retrieval metrics computed", **metrics)
        return metrics
