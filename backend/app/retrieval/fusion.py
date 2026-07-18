"""
RAGScope — Reciprocal Rank Fusion (RRF)

Fuses ranked lists from dense and sparse retrievers using RRF:
    score(d) = Σ 1/(k + rank(d))

RRF operates on ranks not raw scores, sidestepping the BM25-vs-cosine
scale-incompatibility problem. Hybrid improves NDCG roughly 26-31%
over dense-only on mixed-query benchmarks.

Reference: Cormack, Clarke & Büttcher, 2009 SIGIR.
"""

import structlog

from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("retrieval")


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
    id_key: str = "chunk_id",
    top_k: int | None = None,
) -> list[dict]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked result lists (each from a different retriever).
        k: RRF constant. Default 60 (industry standard).
                         Use 10-20 for small corpora where rank differences are more meaningful.
        id_key: Key to identify unique documents across lists.
        top_k: Number of results to return. None = all.

    Returns:
        Fused and re-ranked list of results with rrf_score field.
    """
    with create_span(
        tracer,
        "rrf_fusion",
        "CHAIN",
        {
            "fusion.type": "rrf",
            "fusion.k": k,
            "fusion.num_lists": len(ranked_lists),
        },
    ):
        # Accumulate RRF scores per document
        rrf_scores: dict[str, float] = {}
        doc_data: dict[str, dict] = {}

        for _, ranked_list in enumerate(ranked_lists):
            for rank, doc in enumerate(ranked_list, start=1):
                doc_id = doc[id_key]
                rrf_score = 1.0 / (k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

                # Keep the richest version of the document data
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc.copy()
                else:
                    # Merge scores from different retrievers
                    for key in ("dense_score", "sparse_score"):
                        if key in doc and doc[key] is not None:
                            doc_data[doc_id][key] = doc[key]

        # Build fused results
        fused_results = []
        for doc_id, score in rrf_scores.items():
            result = doc_data[doc_id].copy()
            result["rrf_score"] = round(score, 6)
            fused_results.append(result)

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        if top_k:
            fused_results = fused_results[:top_k]

        logger.info(
            "RRF fusion complete",
            input_lists=len(ranked_lists),
            unique_docs=len(rrf_scores),
            returned=len(fused_results),
            k=k,
        )

        return fused_results


def weighted_fusion(
    ranked_lists: list[list[dict]],
    weights: list[float],
    score_keys: list[str],
    id_key: str = "chunk_id",
    top_k: int | None = None,
) -> list[dict]:
    """
    Alternative: weighted alpha fusion using raw scores.

    Normalizes scores per list to [0,1] then computes weighted sum.
    Use when you want more control than RRF over retriever importance.
    """
    with create_span(
        tracer,
        "weighted_fusion",
        "CHAIN",
        {
            "fusion.type": "weighted",
            "fusion.weights": str(weights),
        },
    ):
        scores: dict[str, float] = {}
        doc_data: dict[str, dict] = {}

        for _, (ranked_list, weight, score_key) in enumerate(
            zip(ranked_lists, weights, score_keys, strict=True)
        ):
            if not ranked_list:
                continue

            # Min-max normalize scores
            raw_scores = [doc.get(score_key, 0) or 0 for doc in ranked_list]
            min_s = min(raw_scores) if raw_scores else 0
            max_s = max(raw_scores) if raw_scores else 1
            range_s = max_s - min_s if max_s != min_s else 1.0

            for doc in ranked_list:
                doc_id = doc[id_key]
                raw = doc.get(score_key, 0) or 0
                normalized = (raw - min_s) / range_s
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * normalized

                if doc_id not in doc_data:
                    doc_data[doc_id] = doc.copy()
                else:
                    for key in ("dense_score", "sparse_score"):
                        if key in doc and doc[key] is not None:
                            doc_data[doc_id][key] = doc[key]

        fused = []
        for doc_id, score in scores.items():
            result = doc_data[doc_id].copy()
            result["rrf_score"] = round(score, 6)
            fused.append(result)

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        if top_k:
            fused = fused[:top_k]

        return fused
