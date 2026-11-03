"""
RAGScope — Reciprocal Rank Fusion tests

RRF decides what the reranker ever sees, so its arithmetic and its tie order
are pinned here. Multi-hop retrieval relies on the tie order: each hop's top
chunk scores 1/(k+1), and first-seen order decides which of them leads.
"""

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def _docs(*ids: str, score_key: str = "dense_score") -> list[dict]:
    return [{"chunk_id": i, "content": i, score_key: 1.0 / n} for n, i in enumerate(ids, 1)]


def test_score_is_sum_of_reciprocal_ranks():
    fused = reciprocal_rank_fusion([_docs("a", "b", "c"), _docs("c", "a")], k=60)
    scores = {d["chunk_id"]: d["rrf_score"] for d in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61, abs=1e-6)
    assert scores["b"] == pytest.approx(1 / 62, abs=1e-6)


def test_agreement_beats_a_single_first_place():
    fused = reciprocal_rank_fusion([_docs("solo", "both"), _docs("x", "both")])
    assert fused[0]["chunk_id"] == "both"


def test_duplicate_chunks_merge_scores_from_each_retriever():
    dense = _docs("a", "b")
    sparse = _docs("b", "a", score_key="sparse_score")
    fused = {d["chunk_id"]: d for d in reciprocal_rank_fusion([dense, sparse])}
    assert len(fused) == 2
    assert {"dense_score", "sparse_score"} <= fused["a"].keys()


def test_top_k_truncates_and_none_returns_everything():
    lists = [_docs("a", "b", "c", "d")]
    assert [d["chunk_id"] for d in reciprocal_rank_fusion(lists, top_k=2)] == ["a", "b"]
    assert len(reciprocal_rank_fusion(lists, top_k=None)) == 4


def test_small_k_sharpens_rank_differences():
    def gap(k: int) -> float:
        fused = reciprocal_rank_fusion([_docs("first", "second")], k=k)
        return fused[0]["rrf_score"] / fused[1]["rrf_score"]

    assert gap(10) > gap(60)


def test_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion([_docs("hop1"), _docs("hop2"), _docs("hop3")])
    assert [d["chunk_id"] for d in fused] == ["hop1", "hop2", "hop3"]


def test_inputs_are_not_mutated():
    dense = _docs("a")
    reciprocal_rank_fusion([dense])
    assert "rrf_score" not in dense[0]


def test_empty_input_fuses_to_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
