"""
RAGScope — Public benchmark scoring

The metrics must match the published definitions, or the comparison with
published numbers means nothing. No network or database here.
"""

import math

from app.scripts.public_bench import multihop_rag_metrics, ndcg_at_10, sentence_windows


def test_ndcg_counts_each_document_once_with_binary_gain():
    assert ndcg_at_10(["a", "b"], {"a"}) == 1.0
    # relevant at rank 2 of 2 relevant docs: (1/log2(3)) / (1 + 1/log2(3))
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_10(["x", "a", "a"], {"a", "b"}), expected)


def test_multihop_rag_metrics_follow_the_benchmark_script():
    gold = ["Fact one.", "Fact two."]
    retrieved = ["noise", "...Fact  one. more", "noise", "noise", "Fact two. end"]
    m = multihop_rag_metrics(retrieved, gold)
    assert m["hits@10"] == 1.0 and m["hits@4"] == 1.0
    assert m["mrr@10"] == 0.5
    assert math.isclose(m["map@10"], (1 / 2 + 1 / 5) / 2)  # newly found gold / rank
    assert m["full_support@10"] == 1.0
    assert multihop_rag_metrics(["noise"] * 4 + ["Fact one."], gold)["hits@4"] == 0.0


def test_windows_keep_every_sentence_whole_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(100))
    windows = sentence_windows(text)
    assert all(len(w.split()) <= 190 for w in windows)
    for i in range(100):
        assert any(f"Sentence number {i} has" in w for w in windows)
    assert windows[0].split(".")[-2] in windows[1]  # the tail of one window starts the next
