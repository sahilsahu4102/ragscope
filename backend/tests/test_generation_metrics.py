"""
RAGScope — Generation metric tests

Pins the fixes to the answer-quality heuristics: faithfulness is a precision
over the answer's content words (the old F1 against the whole context scored
grounded answers ~0.2), tokens are normalized before comparing, and declining
sentences are not counted as claims.
"""

import pytest

from app.eval.metrics.generation_metrics import (
    answer_correctness,
    answer_fact_recall,
    compute_all_generation_metrics,
    fact_present,
    faithfulness_heuristic,
    is_abstention,
    token_f1,
)

GQA = (
    "Llama 3 uses grouped query attention (GQA) with 8 key-value heads to improve inference speed."
)
NOISE = [
    f"Unrelated chunk {i} about subscriptions, carbon budgets and warmup schedules."
    for i in range(19)
]


def test_grounded_answer_stays_faithful_however_large_the_context():
    """The old F1 against all 20 chunks gave this answer about 0.2."""
    answer = "Llama 3 uses grouped query attention with 8 key-value heads [1]."
    assert faithfulness_heuristic(answer, [GQA, *NOISE]) == pytest.approx(
        faithfulness_heuristic(answer, [GQA])
    )
    assert faithfulness_heuristic(answer, [GQA]) >= 0.8


def test_unsupported_claim_lowers_faithfulness():
    grounded = "It uses 8 key-value heads."
    invented = "It uses 8 key-value heads and was trained on Mars by penguins."
    assert faithfulness_heuristic(invented, [GQA]) < faithfulness_heuristic(grounded, [GQA])


def test_punctuation_and_citations_do_not_block_matches():
    assert token_f1("The GPU.", "the gpu") == 1.0
    assert token_f1("8 key-value heads [2].", "8 key-value heads") == 1.0


def test_sources_list_is_not_scored_as_claims():
    answer = "It uses 8 key-value heads [1].\n\nSources:\n[1] llama3_paper.pdf, Page 7"
    assert faithfulness_heuristic(answer, [GQA]) == faithfulness_heuristic(
        "It uses 8 key-value heads [1].", [GQA]
    )
    assert answer_correctness(answer, "8 key-value heads") == answer_correctness(
        "It uses 8 key-value heads [1].", "8 key-value heads"
    )


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have enough information in the provided documents to answer this question.",
        "The provided context does not mention the vocabulary size.",
    ],
)
def test_pure_abstention_is_faithful_and_flagged(answer):
    assert is_abstention(answer)
    assert faithfulness_heuristic(answer, [GQA]) == 1.0
    assert compute_all_generation_metrics("q", answer, [GQA])["abstained"] == 1.0


def test_partial_answer_is_not_an_abstention_and_its_claims_are_checked():
    answer = "It uses 8 key-value heads. The context does not mention the vocabulary size."
    assert not is_abstention(answer)
    assert faithfulness_heuristic(answer, [GQA]) == faithfulness_heuristic(
        "It uses 8 key-value heads.", [GQA]
    )


@pytest.mark.parametrize(
    ("answer", "alternatives", "found"),
    [
        ("a linear warm up of 8,000 steps", ["8000"], True),
        ("a linear warm up of 8000 steps", ["8,000"], True),
        ("about 18,000 steps", ["8,000"], False),  # no match inside a longer number
        ("trained on 16K H100 GPUs", ["16k h100"], True),
        ("a peak learning rate of 8 \u00d7 10\u22125", ["8 x 10-5"], True),  # PDF text
        ("Tectonic, Meta's file system", ["Tectonic"], True),
        ("no answer here", ["Tectonic"], False),
        ("rose 2.450% in 2023", ["2.45%"], True),  # trailing zeros
        ("22.0 billion", ["22 billion"], True),
        ("the 24th, and 31st layers", ["24th and 31st layers"], True),
        ("an MFU of 38\u201343%", ["38-43%"], True),
        ("500 GtCO\u2082", ["500 GtCO2"], True),
        ("about 100 seconds", ["10"], False),  # zero-stripping stays inside decimals
        ("the iMac\u00ae and iPad\u2122", ["iMac and iPad"], True),
        ("**1.5 \u00b0C** of warming", ["1.5 degrees Celsius"], True),
        ("1.5 degrees Celsius", ["1.5\u00b0C"], True),
    ],
)
def test_fact_matching(answer, alternatives, found):
    assert fact_present(answer, alternatives) is found


def test_fact_recall_counts_each_required_fact_once():
    facts = [["4,000", "4000"], ["8,000", "8000"]]
    assert answer_fact_recall("4,000 and 8,000 warmup steps", facts) == 1.0
    assert answer_fact_recall("4000 warmup steps", facts) == 0.5
    assert answer_fact_recall("no numbers", facts) == 0.0


def test_compute_all_adds_fact_recall_only_when_facts_are_given():
    without = compute_all_generation_metrics("q", "8 heads", [GQA], "8 heads")
    with_facts = compute_all_generation_metrics("q", "8 heads", [GQA], "8 heads", [["8 heads"]])
    assert "answer_fact_recall" not in without
    assert with_facts["answer_fact_recall"] == 1.0
