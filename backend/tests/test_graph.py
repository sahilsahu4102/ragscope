"""
RAGScope — Entity-graph retrieval tests

Entity extraction rules and the personalized PageRank walk, on hand-built
chunks. No database: EntityGraph is built straight from chunk dicts.
"""

from app.retrieval.graph import EntityGraph, candidate_entities


def _chunks(*texts: str) -> list[dict]:
    return [{"chunk_id": f"c{i}", "content": t} for i, t in enumerate(texts)]


# ── Extraction ─────────────────────────────────


def test_acronyms_and_letter_digit_tokens_are_names_anywhere():
    found = candidate_entities("RLHF tuned models on H100 GPUs at 405B scale")
    assert found["rlhf"] and found["h100"] and found["gpus"] and found["405b"]


def test_capitalised_word_counts_only_mid_sentence():
    found = candidate_entities("However, the servers in Cupertino stayed up. Afterwards it rained.")
    assert found["cupertino"] is True
    assert found["however"] is False
    assert found["afterwards"] is False


def test_stopwords_numbers_and_lowercase_words_are_ignored():
    assert candidate_entities("The Table shows 2023 sales and 1,000 units") == {}


# ── Graph ──────────────────────────────────────


def test_sentence_initial_only_words_never_become_entities():
    graph = EntityGraph.build(_chunks("However, x.", "However, y.", "Located in Cupertino today."))
    assert "however" not in graph.vocab
    assert "cupertino" in graph.vocab


def test_hub_entities_are_dropped():
    texts = [f"chunk {i} mentions Llama here" for i in range(30)] + ["only one names Teton here"]
    graph = EntityGraph.build(_chunks(*texts))
    assert "llama" not in graph.vocab  # in 30 of 31 chunks
    assert "teton" in graph.vocab


def test_lowercase_question_matches_entities():
    graph = EntityGraph.build(_chunks("servers in Cupertino", "the H100 accelerator"))
    seeds = graph.query_entities("where is cupertino and what is an h100?")
    assert sorted(graph.vocab[k] for k in ("cupertino", "h100")) == seeds
    assert graph.query_entities("nothing named here") == []


def test_walk_reaches_a_bridge_chunk_that_shares_no_words_with_the_question():
    graph = EntityGraph.build(
        _chunks(
            "The platform called Grand Teton hosts H100 accelerators.",  # mentions the question's entity
            "Each H100 draws 700W.",  # bridge: only linked through H100
            "Offices sit in Cupertino.",  # unrelated
        )
    )
    ranked = graph.rank(graph.query_entities("What is Teton?"), top_k=10)
    assert [c["chunk_id"] for c in ranked] == ["c0", "c1"]
    assert ranked[0]["graph_score"] > ranked[1]["graph_score"] > 0


def test_rank_is_deterministic_and_respects_top_k():
    filler = ["plain filler text"] * 100  # keeps Teton under the 5% hub cap
    graph = EntityGraph.build(_chunks(*filler, *[f"unit {i} runs on Teton" for i in range(4)]))
    seeds = graph.query_entities("teton")
    first = graph.rank(seeds, top_k=3)
    assert len(first) == 3
    assert first == graph.rank(seeds, top_k=3)


def test_no_seeds_ranks_nothing():
    assert EntityGraph.build(_chunks("servers in Cupertino")).rank([], top_k=5) == []
