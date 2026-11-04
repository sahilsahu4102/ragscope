"""
RAGScope — Golden-set builder checks

The LLM drafts questions; these checks decide which are fair to score. No
network: the checks are pure functions over chunk dicts.
"""

from app.scripts.build_golden import check_multi, check_record, check_single

GOLD = {
    "id": "g",
    "content": "We used warmup_steps = 4000 and a dropout rate of 0.1 for the base model.",
}
CHUNKS = [
    GOLD,
    {
        "id": "dup",
        "content": "We used warmup_steps = 4000 and a dropout rate of 0.1 for the base model .",
    },
    {"id": "other", "content": "Llama 3 405B uses a linear warm up of 8,000 steps."},
    {"id": "both", "content": "Elsewhere: dropout 0.3 and Tectonic storage."},
]


def _item(question="How many warmup steps did the Transformer use?", facts=None):
    return {"question": question, "answer": "4000 steps.", "facts": facts or [["4000", "4,000"]]}


def test_fair_question_is_kept_and_a_near_duplicate_chunk_is_allowed():
    assert check_single(_item(), GOLD, CHUNKS) is None


def test_fact_missing_from_its_passage_is_rejected():
    assert "not in its passage" in check_single(_item(facts=[["9999"]]), GOLD, CHUNKS)


def test_fact_stated_in_another_chunk_is_rejected():
    gold = CHUNKS[2]
    item = _item("How long was the Llama 3 405B warm up?", [["Tectonic"]])
    assert check_single(item, gold, CHUNKS) == "fact not in its passage"
    shared = {
        "id": "s",
        "content": "Section 3 separately reports 8,000 steps for a different schedule.",
    }
    assert "also stated elsewhere" in check_single(
        _item("How long was the warm up?", [["8,000"]]), gold, [*CHUNKS, shared]
    )


def test_question_that_gives_away_its_answer_is_rejected():
    item = _item("Did the Transformer use 4000 warmup steps?")
    assert "contains its answer" in check_single(item, GOLD, CHUNKS)


def test_malformed_draft_is_rejected():
    assert check_single({"question": "q", "answer": "a", "facts": ["4000"]}, GOLD, CHUNKS) == (
        "malformed fact"
    )


def test_two_hop_question_answerable_from_one_chunk_is_rejected():
    facts_a, facts_b = [["dropout 0.3"]], [["Tectonic"]]
    assert check_multi("What dropout and which storage?", facts_a, facts_b, CHUNKS) == (
        "one chunk states both facts"
    )
    assert check_multi("What warmup and which storage?", [["4000"]], facts_b, CHUNKS) is None


def test_a_rewording_only_widens_the_answer_and_is_not_checked_for_uniqueness():
    """The passage says "8,000 steps"; the extra wording "8000" also matches another chunk."""
    gold = {"id": "g2", "content": "Training used a warm up of 8,000 steps then cosine decay."}
    other = {"id": "o", "content": "Table 4 lists 8000 as the batch size."}
    item = _item("How long was the warm up?", [["8,000 steps", "8000"]])
    assert check_single(item, gold, [gold, other]) is None


def test_finished_rows_are_rechecked_single_and_two_hop():
    tectonic = {"id": "t", "content": "Meta stores checkpoints on Tectonic, its file system."}
    chunks = [*CHUNKS[:3], tectonic]
    single = {
        "question": "How many warmup steps did the Transformer use?",
        "gold_answer": "4000.",
        "gold_contexts": [GOLD["content"]],
        "answer_facts": [["4000"]],
    }
    two = {
        **single,
        "question": "How many warmup steps did the Transformer use, and where does Meta "
        "store checkpoints?",
        "gold_contexts": [GOLD["content"], tectonic["content"]],
        "answer_facts": [["4000"], ["Tectonic"]],
    }
    assert check_record(single, chunks) is None
    assert check_record(two, chunks) is None
    assert check_record({**two, "answer_facts": [["4000"], ["Colossus"]]}, chunks) == (
        "fact 'Colossus' in none of its passages"
    )
    assert check_record({**two, "answer_facts": [["4000"], ["dropout rate"]]}, chunks) == (
        "a passage contributes no fact"
    )
    assert check_record({**single, "gold_contexts": ["elsewhere"]}, chunks) == (
        "gold context not in the corpus"
    )
