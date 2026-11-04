"""
RAGScope — Multi-hop retrieval tests

The LLM and the per-hop retrieval are faked, so these pin the control flow:
which queries get retrieved, when the follow-up loop stops, and that every
hop's best chunk survives fusion.
"""

from unittest.mock import AsyncMock

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.multihop import MultiHopRetriever, merge_hops, parse_followup
from app.retrieval.pipeline import AGENTIC_MAX_FOLLOWUPS, RetrievalPipeline


class FakeTransformer:
    def __init__(self, sub_queries: list[str], replies: list[str] | None = None):
        self.sub_queries = sub_queries
        self.replies = list(replies or [])
        self.prompts: list[str] = []

    async def decompose(self, query: str) -> list[str]:
        return self.sub_queries

    async def _llm_call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "DONE"


class FakeIndex:
    """query -> ranked chunk ids; records every query retrieved."""

    def __init__(self, results: dict[str, list[str]]):
        self.results = results
        self.queries: list[str] = []

    async def __call__(self, query: str) -> list[dict]:
        self.queries.append(query)
        return [{"chunk_id": i, "content": f"text of {i}"} for i in self.results.get(query, [])]


# ── parse_followup ─────────────────────────────


@pytest.mark.parametrize("reply", ["DONE", "done.", "  \n DONE - all answered", ""])
def test_done_replies_end_the_loop(reply):
    assert parse_followup(reply, asked=["q"]) is None


@pytest.mark.parametrize(
    ("reply", "query"),
    [
        ("Query: llama 3 405b layers", "llama 3 405b layers"),
        ('"How many H100 GPUs trained Llama 3?"', "How many H100 GPUs trained Llama 3?"),
        (
            "Search query: RoPE base frequency\nBecause the evidence lacks it.",
            "RoPE base frequency",
        ),
    ],
)
def test_followup_query_is_cleaned(reply, query):
    assert parse_followup(reply, asked=["original question"]) == query


def test_repeating_an_asked_query_counts_as_done():
    assert (
        parse_followup("How many LAYERS does it have?", asked=["how many layers does it have"])
        is None
    )


# ── MultiHopRetriever ──────────────────────────


def _ranked(*ids: str) -> list[dict]:
    return [{"chunk_id": i} for i in ids]


def test_merge_keeps_the_originals_best_then_one_chunk_per_hop():
    merged = merge_hops(
        [
            _ranked("q1", "q2", "q3", "q4"),
            _ranked("hop_a", "x"),
            _ranked("q1", "y"),  # its best chunk is already placed: adds nothing
            _ranked("hop_c"),
        ],
        top_k=6,
    )
    assert [c["chunk_id"] for c in merged] == ["q1", "hop_a", "hop_c", "q2", "q3", "q4"]


def test_a_generic_chunk_shared_by_every_hop_does_not_take_the_top():
    """RRF ranked this hub first (3 x 1/62 > 1/61); measured as the agentic v1 failure."""
    ranked = [_ranked("answer", "q2"), *[_ranked(f"hop{i}_best", "hub") for i in range(3)]]
    assert reciprocal_rank_fusion(ranked)[0]["chunk_id"] == "hub"
    merged = [c["chunk_id"] for c in merge_hops(ranked, top_k=5)]
    assert merged == ["answer", "hop0_best", "hop1_best", "hop2_best", "q2"]


@pytest.mark.asyncio
async def test_each_hops_best_chunk_reaches_querys_default_top_3():
    """With 2 protected chunks, /query's top_k=3 had room for only one hop."""
    index = FakeIndex(
        {
            "Compare the layer counts": ["generic", "noise1", "noise2"],
            "Transformer layers?": ["transformer_6_layers", "generic", "noise3"],
            "Llama 3 405B layers?": ["llama_126_layers", "noise4", "noise5"],
        }
    )
    retriever = MultiHopRetriever(
        index, FakeTransformer(["Transformer layers?", "Llama 3 405B layers?"]), max_followups=0
    )
    top3 = [c["chunk_id"] for c in await retriever.retrieve("Compare the layer counts", top_k=3)]
    assert top3 == ["generic", "transformer_6_layers", "llama_126_layers"]


@pytest.mark.asyncio
async def test_followup_retrieves_what_the_evidence_lacked():
    index = FakeIndex({"q": ["c1"], "sub": ["c2"], "missing fact": ["c3"]})
    transformer = FakeTransformer(["sub"], replies=["Query: missing fact", "DONE"])
    retriever = MultiHopRetriever(index, transformer, max_followups=3)

    chunks = await retriever.retrieve("q", top_k=10)

    assert index.queries == ["q", "sub", "missing fact"]
    assert "c3" in [c["chunk_id"] for c in chunks]
    assert len(transformer.prompts) == 2  # stopped at DONE, not at max_followups
    assert "text of c1" in transformer.prompts[0] and "Question: q" in transformer.prompts[0]


@pytest.mark.asyncio
async def test_followups_are_capped():
    index = FakeIndex({})
    transformer = FakeTransformer([], replies=["one", "two", "three"])
    await MultiHopRetriever(index, transformer, max_followups=2).retrieve("q", top_k=5)
    assert index.queries == ["q", "one", "two"]


@pytest.mark.asyncio
async def test_decompose_mode_never_calls_the_followup_llm():
    transformer = FakeTransformer(["a", "b"], replies=["should not be asked"])
    await MultiHopRetriever(FakeIndex({}), transformer, max_followups=0).retrieve("q", top_k=5)
    assert transformer.prompts == []


@pytest.mark.asyncio
async def test_sub_queries_that_restate_the_question_are_retrieved_once():
    index = FakeIndex({})
    transformer = FakeTransformer(["How many layers?", "how many LAYERS", "Which GPUs?"])
    await MultiHopRetriever(index, transformer, max_followups=0).retrieve(
        "How many layers?", top_k=5
    )
    assert index.queries == ["How many layers?", "Which GPUs?"]


# ── Pipeline routing ───────────────────────────


@pytest.mark.asyncio
async def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        await RetrievalPipeline(db=None).retrieve("q", strategy="bogus")


@pytest.mark.asyncio
async def test_decompose_and_agentic_route_through_multihop(monkeypatch):
    """decompose used to retrieve on the first sub-query only and drop the rest."""
    calls: list[int] = []

    class Recorder:
        def __init__(self, retrieve_one, transformer, max_followups):
            calls.append(max_followups)

        async def retrieve(self, question, top_k):
            return [{"chunk_id": "x"}]

    monkeypatch.setattr("app.retrieval.pipeline.MultiHopRetriever", Recorder)
    pipeline = RetrievalPipeline(db=None)

    assert await pipeline.retrieve("q", query_transform="decompose") == [{"chunk_id": "x"}]
    await pipeline.retrieve("q", strategy="agentic")
    assert calls == [0, AGENTIC_MAX_FOLLOWUPS]


@pytest.mark.asyncio
async def test_decompose_drops_sub_questions_unrelated_to_the_question(monkeypatch):
    """llama3.2:3b echoed the prompt's example ("BERT's batch size") for 4 of 10 dev questions."""
    from app.retrieval.query_transform import QueryTransformer

    transformer = QueryTransformer()
    reply = "1. What batch size was BERT trained with?\n2. How many layers does Llama 3 405B have?"
    monkeypatch.setattr(transformer, "_llm_call", AsyncMock(return_value=reply))
    question = "How many layers does the Transformer encoder have, and how many does Llama 3 405B?"
    assert await transformer.decompose(question) == ["How many layers does Llama 3 405B have?"]

    monkeypatch.setattr(transformer, "_llm_call", AsyncMock(return_value="1. What is BERT?"))
    assert await transformer.decompose(question) == [question]
