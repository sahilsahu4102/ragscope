"""
RAGScope — Hosted answer judge tests

HTTP is mocked: these pin provider routing, what is retried, what fails fast,
and how batched replies become per-answer scores.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.eval import answer_judge
from app.eval.answer_judge import AnswerJudge, llm_chat, parse_items, parse_verdict

GEMINI = "gemini-3.5-flash-lite"


def _gemini(text: str, status: int = 200, body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/x")
    payload = body or {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return httpx.Response(status, json=payload, request=request)


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """No pacing or backoff delays in tests; keys present by default."""
    monkeypatch.setattr(answer_judge, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(answer_judge.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(answer_judge.settings, "gemini_api_key", "g-key")
    monkeypatch.setattr(answer_judge.settings, "openrouter_api_key", "or-key")


def _client(*responses) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(side_effect=list(responses))
    return client


@pytest.mark.asyncio
async def test_gemini_models_go_to_google_with_the_key_in_a_header():
    client = _client(_gemini('{"ok": 1}'))
    assert await llm_chat(client, "p", GEMINI) == '{"ok": 1}'
    call = client.post.await_args
    assert "generativelanguage.googleapis.com" in call.args[0]
    assert "key=" not in call.args[0]
    assert call.kwargs["headers"] == {"x-goog-api-key": "g-key"}


@pytest.mark.asyncio
async def test_other_models_go_through_openrouter():
    request = httpx.Request("POST", answer_judge.OPENROUTER_URL)
    reply = httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]}, request=request)
    client = _client(reply)
    assert await llm_chat(client, "p", "anthropic/claude-haiku-5.5") == "hi"
    assert client.post.await_args.kwargs["headers"] == {"Authorization": "Bearer or-key"}


@pytest.mark.asyncio
async def test_quota_429_waits_as_long_as_google_says():
    limited = _gemini("", 429, {"error": {"details": [{"retryDelay": "17s"}]}})
    client = _client(limited, _gemini("done"))
    assert await llm_chat(client, "p", GEMINI) == "done"
    answer_judge.asyncio.sleep.assert_any_await(18.0)


@pytest.mark.asyncio
async def test_bad_key_fails_fast():
    client = _client(_gemini("", 403, {"error": {"message": "denied"}}))
    with pytest.raises(httpx.HTTPStatusError):
        await llm_chat(client, "p", GEMINI)
    assert client.post.await_count == 1


def test_missing_key_is_caught_before_any_call(monkeypatch):
    monkeypatch.setattr(answer_judge.settings, "gemini_api_key", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        AnswerJudge(model=GEMINI)


@pytest.mark.parametrize(
    ("entry", "correctness", "faithfulness"),
    [
        ({"correctness": "correct", "faithfulness": "faithful"}, 1.0, 1.0),
        ({"correctness": "Partial", "faithfulness": "minor"}, 0.5, 0.5),
        ({"correctness": "incorrect", "faithfulness": "unfaithful"}, 0.0, 0.0),
    ],
)
def test_verdict_labels_map_to_scores(entry, correctness, faithfulness):
    verdict = parse_verdict(entry)
    assert (verdict["judge_correctness"], verdict["judge_faithfulness"]) == (
        correctness,
        faithfulness,
    )


@pytest.mark.parametrize(
    "reply",
    [
        '[{"item": 1}, {"item": 2}]',  # Gemini's JSON mode: a bare list, wrapper ignored
        '{"verdicts": [{"item": 1}, {"item": 2}]}',
        '[{"verdicts": [{"item": 1}, {"item": 2}]}]',  # the wrapper inside a list
        'Sure:\n```json\n[{"item": 1}, {"item": 2}]\n```',
    ],
)
def test_items_parse_from_a_bare_list_a_wrapper_or_a_fenced_reply(reply):
    """A first-{-to-last-} grab across a bare list is not JSON; that lost whole batches."""
    assert [i["item"] for i in parse_items(reply, "verdicts")] == [1, 2]


def test_unusable_replies_raise():
    with pytest.raises(ValueError):
        parse_items("no json at all", "verdicts")
    with pytest.raises(ValueError):
        parse_verdict({"correctness": "mostly", "faithfulness": "faithful"})


@pytest.mark.asyncio
async def test_batch_maps_verdicts_by_item_and_flags_the_ones_it_cannot_use():
    reply = json.dumps(
        {
            "verdicts": [
                {"item": 2, "correctness": "partial", "faithfulness": "faithful", "reason": "r"},
                {"item": 1, "correctness": "correct", "faithfulness": "minor", "reason": "r"},
                {"item": 3, "correctness": "sort of", "faithfulness": "faithful"},
            ]
        }
    )
    judge = AnswerJudge(model=GEMINI, client=_client(_gemini(reply)))
    items = [{"question": f"q{i}", "gold": "g", "context": "c", "answer": "a"} for i in range(4)]

    results = await judge.judge_batch(items)

    assert results[0]["judge_correctness"] == 1.0 and results[0]["judge_faithfulness"] == 0.5
    assert results[1]["judge_correctness"] == 0.5
    assert "judge_error" in results[2]  # mislabelled
    assert "judge_error" in results[3]  # missing from the reply
    assert all("judge_correctness" not in r for r in results[2:])


def test_a_list_under_another_key_is_still_found():
    assert [i["item"] for i in parse_items('{"results": [{"item": 1}]}', "verdicts")] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("numbers", [[None, None], [0, 1], ["Item 1", "Item 2"]])
async def test_unnumbered_or_renumbered_verdicts_are_taken_in_order(numbers):
    """One batch came back with no usable item numbers and lost all four grades."""
    entries = [
        {"correctness": label, "faithfulness": "faithful"} for label in ("correct", "incorrect")
    ]
    for entry, number in zip(entries, numbers, strict=True):
        if number is not None:
            entry["item"] = number
    judge = AnswerJudge(model=GEMINI, client=_client(_gemini(json.dumps(entries))))
    items = [{"question": "q", "gold": "g", "context": "c", "answer": "a"}] * 2

    results = await judge.judge_batch(items)

    assert [r["judge_correctness"] for r in results] == [1.0, 0.0]
