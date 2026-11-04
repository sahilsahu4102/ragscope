"""
RAGScope — Hosted answer judge

Grades generated answers with a stronger hosted model: correctness against the
gold answer, and faithfulness against the exact context the generator was
given. The local 3B generator grading its own answers is the model marking its
own homework, so app/eval/llm_judge.py stays for offline runs and this judge is
used when a key is set.

Models named "gemini-*" go straight to Google's API (GEMINI_API_KEY); anything
else goes through OpenRouter (OPENROUTER_API_KEY). Free tiers cap requests per
minute and per day, not tokens, so answers are graded several per request.

Labels, not 1-5 scores: three-way verdicts are easier for a judge to apply
consistently and easier to audit from the stored reason.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import httpx

from app.config import settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

JUDGE_PROMPT = """You are grading answers produced by a retrieval-augmented question answering system.

For each numbered item, grade two things.

1. correctness, against the reference answer, which is ground truth:
   - "correct": states every fact the question asks for, as in the reference. Other wording, number formats or extra correct detail are fine.
   - "partial": states some of the asked-for facts, or states them with a minor error.
   - "incorrect": states none of them, gives a different value, or does not answer.
   An answer that says the information is not available is "incorrect", even when the context really lacks it: that is credited under faithfulness, not here.

2. faithfulness, against that item's context only (not the reference, not world knowledge):
   - "faithful": every factual claim is supported by the context.
   - "minor": one small claim is not supported by the context.
   - "unfaithful": a central claim is unsupported by or contradicts the context.
   An answer that only says the context lacks the information is "faithful".

Grade each item on its own.

{items}

Reply with JSON only:
{{"verdicts": [{{"item": 1, "correctness": "correct|partial|incorrect", "faithfulness": "faithful|minor|unfaithful", "reason": "<one sentence>"}}]}}"""

ITEM = """### Item {n}
Question: {question}
Reference answer: {gold}
Context the system was given:
{context}
System answer: {answer}"""

CORRECTNESS = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
FAITHFULNESS = {"faithful": 1.0, "minor": 0.5, "unfaithful": 0.0}
_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_JSON_ANY = re.compile(r"[\[{].*[\]}]", re.DOTALL)
_RETRY_DELAY = re.compile(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"')

# Free tiers allow roughly 10-15 requests a minute; one request at a time,
# at least this far apart, stays under that without needing per-model limits.
MIN_INTERVAL_S = 4.5
_pace_lock = asyncio.Lock()
_last_call = 0.0


async def _pace() -> None:
    global _last_call
    async with _pace_lock:
        wait = _last_call + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call = time.monotonic()


def _request(model: str, prompt: str, max_tokens: int) -> tuple[str, dict, dict]:
    if model.startswith("gemini"):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        return GEMINI_URL.format(model=model), {"x-goog-api-key": settings.gemini_api_key}, body
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    return OPENROUTER_URL, {"Authorization": f"Bearer {settings.openrouter_api_key}"}, body


def _text(model: str, data: dict) -> str:
    if model.startswith("gemini"):
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    return data["choices"][0]["message"]["content"] or ""


async def llm_chat(
    client: httpx.AsyncClient,
    prompt: str,
    model: str | None = None,
    max_tokens: int = 1024,
    max_attempts: int = 5,
) -> str:
    """One paced completion at temperature 0, with retries for 429/5xx.

    Keys travel in headers, never in the URL, so they cannot end up in request
    logs. Other HTTP errors (bad key, no credits) raise at once: retrying cannot
    fix them, and every later call would fail the same way.
    """
    model = model or settings.judge_model
    url, headers, body = _request(model, prompt, max_tokens)
    last_error = ""
    response = None
    for attempt in range(max_attempts):
        await _pace()
        delay = 2.0**attempt
        try:
            response = await client.post(url, json=body, headers=headers)
            if response.status_code not in _RETRYABLE:
                response.raise_for_status()
                return _text(model, response.json())
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            hinted = _RETRY_DELAY.search(response.text)
            if hinted:  # Google says how long until the quota refills
                delay = float(hinted.group(1)) + 1
        except httpx.TransportError as e:
            last_error = str(e)[:200]
        except (KeyError, IndexError, TypeError, ValueError):
            # A 200 carrying a blocked or empty candidate instead of text.
            last_error = f"malformed response: {getattr(response, 'text', '')[:200]}"
        await asyncio.sleep(delay)
    raise RuntimeError(f"{model} failed after {max_attempts} attempts: {last_error}")


def parse_items(text: str, list_key: str) -> list[dict]:
    """The list of item dicts in a reply; ValueError if there is none.

    Accepts a bare JSON list or an object holding the list under `list_key`.
    Gemini's JSON mode returns the bare list even when the prompt shows the
    wrapper, and a first-'{'-to-last-'}' grab across list items is not JSON.
    """
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except ValueError:
        match = _JSON_ANY.search(text)  # reply wrapped in prose or a code fence
        if not match:
            raise ValueError(f"no JSON in reply: {text[:120]!r}") from None
        data = json.loads(match.group())
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0] if list_key in data[0] else data  # [{"verdicts": [...]}]
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        data = data.get(list_key, lists[0] if len(lists) == 1 else [data])
    if not isinstance(data, list):
        raise ValueError("reply holds no list of items")
    return [item for item in data if isinstance(item, dict)]


def parse_verdict(entry: dict) -> dict:
    """Map one judge verdict to scores; ValueError if its labels are unusable."""
    correctness = str(entry.get("correctness", "")).strip().lower()
    faithfulness = str(entry.get("faithfulness", "")).strip().lower()
    if correctness not in CORRECTNESS or faithfulness not in FAITHFULNESS:
        raise ValueError(f"unknown labels: {correctness!r}, {faithfulness!r}")
    return {
        "judge_correctness": CORRECTNESS[correctness],
        "judge_faithfulness": FAITHFULNESS[faithfulness],
        "judge_reason": str(entry.get("reason", ""))[:300],
    }


class AnswerJudge:
    """Grades answers in batches with settings.judge_model."""

    def __init__(self, model: str | None = None, client: httpx.AsyncClient | None = None):
        self.model = model or settings.judge_model
        _request(self.model, "", 1)  # fail now, not mid-run, if the key is missing
        self.client = client or httpx.AsyncClient(timeout=90.0)

    async def judge_batch(self, items: list[dict]) -> list[dict]:
        """items: question, gold, context, answer. One verdict dict per item.

        A verdict the judge skipped or mislabelled becomes a judge_error entry
        rather than a guessed score, so it drops out of the averages.
        """
        prompt = JUDGE_PROMPT.format(
            items="\n\n".join(
                ITEM.format(
                    n=i,
                    question=it["question"],
                    gold=it.get("gold") or "(none given)",
                    context=it["context"],
                    answer=it["answer"],
                )
                for i, it in enumerate(items, 1)
            )
        )
        try:
            reply = await llm_chat(self.client, prompt, self.model, 300 * len(items))
            verdicts = parse_items(reply, "verdicts")
        except (ValueError, RuntimeError) as e:
            return [{"judge_error": str(e)[:200], "judge_model": self.model} for _ in items]
        by_item = {}
        for verdict in verdicts:
            digits = re.sub(r"\D", "", str(verdict.get("item", "")))  # 2, "2", "Item 2"
            if digits:
                by_item[int(digits)] = verdict
        if set(by_item) != set(range(1, len(items) + 1)) and len(verdicts) == len(items):
            by_item = dict(enumerate(verdicts, 1))  # unnumbered or renumbered: take in order
        results = []
        for i in range(1, len(items) + 1):
            try:
                results.append({**parse_verdict(by_item[i]), "judge_model": self.model})
            except (KeyError, ValueError) as e:
                results.append(
                    {
                        "judge_error": f"item {i}: {e!r}; reply: {reply[:150]}",
                        "judge_model": self.model,
                    }
                )
        return results

    async def judge(self, question: str, gold: str, context: str, answer: str) -> dict:
        items = [{"question": question, "gold": gold, "context": context, "answer": answer}]
        return (await self.judge_batch(items))[0]

    async def aclose(self) -> None:
        await self.client.aclose()
