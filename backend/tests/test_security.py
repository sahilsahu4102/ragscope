"""
RAGScope — Security Tests

Covers the two pieces added for public deployment:

  StreamingRedactor — PII spans token boundaries, so the interesting cases are
  the ones where naive per-token redaction leaks.

  RateLimiter — the interesting cases are that it actually blocks, that limits
  are per-client, and that a Redis outage fails open rather than taking the
  API down.

No services required: Redis is faked, matching the rest of the suite.
"""

from typing import Any, cast

import pytest
from fastapi import HTTPException

from app.guardrails.pii import PIIRedactor, StreamingRedactor

# ── StreamingRedactor ────────────────────────────


def _stream(tokens: list[str], tail: int = 96) -> str:
    r = StreamingRedactor(PIIRedactor(), tail_chars=tail)
    out = "".join(r.feed(t) for t in tokens)
    return out + r.flush()


def test_email_split_across_tokens_is_redacted():
    """The case per-token redaction misses: neither half is PII alone."""
    out = _stream(["Contact ", "john", "@examp", "le.com", " for details"])
    assert "john@example.com" not in out
    assert "[EMAIL_REDACTED]" in out


def test_clean_text_passes_through_unchanged():
    tokens = ["The ", "answer ", "is ", "42 ", "according ", "to ", "the ", "docs."]
    assert _stream(tokens) == "".join(tokens)


def test_nothing_released_before_tail_window_fills():
    """Text inside the tail window must be withheld — it could still become PII."""
    r = StreamingRedactor(PIIRedactor(), tail_chars=96)
    assert r.feed("short text") == ""


def test_full_text_is_recoverable_and_redacted():
    r = StreamingRedactor(PIIRedactor(), tail_chars=32)
    for t in ["Reach me at ", "a", "lice@corp.io", " anytime"]:
        r.feed(t)
    assert "alice@corp.io" not in r.redacted_text
    assert "[EMAIL_REDACTED]" in r.redacted_text


def test_disabled_redactor_is_passthrough():
    """PII redaction can be switched off; the stream must still work."""
    r = StreamingRedactor(None)
    assert r.feed("john@example.com") == "john@example.com"
    assert r.flush() == ""
    assert r.redacted_text == "john@example.com"


def test_emitted_text_is_never_retracted():
    """Released text must be final — a client cannot un-see a token."""
    r = StreamingRedactor(PIIRedactor(), tail_chars=48)
    emitted = ""
    for tok in ["Here is a fairly long sentence to push past the window, ", "bob@x.io", " end"]:
        emitted += r.feed(tok)
    final = emitted + r.flush()
    assert final.startswith(emitted)
    assert "bob@x.io" not in final


# ── RateLimiter ──────────────────────────────────


class FakeRedis:
    def __init__(self, fail: bool = False):
        self.store: dict[str, int] = {}
        self.fail = fail

    async def incr(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    async def expire(self, key, seconds):
        return True


class FakeRequest:
    def __init__(self, host: str = "1.2.3.4", headers: dict | None = None):
        self.client = type("C", (), {"host": host})()
        self.headers = headers or {}


@pytest.fixture
def limiter_env(monkeypatch):
    import app.security as sec

    fake = FakeRedis()

    async def _fake_redis():
        return fake

    monkeypatch.setattr(sec, "_get_redis", _fake_redis)
    monkeypatch.setattr(sec.settings, "rate_limit_enabled", True)
    monkeypatch.setattr(sec.settings, "trust_proxy_headers", False)
    return sec, fake


@pytest.mark.asyncio
async def test_allows_requests_under_limit(limiter_env):
    sec, _ = limiter_env
    limiter = sec.RateLimiter(limit=3, window_seconds=60, scope="t")
    for _ in range(3):
        await limiter(cast(Any, FakeRequest()))


@pytest.mark.asyncio
async def test_blocks_over_limit_with_retry_after(limiter_env):
    sec, _ = limiter_env
    limiter = sec.RateLimiter(limit=2, window_seconds=60, scope="t")
    await limiter(cast(Any, FakeRequest()))
    await limiter(cast(Any, FakeRequest()))

    with pytest.raises(HTTPException) as exc:
        await limiter(cast(Any, FakeRequest()))
    assert exc.value.status_code == 429
    # Clients need to know when to retry, or they hammer the endpoint.
    assert "Retry-After" in (exc.value.headers or {})


@pytest.mark.asyncio
async def test_limits_are_per_client(limiter_env):
    """One client exhausting its quota must not block everyone else."""
    sec, _ = limiter_env
    limiter = sec.RateLimiter(limit=1, window_seconds=60, scope="t")
    await limiter(cast(Any, FakeRequest(host="1.1.1.1")))
    with pytest.raises(HTTPException):
        await limiter(cast(Any, FakeRequest(host="1.1.1.1")))
    # Different client, own bucket.
    await limiter(cast(Any, FakeRequest(host="2.2.2.2")))


@pytest.mark.asyncio
async def test_scopes_are_independent(limiter_env):
    sec, _ = limiter_env
    q = sec.RateLimiter(limit=1, window_seconds=60, scope="query")
    h = sec.RateLimiter(limit=1, window_seconds=60, scope="heavy")
    await q(cast(Any, FakeRequest()))
    await h(cast(Any, FakeRequest()))  # separate budget


@pytest.mark.asyncio
async def test_redis_outage_fails_open(monkeypatch):
    """A limiter that cannot reach Redis must not reject all traffic."""
    import app.security as sec

    async def _broken():
        return FakeRedis(fail=True)

    monkeypatch.setattr(sec, "_get_redis", _broken)
    monkeypatch.setattr(sec.settings, "rate_limit_enabled", True)

    limiter = sec.RateLimiter(limit=1, window_seconds=60, scope="t")
    for _ in range(5):
        await limiter(cast(Any, FakeRequest()))


@pytest.mark.asyncio
async def test_disabled_limiter_is_noop(monkeypatch):
    import app.security as sec

    monkeypatch.setattr(sec.settings, "rate_limit_enabled", False)
    limiter = sec.RateLimiter(limit=1, window_seconds=60, scope="t")
    for _ in range(10):
        await limiter(cast(Any, FakeRequest()))


# ── Client identity / proxy trust ────────────────


def test_forwarded_header_ignored_when_proxy_not_trusted(monkeypatch):
    """Otherwise a caller resets their own limit by spoofing the header."""
    import app.security as sec

    monkeypatch.setattr(sec.settings, "trust_proxy_headers", False)
    req = FakeRequest(host="9.9.9.9", headers={"x-forwarded-for": "1.1.1.1"})
    assert sec.client_identity(cast(Any, req)) == "9.9.9.9"


def test_forwarded_header_used_when_proxy_trusted(monkeypatch):
    import app.security as sec

    monkeypatch.setattr(sec.settings, "trust_proxy_headers", True)
    req = FakeRequest(host="9.9.9.9", headers={"x-forwarded-for": "1.1.1.1, 10.0.0.1"})
    assert sec.client_identity(cast(Any, req)) == "1.1.1.1"


# ── API key ──────────────────────────────────────


@pytest.mark.asyncio
async def test_api_key_disabled_by_default(monkeypatch):
    import app.security as sec

    monkeypatch.setattr(sec.settings, "api_key", "")
    await sec.require_api_key(None)


@pytest.mark.asyncio
async def test_api_key_required_when_configured(monkeypatch):
    import app.security as sec

    monkeypatch.setattr(sec.settings, "api_key", "secret")
    await sec.require_api_key("secret")

    for bad in (None, "", "wrong", "secre"):
        with pytest.raises(HTTPException) as exc:
            await sec.require_api_key(bad)
        assert exc.value.status_code == 401


# ── Production readiness gate ────────────────────


def test_insecure_defaults_are_flagged(monkeypatch):
    """The point of the gate: catch config that is fine locally and unsafe live."""
    from app.config import Settings

    s = Settings(
        postgres_password=Settings.INSECURE_DEFAULT_PASSWORD,
        api_key="",
        cors_origins="*",
        rate_limit_enabled=False,
    )
    problems = s.production_readiness_errors()
    joined = " ".join(problems).lower()
    assert "password" in joined
    assert "api_key" in joined
    assert "cors" in joined
    assert "rate_limit" in joined


def test_safe_config_passes_gate():
    from app.config import Settings

    s = Settings(
        postgres_password="a-real-password",
        api_key="a-real-key",
        cors_origins="https://app.example.com",
        rate_limit_enabled=True,
    )
    assert s.production_readiness_errors() == []


def test_docs_disabled_in_production_even_if_flag_set():
    """expose_api_docs must not be able to re-open /docs in production."""
    from app.config import Settings

    s = Settings(app_env="production", expose_api_docs=True)
    assert s.is_production is True
    assert s.docs_enabled is False


def test_docs_enabled_in_development():
    from app.config import Settings

    assert Settings(app_env="development", expose_api_docs=True).docs_enabled is True
    assert Settings(app_env="development", expose_api_docs=False).docs_enabled is False


def test_cors_origins_parse_to_list():
    from app.config import Settings

    s = Settings(cors_origins="https://a.com, https://b.com ,")
    assert s.cors_origin_list == ["https://a.com", "https://b.com"]
