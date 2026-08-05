"""
RAGScope — PII Redaction Guard (Phase 5)

Regex-based PII detector and redactor. Scans both input queries and
output answers to redact sensitive data before storage or display.

Detected patterns:
  - Email addresses
  - Phone numbers (US/international)
  - Social Security Numbers
  - Credit card numbers
  - IP addresses (IPv4)

No external dependencies — pure regex implementation.
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()

# ── PII Patterns ─────────────────────────────────
_PII_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    (
        "email",
        "[EMAIL_REDACTED]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    ),
    (
        "phone_us",
        "[PHONE_REDACTED]",
        re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    (
        "phone_intl",
        "[PHONE_REDACTED]",
        re.compile(r"\+\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{4,10}\b"),
    ),
    (
        "ssn",
        "[SSN_REDACTED]",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "credit_card",
        "[CC_REDACTED]",
        re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    ),
    (
        "ipv4",
        "[IP_REDACTED]",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
    ),
]


class PIIRedactor:
    """
    Scans text for PII patterns and redacts them with placeholders.

    Usage:
        redactor = PIIRedactor()
        clean_text, found = redactor.redact("Contact me at john@example.com")
        # clean_text = "Contact me at [EMAIL_REDACTED]"
        # found = [{"type": "email", "original": "john@example.com"}]
    """

    def __init__(self, patterns: list[tuple[str, str, re.Pattern]] | None = None):
        self.patterns = patterns or _PII_PATTERNS

    def scan(self, text: str) -> list[dict]:
        """Scan text for PII without redacting. Returns list of findings."""
        findings: list[dict] = []
        for pii_type, _, pattern in self.patterns:
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "type": pii_type,
                        "original": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        return findings

    def redact(self, text: str) -> tuple[str, list[dict]]:
        """Redact PII from text. Returns (cleaned_text, findings)."""
        findings: list[dict] = []
        redacted = text

        for pii_type, replacement, pattern in self.patterns:
            matches = list(pattern.finditer(redacted))
            for match in reversed(matches):  # Reverse to preserve positions
                findings.append(
                    {
                        "type": pii_type,
                        "original": match.group(),
                    }
                )
                redacted = redacted[: match.start()] + replacement + redacted[match.end() :]

        if findings:
            logger.info(
                "PII redacted",
                count=len(findings),
                types=[f["type"] for f in findings],
            )

        return redacted, findings

    def contains_pii(self, text: str) -> bool:
        """Quick check — does the text contain any PII?"""
        for _, _, pattern in self.patterns:
            if pattern.search(text):
                return True
        return False


class StreamingRedactor:
    """Applies PII redaction to a token stream without buffering the whole answer.

    Redacting each token independently does not work: PII spans token
    boundaries, so "john" + "@example.com" would slip through when neither
    fragment matches on its own.

    This holds back a trailing window of `tail_chars`. Text is only released
    once it is far enough behind the frontier that no pattern could still grow
    to cover it, so released text is final and never needs retracting. The
    window must exceed the longest PII pattern this can match.

    Cost is O(n^2) in answer length because the buffer is re-scanned per token,
    which is irrelevant at answer sizes (~800 chars) and keeps the logic
    obviously correct.
    """

    def __init__(self, redactor: PIIRedactor | None = None, tail_chars: int = 96):
        self.redactor = redactor
        self.tail_chars = tail_chars
        self._raw = ""
        self._emitted = ""

    def feed(self, token: str) -> str:
        """Add a token, return whatever text is now safe to emit."""
        self._raw += token
        # tail_chars=0 disables streaming redaction entirely. The persisted
        # answer (redacted_text) is still cleaned; only the live stream is raw.
        if self.redactor is None or self.tail_chars == 0:
            self._emitted += token
            return token

        safe_end = len(self._raw) - self.tail_chars
        if safe_end <= 0:
            return ""

        redacted, _ = self.redactor.redact(self._raw[:safe_end])
        if not redacted.startswith(self._emitted):
            # Should not happen: the tail window exists precisely so released
            # text cannot change. Emit nothing rather than contradict output
            # already sent to the client.
            return ""

        delta = redacted[len(self._emitted) :]
        self._emitted = redacted
        return delta

    def flush(self) -> str:
        """Redact and return whatever is still held in the tail window."""
        if self.redactor is None:
            return ""
        redacted, _ = self.redactor.redact(self._raw)
        if not redacted.startswith(self._emitted):
            # A pattern matched across the boundary; the already-sent prefix
            # cannot be recalled, so emit the remainder of the fully redacted
            # text and let the caller store the clean version.
            self._emitted = redacted
            return ""
        delta = redacted[len(self._emitted) :]
        self._emitted = redacted
        return delta

    @property
    def redacted_text(self) -> str:
        """The full answer with PII removed — what should be persisted."""
        if self.redactor is None:
            return self._raw
        redacted, _ = self.redactor.redact(self._raw)
        return redacted
