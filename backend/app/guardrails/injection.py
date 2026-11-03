"""
RAGScope — Prompt Injection Detection Guard (Phase 5)

Heuristic prompt injection detector using pattern matching and
confidence scoring. Catches common injection techniques:
  - "Ignore previous instructions"
  - System/assistant role impersonation
  - Instruction override patterns
  - Encoding/obfuscation tricks

No external dependencies — pure Python implementation.
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()

# ── Injection Patterns ───────────────────────────
# Each pattern has: (name, regex, weight 0.0-1.0)
#
# Word-based alternatives are anchored with \b. Without it, "DAN" matched inside
# "guidance", "redundant" and "danger", and "act as" inside "impact as", so
# ordinary questions about the indexed IPCC and Llama 3 documents were blocked
# as jailbreaks at 0.90 confidence.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # Direct instruction override
    (
        "ignore_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+"
            r"(?:all\s+)?(?:(?:previous|prior|above|earlier|your|the)\s+){1,2}"
            r"(?:instructions|prompts|rules|constraints|guidelines)",
            re.IGNORECASE,
        ),
        0.95,
    ),
    # Role impersonation
    (
        "role_impersonation",
        re.compile(
            r"(?:\byou\s+are\s+now\b|\bact\s+as\b|\bpretend\s+(?:to\s+be|you\s+are)\b|"
            r"\bnew\s+instruction|\bsystem\s*:\s*|\bassistant\s*:\s*|"
            r"<\|?system\|?>|<<SYS>>)",
            re.IGNORECASE,
        ),
        0.85,
    ),
    # Prompt leaking. The optional second qualifier is what catches the most
    # common phrasing, "reveal your system prompt".
    (
        "prompt_leak",
        re.compile(
            r"\b(?:show|reveal|print|repeat|output|display)\s+(?:me\s+)?"
            r"(?:your|the|system|initial|original|hidden)\s+"
            r"(?:(?:system|initial|original|hidden)\s+)?"
            r"(?:prompt|instructions|rules|message)",
            re.IGNORECASE,
        ),
        0.80,
    ),
    # Jailbreak patterns. "DAN" stays case-sensitive so the name Dan is not a
    # jailbreak.
    (
        "jailbreak",
        re.compile(
            r"\b(?:(?-i:DAN)|do\s+anything\s+now|unrestricted\s+mode|"
            r"no\s+restrictions|bypass\s+(?:safety|filter)|"
            r"developer\s+mode|jailbreak\w*)\b",
            re.IGNORECASE,
        ),
        0.90,
    ),
    # Delimiter/format injection
    (
        "delimiter_injection",
        re.compile(
            r"(?:```(?:system|instruction)|---\s*(?:NEW|OVERRIDE)|"
            r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|endoftext\|>)",
            re.IGNORECASE,
        ),
        0.85,
    ),
    # Encoding tricks (base64 instruction injection)
    (
        "encoding_trick",
        re.compile(
            r"\b(?:base64|rot13|hex|decode|eval|exec)\s*[\(:]",
            re.IGNORECASE,
        ),
        0.70,
    ),
    # Multi-shot injection ("Answer the above, then..."). Phrases like "after
    # that" and "but first" are common in honest questions, so this weighs in
    # below the 0.5 threshold: it can only raise the score of a request that
    # another pattern already flagged.
    (
        "multi_step",
        re.compile(
            r"\b(?:after\s+(?:answering|that|this)|"
            r"then\s+(?:ignore|do|execute|run|forget)|"
            r"but\s+(?:first|before\s+that))\b",
            re.IGNORECASE,
        ),
        0.45,
    ),
]


class InjectionDetector:
    """
    Prompt injection detection using pattern matching and scoring.

    Usage:
        detector = InjectionDetector(threshold=0.5)
        result = detector.detect("Ignore previous instructions and...")
        # result = {
        #     "is_injection": True,
        #     "confidence": 0.95,
        #     "matched_patterns": [{"name": "ignore_instructions", "weight": 0.95}],
        #     "risk_level": "critical"
        # }
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.patterns = _INJECTION_PATTERNS

    def detect(self, text: str) -> dict:
        """Analyze text for injection patterns. Returns detection result."""
        matched: list[dict] = []

        for name, pattern, weight in self.patterns:
            if pattern.search(text):
                matched.append({"name": name, "weight": weight})

        if not matched:
            return {
                "is_injection": False,
                "confidence": 0.0,
                "matched_patterns": [],
                "risk_level": "safe",
            }

        # Aggregate confidence: max weight + diminishing bonus from others
        weights = sorted([m["weight"] for m in matched], reverse=True)
        confidence = weights[0]
        for w in weights[1:]:
            confidence += w * 0.1  # Small additive bonus for multiple patterns
        confidence = min(1.0, confidence)

        risk_level = (
            "critical"
            if confidence >= 0.85
            else "high"
            if confidence >= 0.70
            else "medium"
            if confidence >= 0.50
            else "low"
        )

        is_injection = confidence >= self.threshold

        if is_injection:
            logger.warning(
                "Prompt injection detected",
                confidence=round(confidence, 3),
                risk_level=risk_level,
                patterns=[m["name"] for m in matched],
            )

        return {
            "is_injection": is_injection,
            "confidence": round(confidence, 4),
            "matched_patterns": matched,
            "risk_level": risk_level,
        }

    def is_safe(self, text: str) -> bool:
        """Quick check — is the text safe (no injection detected)?"""
        return not self.detect(text)["is_injection"]
