"""
RAGScope — Token Counting & Cost Calculation (Phase 4)

Maps token usage to USD cost per model. Self-hosted models (Ollama) are free
to run, so they carry a $0 rate — but we still track tokens for throughput and
capacity planning, and we keep real API rates so a config switch to a hosted
model surfaces the true cost delta on the analytics dashboard.

Prices are USD per 1M tokens (input / output). Update as vendor pricing changes.
"""

from __future__ import annotations

# model name (or prefix) -> (input_per_1m, output_per_1m) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # ── Self-hosted (Ollama) — no per-token cost ──
    "llama3.1:8b": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),
    "qwen3": (0.0, 0.0),
    "nomic-embed-text": (0.0, 0.0),
    "bge-m3": (0.0, 0.0),
    "ollama": (0.0, 0.0),
    # ── Google Gemini (hosted reference rates) ──
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "text-embedding-004": (0.0125, 0.0),
    # ── OpenAI (hosted reference rates) ──
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}

_DEFAULT_RATE = (0.0, 0.0)


def _lookup_rate(model: str) -> tuple[float, float]:
    """Resolve pricing for a model name, tolerating tags/prefixes."""
    if not model:
        return _DEFAULT_RATE
    key = model.strip().lower()
    if key in MODEL_PRICING:
        return MODEL_PRICING[key]
    # Prefix match (e.g. "llama3.1:8b-instruct-q4" -> "llama3.1")
    for prefix, rate in MODEL_PRICING.items():
        if key.startswith(prefix):
            return rate
    return _DEFAULT_RATE


def calculate_cost(model: str, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Return USD cost for a generation given token counts.

    Returns 0.0 for self-hosted models or unknown models (fail-safe: never
    over-report cost).
    """
    in_rate, out_rate = _lookup_rate(model)
    cost = (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate
    return round(cost, 8)


def estimate_tokens(text: str) -> int:
    """Cheap heuristic token count (~4 chars/token) when no exact count exists.

    Used only as a fallback — real usage counts come from the LLM response
    (Ollama's eval_count / prompt_eval_count).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def is_free_model(model: str) -> bool:
    """True when the model has no per-token cost (self-hosted)."""
    return _lookup_rate(model) == (0.0, 0.0)
