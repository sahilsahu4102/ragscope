"""
RAGScope — Hallucination Detection Guard (Phase 5)

LLM-based hallucination detector that verifies whether a generated
answer is grounded in the provided context chunks.

Uses the same Ollama model as the generator to score groundedness
on a 0-1 scale. Flags answers with scores below a threshold.
"""

from __future__ import annotations

import re

import structlog

from app.config import settings
from app.http_client import get_http_client

logger = structlog.get_logger()

_HALLUCINATION_PROMPT = """You are a factual accuracy evaluator. Your task is to evaluate whether an answer is fully grounded in the provided context.

Context passages:
{context}

Answer to evaluate:
{answer}

Evaluate the answer on a scale of 0-10:
- 10: Every claim is directly supported by the context
- 7-9: Most claims are supported, minor extrapolations
- 4-6: Some claims are supported, some are not
- 1-3: Most claims are not supported by the context
- 0: The answer is completely fabricated

Output ONLY a JSON object in this exact format:
{{"score": <number>, "reasoning": "<one sentence explanation>"}}"""


class HallucinationDetector:
    """
    Detects hallucination by scoring answer groundedness against context.

    Usage:
        detector = HallucinationDetector(threshold=0.7)
        result = await detector.detect(answer="...", context_chunks=[...])
        # result = {
        #     "is_hallucination": False,
        #     "groundedness_score": 0.85,
        #     "reasoning": "All claims are supported by context passages 1 and 3."
        # }
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url

    async def detect(
        self,
        answer: str,
        context_chunks: list[dict],
    ) -> dict:
        """Score answer groundedness against context chunks."""
        if not answer or not context_chunks:
            return {
                "is_hallucination": False,
                "groundedness_score": 1.0,
                "reasoning": "No answer or context to evaluate",
            }

        # Format context
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            content = chunk.get("content", "")[:500]
            context_text += f"[{i + 1}] {content}\n\n"

        prompt = _HALLUCINATION_PROMPT.format(
            context=context_text,
            answer=answer[:2000],  # Limit answer length
        )

        try:
            client = get_http_client()
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 200,
                    },
                },
            )
            response.raise_for_status()
            result_text = response.json().get("response", "")

            # Parse score from response
            score, reasoning = self._parse_response(result_text)
            groundedness = score / 10.0
            is_hallucination = groundedness < self.threshold

            if is_hallucination:
                logger.warning(
                    "Hallucination detected",
                    groundedness=round(groundedness, 3),
                    threshold=self.threshold,
                    reasoning=reasoning,
                )
            else:
                logger.info(
                    "Answer grounded",
                    groundedness=round(groundedness, 3),
                )

            return {
                "is_hallucination": is_hallucination,
                "groundedness_score": round(groundedness, 4),
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.warning(
                "Hallucination detection failed, allowing answer",
                error=str(e),
            )
            return {
                "is_hallucination": False,
                "groundedness_score": 1.0,
                "reasoning": f"Detection failed: {e}",
            }

    @staticmethod
    def _parse_response(text: str) -> tuple[float, str]:
        """Parse score and reasoning from LLM response."""
        import json as json_mod

        # Try JSON parse first
        try:
            # Find JSON object in response
            json_match = re.search(r"\{[^}]+\}", text)
            if json_match:
                data = json_mod.loads(json_match.group())
                score = float(data.get("score", 5))
                reasoning = data.get("reasoning", "")
                return min(10.0, max(0.0, score)), reasoning
        except (json_mod.JSONDecodeError, ValueError):
            pass

        # Fallback: extract number from text
        number_match = re.search(r"(\d+\.?\d*)", text)
        if number_match:
            score = float(number_match.group(1))
            return min(10.0, max(0.0, score)), text.strip()[:200]

        return 5.0, "Could not parse score"
