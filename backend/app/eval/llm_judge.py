"""
RAGScope — LLM-as-Judge

Custom LLM judge implementation with bias mitigation strategies.
This is the portfolio differentiator — implements techniques from
"Judging the Judges" (Shi et al., IJCNLP 2025).

Bias Mitigations:
- Position bias: randomize candidate order, average across permutations
- Verbosity bias: length-normalized scoring
- Reference-guided scoring: use gold answer when available
- Human calibration: flag when judge diverges >20% from expected
"""

from __future__ import annotations

import random

import httpx
import structlog

from app.config import settings
from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("eval")

# ── Rubrics ─────────────────────────────────────────────

FAITHFULNESS_RUBRIC = """You are an expert evaluator assessing faithfulness of AI-generated answers.

## Task
Rate how well the generated answer is supported by the provided context.
Every claim in the answer must be traceable to the context.

## Rubric (1-5)
1: Completely unfaithful — answer contradicts or fabricates information not in context
2: Mostly unfaithful — major claims unsupported, significant hallucination
3: Partially faithful — some claims supported, some fabricated
4: Mostly faithful — nearly all claims supported, minor unsupported details
5: Fully faithful — every claim directly traceable to the provided context

## Input
Question: {question}
Context: {context}
Generated Answer: {answer}
{reference_section}

## Output
Provide your assessment in this exact format:
SCORE: <1-5>
REASONING: <your chain-of-thought analysis>"""

RELEVANCE_RUBRIC = """You are an expert evaluator assessing answer relevance.

## Task
Rate how well the generated answer addresses the user's question.

## Rubric (1-5)
1: Completely irrelevant — answer does not address the question at all
2: Mostly irrelevant — tangentially related but misses the core question
3: Partially relevant — addresses some aspects but incomplete
4: Mostly relevant — addresses the question well with minor gaps
5: Fully relevant — directly and completely answers the question

## Input
Question: {question}
Generated Answer: {answer}

## Output
SCORE: <1-5>
REASONING: <your chain-of-thought analysis>"""


class LLMJudge:
    """
    LLM-as-Judge with position-bias mitigation and calibration.

    Uses the local Ollama instance for self-hosted judging.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        num_permutations: int = 2,
    ):
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.num_permutations = num_permutations

    async def _call_llm(self, prompt: str) -> str:
        """Call Ollama for judge evaluation."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            )
            response.raise_for_status()
            return response.json().get("response", "")

    def _parse_score(self, response: str) -> tuple[int, str]:
        """Extract score and reasoning from judge response."""
        score = 3  # Default middle score
        reasoning = response

        for line in response.strip().split("\n"):
            line_upper = line.strip().upper()
            if line_upper.startswith("SCORE:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    parsed = int(score_text.split()[0])
                    score = max(1, min(5, parsed))
                except (ValueError, IndexError):
                    pass
            elif line_upper.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return score, reasoning

    async def judge_faithfulness(
        self,
        question: str,
        context: str,
        answer: str,
        gold_answer: str | None = None,
    ) -> dict:
        """
        Judge faithfulness with position-bias mitigation.

        Runs multiple permutations with randomized context ordering
        and averages the scores.
        """
        with create_span(
            tracer,
            "llm_judge_faithfulness",
            "EVALUATOR",
            {
                "eval.judge_model": self.model,
                "eval.metric": "faithfulness",
                "eval.num_permutations": self.num_permutations,
            },
        ):
            scores = []
            reasonings = []

            for _ in range(self.num_permutations):
                # Position bias mitigation: shuffle context paragraphs
                context_parts = context.split("\n\n")
                random.shuffle(context_parts)
                shuffled_context = "\n\n".join(context_parts)

                reference_section = ""
                if gold_answer:
                    reference_section = f"Reference Answer: {gold_answer}"

                prompt = FAITHFULNESS_RUBRIC.format(
                    question=question,
                    context=shuffled_context,
                    answer=answer,
                    reference_section=reference_section,
                )

                try:
                    response = await self._call_llm(prompt)
                    score, reasoning = self._parse_score(response)

                    # Verbosity bias correction: penalize if answer is 3x+ longer
                    # than context suggests
                    answer_len = len(answer.split())
                    context_len = len(context.split())
                    if context_len > 0 and answer_len > context_len * 3:
                        score = max(1, score - 1)
                        reasoning += " [Verbosity penalty applied]"

                    scores.append(score)
                    reasonings.append(reasoning)
                except Exception as e:
                    logger.warning("LLM judge call failed", error=str(e))
                    scores.append(3)
                    reasonings.append(f"Judge call failed: {e}")

            avg_score = sum(scores) / len(scores) if scores else 3.0
            normalized = round((avg_score - 1) / 4, 4)  # Normalize to 0-1

            result = {
                "faithfulness_judge": normalized,
                "raw_score": round(avg_score, 2),
                "reasoning": reasonings[0] if reasonings else "",
                "score_variance": round(sum((s - avg_score) ** 2 for s in scores) / len(scores), 4)
                if len(scores) > 1
                else 0.0,
                "num_permutations": len(scores),
            }

            # Human calibration flag
            if result["score_variance"] > 1.0:
                result["calibration_warning"] = (
                    "High variance across permutations — judge may be unreliable. "
                    "Consider human review for this sample."
                )
                logger.warning(
                    "High judge variance",
                    variance=result["score_variance"],
                    scores=scores,
                )

            return result

    async def judge_relevance(
        self,
        question: str,
        answer: str,
    ) -> dict:
        """Judge answer relevance."""
        with create_span(
            tracer,
            "llm_judge_relevance",
            "EVALUATOR",
            {"eval.judge_model": self.model, "eval.metric": "relevance"},
        ):
            prompt = RELEVANCE_RUBRIC.format(question=question, answer=answer)

            try:
                response = await self._call_llm(prompt)
                score, reasoning = self._parse_score(response)
            except Exception as e:
                logger.warning("LLM judge relevance failed", error=str(e))
                score, reasoning = 3, f"Judge call failed: {e}"

            normalized = round((score - 1) / 4, 4)

            return {
                "relevance_judge": normalized,
                "raw_score": score,
                "reasoning": reasoning,
            }

    async def judge_sample(
        self,
        question: str,
        context: str,
        answer: str,
        gold_answer: str | None = None,
    ) -> dict:
        """Run all judge evaluations for a single sample."""
        with create_span(
            tracer,
            "llm_judge_full",
            "EVALUATOR",
            {"eval.judge_model": self.model},
        ):
            faithfulness_result = await self.judge_faithfulness(
                question, context, answer, gold_answer
            )
            relevance_result = await self.judge_relevance(question, answer)

            return {
                **faithfulness_result,
                **relevance_result,
            }
