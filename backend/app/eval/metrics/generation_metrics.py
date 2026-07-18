"""
RAGScope — Generation Metrics

Measures the quality of generated answers.
Uses LLM-as-Judge for semantic evaluation where simple heuristics fail.

Metrics:
- Faithfulness: is the answer supported by the retrieved context?
- Answer Relevance: does the answer address the question?
- Answer Correctness: semantic similarity to gold answer
- Hallucination Score: fraction of claims not grounded in context
"""

from __future__ import annotations

import structlog

from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("eval")


def _token_overlap_score(text_a: str, text_b: str) -> float:
    """Compute token-level F1 overlap between two texts."""
    if not text_a or not text_b:
        return 0.0

    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.0

    common = tokens_a & tokens_b
    precision = len(common) / len(tokens_a) if tokens_a else 0
    recall = len(common) / len(tokens_b) if tokens_b else 0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def faithfulness_heuristic(
    generated_answer: str,
    retrieved_contexts: list[str],
) -> float:
    """
    Faithfulness (heuristic): how much of the answer is grounded in context?

    Measures token overlap between the generated answer and retrieved contexts.
    For production, this should be supplemented by LLM-as-Judge (see llm_judge.py).
    """
    if not generated_answer or not retrieved_contexts:
        return 0.0

    combined_context = " ".join(retrieved_contexts)
    return _token_overlap_score(generated_answer, combined_context)


def answer_relevance_heuristic(
    question: str,
    generated_answer: str,
) -> float:
    """
    Answer Relevance (heuristic): does the answer address the question?

    Measures token overlap between question and answer as a proxy.
    """
    if not question or not generated_answer:
        return 0.0

    return _token_overlap_score(question, generated_answer)


def answer_correctness(
    generated_answer: str,
    gold_answer: str,
) -> float:
    """
    Answer Correctness: similarity to the gold reference answer.

    Uses token-level F1 as a lightweight proxy for semantic similarity.
    """
    if not generated_answer or not gold_answer:
        return 0.0

    return _token_overlap_score(generated_answer, gold_answer)


def hallucination_score(
    generated_answer: str,
    retrieved_contexts: list[str],
) -> float:
    """
    Hallucination Score: fraction of answer tokens NOT grounded in context.

    hallucination = 1 - faithfulness
    Lower is better. 0.0 = fully grounded. 1.0 = pure hallucination.
    """
    faith = faithfulness_heuristic(generated_answer, retrieved_contexts)
    return round(1.0 - faith, 4)


def compute_all_generation_metrics(
    question: str,
    generated_answer: str,
    retrieved_contexts: list[str],
    gold_answer: str | None = None,
) -> dict[str, float]:
    """Compute all generation metrics in one pass."""
    with create_span(
        tracer,
        "compute_generation_metrics",
        "EVALUATOR",
        {"eval.metric_type": "generation"},
    ):
        metrics = {
            "faithfulness": round(faithfulness_heuristic(generated_answer, retrieved_contexts), 4),
            "answer_relevance": round(answer_relevance_heuristic(question, generated_answer), 4),
            "hallucination_score": hallucination_score(generated_answer, retrieved_contexts),
        }

        if gold_answer:
            metrics["answer_correctness"] = round(
                answer_correctness(generated_answer, gold_answer), 4
            )

        logger.info("Generation metrics computed", **metrics)
        return metrics
