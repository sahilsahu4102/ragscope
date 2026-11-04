"""
RAGScope — Generation Metrics (heuristic)

Cheap, deterministic signals for regression checks. The hosted judge
(app/eval/answer_judge.py) gives the semantic verdicts.

- answer_correctness: SQuAD-style token F1 against the gold answer
- answer_fact_recall: share of a question's required facts the answer states
- faithfulness: share of the answer's content words that appear in the context
  the generator was given
- hallucination_score: 1 - faithfulness
- answer_relevance: token F1 between question and answer (a weak proxy)
- abstained: 1 when the answer declines for lack of information

Until 2026-11 faithfulness was token-set F1 between the answer and every
retrieved chunk joined together. With 20 chunks the context's vocabulary
dwarfs any answer, so recall sat near zero and a fully grounded answer scored
about 0.2, while hallucination_score = 1 - faithfulness reported roughly 80%
hallucination for everything. Tokens were also compared raw, so "GPU." never
matched "GPU". Numbers measured before that change are not comparable.
"""

from __future__ import annotations

import re
import string
from collections import Counter

import structlog

from app.observability.tracer import create_span, get_tracer

logger = structlog.get_logger()
tracer = get_tracer("eval")

_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")
# ASCII punctuation plus the typographic marks PDFs and models emit: curly
# quotes, en/em dashes, minus sign, ellipsis, bullet, multiplication sign.
_PUNCTUATION = str.maketrans(
    "", "", string.punctuation + "\u201c\u201d\u2018\u2019\u2013\u2014\u2212\u2026\u2022\u00d7"
)
# A trailing "Sources:" list is metadata about the answer, not claims in it.
_SOURCES_SECTION = re.compile(
    r"\n[\s#*]*(?:sources(?:\s+used)?|references)[\s*]*:.*\Z", re.IGNORECASE | re.DOTALL
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# A sentence that declines for lack of information. Dropping these sentences
# rather than whole answers keeps "6 layers; the context does not mention
# Llama 3's count" from passing as a pure abstention.
_ABSTENTION = re.compile(
    r"(?:do(?:es)?n'?t|do(?:es)? not) have (?:enough|sufficient) information"
    r"|not (?:enough|sufficient) information"
    r"|(?:context|documents?|chunks?|sources?)\b[^.]{0,40}(?:do(?:es)?n'?t|do(?:es)? not) "
    r"(?:contain|provide|include|mention|specify|state|say)"
    r"|cannot (?:be )?(?:answered|determined|answer|determine)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    "is are was were be been being of to in on for and or with by as at from that this "
    "these those it its which what who how why when where does do did has have had not no "
    "can could will would than then there their they them we our you your i he she his her "
    "also such into about over per each both any all some more most other only so if but "
    "based according provided context documents document answer question information".split()
)


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, drop citation markers, punctuation and articles (SQuAD-style)."""
    text = _CITATION.sub(" ", text.lower())
    text = text.translate(_PUNCTUATION)
    return _ARTICLES.sub(" ", text).split()


def strip_sources(answer: str) -> str:
    return _SOURCES_SECTION.sub("", answer).strip()


def _claim_words(answer: str) -> list[str]:
    """Content words of the sentences that make claims (not the ones that decline)."""
    claims = " ".join(
        s for s in _SENTENCE.split(strip_sources(answer or "")) if not _ABSTENTION.search(s)
    )
    return [w for w in normalize_tokens(claims) if w not in _STOPWORDS]


def is_abstention(answer: str) -> bool:
    """True when the answer declines and states nothing else."""
    return bool((answer or "").strip()) and not _claim_words(answer)


def token_f1(prediction: str, reference: str) -> float:
    """Token F1 with multiplicity over normalized tokens."""
    pred, ref = normalize_tokens(prediction), normalize_tokens(reference)
    if not pred or not ref:
        return 0.0
    same = sum((Counter(pred) & Counter(ref)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def faithfulness_heuristic(generated_answer: str, retrieved_contexts: list[str]) -> float:
    """Share of the answer's content words found in the context it was given.

    A precision measure: the context may hold far more than the answer needs,
    and that must not count against the answer. Sentences that decline make no
    claims, so an answer that only declines cannot be unfaithful.
    """
    if not strip_sources(generated_answer or ""):
        return 0.0
    words = _claim_words(generated_answer)
    if not words:
        return 1.0
    if not retrieved_contexts:
        return 0.0
    vocabulary = set(normalize_tokens(" ".join(retrieved_contexts)))
    return sum(w in vocabulary for w in words) / len(words)


def answer_relevance_heuristic(question: str, generated_answer: str) -> float:
    """Token F1 between question and answer: a weak proxy, kept for regression checks."""
    if not question or not generated_answer:
        return 0.0
    return token_f1(question, strip_sources(generated_answer))


def answer_correctness(generated_answer: str, gold_answer: str) -> float:
    """Token F1 against the gold answer."""
    if not generated_answer or not gold_answer:
        return 0.0
    return token_f1(strip_sources(generated_answer), gold_answer)


# Markdown bold and (R)/(TM)/(C) dropped; minus sign and en dash read as a
# hyphen, the multiplication sign as x, subscript 2 (CO2) as 2.
_UNIFY = str.maketrans(
    {
        "*": None,
        "\u00ae": None,
        "\u2122": None,
        "\u00a9": None,
        "\u2212": "-",
        "\u2013": "-",
        "\u00d7": "x",
        "\u2082": "2",
    }
)
_TRAILING_ZEROS = re.compile(r"(\d+\.\d*?)0+(?!\d)")
_CELSIUS = re.compile(r"\s*(?:\u00b0\s*c\b|degrees? celsius)")


def _fact_text(text: str) -> str:
    """One spelling for the ways a fact gets written.

    4,000 = 4000; 2.450% = 2.45%; 22.0 billion = 22 billion; iMac(R) = iMac;
    1.4 degC = 1.4 degrees Celsius. Applied to answers and facts alike.
    """
    text = text.lower().translate(_UNIFY).replace(", and ", " and ")  # Oxford comma
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = _TRAILING_ZEROS.sub(lambda m: m.group(1).rstrip("."), text)
    text = _CELSIUS.sub("\u00b0c", text)
    return " ".join(text.split())


def fact_present(answer: str, alternatives: list[str]) -> bool:
    """Whether any alternative wording of one fact appears as a whole term."""
    haystack = _fact_text(answer)
    for alternative in alternatives:
        needle = _fact_text(alternative)
        if needle and re.search(rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])", haystack):
            return True
    return False


def answer_fact_recall(generated_answer: str, answer_facts: list[list[str]]) -> float:
    """Share of required facts the answer states; each fact lists accepted wordings."""
    if not answer_facts:
        return 1.0
    answer = strip_sources(generated_answer or "")
    return sum(fact_present(answer, fact) for fact in answer_facts) / len(answer_facts)


def hallucination_score(generated_answer: str, retrieved_contexts: list[str]) -> float:
    """1 - faithfulness. Lower is better."""
    return round(1.0 - faithfulness_heuristic(generated_answer, retrieved_contexts), 4)


def compute_all_generation_metrics(
    question: str,
    generated_answer: str,
    retrieved_contexts: list[str],
    gold_answer: str | None = None,
    answer_facts: list[list[str]] | None = None,
) -> dict[str, float]:
    """Compute all generation metrics in one pass."""
    with create_span(
        tracer,
        "compute_generation_metrics",
        "EVALUATOR",
        {"eval.metric_type": "generation"},
    ):
        faithfulness = faithfulness_heuristic(generated_answer, retrieved_contexts)
        metrics = {
            "faithfulness": round(faithfulness, 4),
            "answer_relevance": round(answer_relevance_heuristic(question, generated_answer), 4),
            "hallucination_score": round(1.0 - faithfulness, 4),
            "abstained": 1.0 if is_abstention(generated_answer or "") else 0.0,
        }
        if gold_answer:
            metrics["answer_correctness"] = round(
                answer_correctness(generated_answer, gold_answer), 4
            )
        if answer_facts:
            metrics["answer_fact_recall"] = round(
                answer_fact_recall(generated_answer, answer_facts), 4
            )

        logger.info("Generation metrics computed", **metrics)
        return metrics
