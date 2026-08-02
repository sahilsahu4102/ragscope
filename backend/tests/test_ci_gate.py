"""
RAGScope — CI Regression Gate Tests

Covers the pure logic of the eval gate: dataset loading and metric
resolution. No Postgres/Redis/Ollama needed, matching the rest of the suite.

The gate decides whether a PR passes, so its failure modes matter more than
its happy path — particularly that it refuses to pass when it cannot actually
check something.
"""

import io

import pytest

# ── resolve_metric ───────────────────────────────


def test_resolve_metric_exact_match():
    from app.eval.ci_gate import resolve_metric

    assert resolve_metric({"faithfulness": 0.9}, "faithfulness") == ("faithfulness", 0.9)


def test_resolve_metric_prefix_match():
    """The runner emits 'ndcg@10', but the CLI flag is --ndcg-threshold.

    Without prefix resolution the gate would never find the metric and would
    fail every run.
    """
    from app.eval.ci_gate import resolve_metric

    assert resolve_metric({"ndcg@10": 0.85}, "ndcg") == ("ndcg@10", 0.85)


def test_resolve_metric_prefers_exact_over_prefix():
    from app.eval.ci_gate import resolve_metric

    metrics = {"ndcg": 0.5, "ndcg@10": 0.9}
    assert resolve_metric(metrics, "ndcg") == ("ndcg", 0.5)


def test_resolve_metric_absent_returns_none():
    from app.eval.ci_gate import resolve_metric

    assert resolve_metric({"faithfulness": 0.9}, "ndcg") is None


# ── load_jsonl ───────────────────────────────────


def _write(tmp_path, text: str):
    p = tmp_path / "golden.jsonl"
    with io.open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def test_load_jsonl_parses_samples(tmp_path):
    from app.eval.ci_gate import load_jsonl

    p = _write(
        tmp_path,
        '{"question": "Q1", "gold_answer": "A1"}\n{"question": "Q2", "gold_answer": "A2"}\n',
    )
    samples = load_jsonl(p)
    assert [s["question"] for s in samples] == ["Q1", "Q2"]


def test_load_jsonl_skips_blank_lines(tmp_path):
    from app.eval.ci_gate import load_jsonl

    p = _write(tmp_path, '{"question": "Q1"}\n\n   \n{"question": "Q2"}\n')
    assert len(load_jsonl(p)) == 2


def test_load_jsonl_rejects_invalid_json_with_line_number(tmp_path):
    from app.eval.ci_gate import load_jsonl

    p = _write(tmp_path, '{"question": "Q1"}\nnot-json\n')
    with pytest.raises(ValueError) as exc:
        load_jsonl(p)
    assert ":2" in str(exc.value)


def test_load_jsonl_rejects_sample_without_question(tmp_path):
    """A sample with no question cannot be evaluated, so loading must fail
    loudly rather than silently shrinking the gated dataset."""
    from app.eval.ci_gate import load_jsonl

    p = _write(tmp_path, '{"question": "Q1"}\n{"gold_answer": "orphan"}\n')
    with pytest.raises(ValueError) as exc:
        load_jsonl(p)
    assert "question" in str(exc.value)


def test_load_jsonl_rejects_empty_file(tmp_path):
    """An empty dataset would make the gate pass vacuously."""
    from app.eval.ci_gate import load_jsonl

    p = _write(tmp_path, "\n\n")
    with pytest.raises(ValueError):
        load_jsonl(p)


# ── gate configuration ───────────────────────────


def test_gated_metrics_cover_cli_flags():
    """Every gated metric must map to an argparse destination, or a threshold
    would be silently ignored."""
    from app.eval.ci_gate import GATED_METRICS

    assert GATED_METRICS["ndcg"] == "ndcg_threshold"
    assert GATED_METRICS["faithfulness"] == "faithfulness_threshold"
    assert GATED_METRICS["context_recall"] == "context_recall_threshold"
