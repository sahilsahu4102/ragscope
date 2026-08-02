"""
RAGScope — Phase 4 Tests (Observability & Experiments)

Pure-function + schema + wiring tests that run in CI without Postgres, Redis,
or Ollama. Cover cost calc, tracer attribute builders, the trace collector's
extraction helpers, and the A/B experiment math (deltas + bootstrap).
"""

from datetime import UTC

# ── Cost ──────────────────────────────────────


def test_cost_self_hosted_is_free():
    from app.observability.cost import calculate_cost, is_free_model

    assert calculate_cost("llama3.1:8b", 1000, 2000) == 0.0
    assert is_free_model("llama3.1:8b") is True
    assert is_free_model("nomic-embed-text") is True


def test_cost_hosted_model():
    from app.observability.cost import calculate_cost, is_free_model

    # gpt-4o: 2.50/1M input, 10.00/1M output
    cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
    assert cost == round(2.50 + 10.00, 8)
    assert is_free_model("gpt-4o") is False


def test_cost_unknown_model_is_free():
    from app.observability.cost import calculate_cost

    assert calculate_cost("some-unknown-model", 1000, 1000) == 0.0


def test_cost_prefix_match():
    from app.observability.cost import calculate_cost

    # Tagged variant should resolve to the llama3.1 prefix (free).
    assert calculate_cost("llama3.1:8b-instruct-q4_0", 500, 500) == 0.0


def test_estimate_tokens():
    from app.observability.cost import estimate_tokens

    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10


# ── Tracer attribute builders ─────────────────


def test_llm_attributes_shape():
    from app.observability.tracer import llm_attributes

    attrs = llm_attributes(system="ollama", model="llama3.1:8b", input_tokens=100, output_tokens=50)
    assert attrs["gen_ai.system"] == "ollama"
    assert attrs["gen_ai.request.model"] == "llama3.1:8b"
    assert attrs["gen_ai.usage.total_tokens"] == 150
    assert attrs["cost.usd"] == 0.0


def test_reranker_attributes_shape():
    from app.observability.tracer import reranker_attributes

    attrs = reranker_attributes(
        model_name="bge-reranker", top_k=5, input_documents=20, output_documents=5
    )
    assert attrs["reranker.model_name"] == "bge-reranker"
    assert attrs["reranker.input_documents"] == 20
    assert attrs["reranker.output_documents"] == 5


def test_retriever_attributes_documents():
    from app.observability.tracer import retriever_attributes

    docs = [{"chunk_id": "a", "dense_score": 0.9}, {"chunk_id": "b", "dense_score": 0.5}]
    attrs = retriever_attributes(retriever_type="dense", top_k=5, documents=docs)
    assert attrs["retriever.document_count"] == 2
    assert attrs["retrieval.documents.0.document.id"] == "a"
    assert attrs["retrieval.documents.0.document.score"] == 0.9


def test_clean_attributes_drops_none():
    from app.observability.tracer import _clean_attributes

    cleaned = _clean_attributes({"a": 1, "b": None, "c": [1, None, 2]})
    assert "b" not in cleaned
    assert cleaned["c"] == [1, 2]


def test_get_current_trace_id_outside_span():
    from app.observability.tracer import get_current_trace_id

    # No active recording span -> invalid context -> None.
    assert get_current_trace_id() is None


# ── Trace collector helpers ───────────────────


def test_collector_extract_tokens():
    from app.observability.collector import _extract_tokens

    assert _extract_tokens({"gen_ai.usage.total_tokens": 42}) == 42
    assert _extract_tokens({"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 5}) == 15
    assert _extract_tokens({}) is None


def test_collector_extract_cost_explicit():
    from app.observability.collector import _extract_cost

    assert _extract_cost({"cost.usd": 0.005}, 100) == 0.005


def test_collector_extract_cost_from_model():
    from app.observability.collector import _extract_cost

    attrs = {
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 1_000_000,
        "gen_ai.usage.output_tokens": 0,
    }
    assert _extract_cost(attrs, 1_000_000) == 2.50


def test_collector_hex_helpers():
    from app.observability.collector import _hex_span_id, _hex_trace_id

    assert _hex_trace_id(255) == "0" * 30 + "ff"
    assert _hex_span_id(255) == "0" * 14 + "ff"


def test_collector_ns_to_dt():
    from app.observability.collector import _ns_to_dt

    assert _ns_to_dt(None) is None
    dt = _ns_to_dt(1_000_000_000)  # 1 second after epoch
    assert dt is not None
    assert dt.tzinfo == UTC


def test_collector_sampling_decision():
    from app.observability.collector import SpanCollector

    assert SpanCollector(sampling_rate=1.0)._sampled(12345) is True
    assert SpanCollector(sampling_rate=0.0)._sampled(12345) is False
    # Deterministic per trace id.
    c = SpanCollector(sampling_rate=0.5)
    assert c._sampled(100) == c._sampled(100)


def test_collector_drain_empty():
    from app.observability.collector import SpanCollector

    assert SpanCollector().drain("deadbeef") == []


# ── Experiment math ───────────────────────────


def test_diff_metrics_higher_is_better():
    from app.eval.experiments import diff_metrics

    d = diff_metrics({"faithfulness": 0.70}, {"faithfulness": 0.80})
    assert d["faithfulness"]["winner"] == "B"
    assert d["faithfulness"]["delta"] == 0.1
    assert d["faithfulness"]["pct_change"] == round((0.1 / 0.70) * 100, 2)


def test_diff_metrics_lower_is_better():
    from app.eval.experiments import diff_metrics

    # For hallucination_score, a lower B should win.
    d = diff_metrics({"hallucination_score": 0.30}, {"hallucination_score": 0.20})
    assert d["hallucination_score"]["winner"] == "B"
    assert d["hallucination_score"]["delta"] == -0.1


def test_diff_metrics_tie_and_missing():
    from app.eval.experiments import diff_metrics

    d = diff_metrics({"ndcg@10": 0.5, "only_a": 0.9}, {"ndcg@10": 0.5, "only_b": 0.1})
    assert d["ndcg@10"]["winner"] == "tie"
    assert "only_a" not in d  # missing in B -> skipped
    assert "only_b" not in d


def test_bootstrap_significant_clear_separation():
    from app.eval.experiments import _bootstrap_significant

    a = [0.10, 0.12, 0.11, 0.13, 0.09]
    b = [0.30, 0.32, 0.31, 0.33, 0.29]
    significant, p = _bootstrap_significant(a, b)
    assert significant is True
    assert p == 0.0


def test_bootstrap_not_significant_identical():
    from app.eval.experiments import _bootstrap_significant

    a = [0.5, 0.5, 0.5, 0.5]
    significant, _ = _bootstrap_significant(a, a)
    assert significant is False


def test_bootstrap_too_few_samples():
    from app.eval.experiments import _bootstrap_significant

    significant, p = _bootstrap_significant([0.5], [0.9])
    assert significant is False
    assert p == 1.0


# ── Analytics helpers ─────────────────────────


def test_percentiles_empty():
    from app.api.v1.analytics import _percentiles

    p = _percentiles([])
    assert p.count == 0
    assert p.p50 is None


def test_percentiles_values():
    from app.api.v1.analytics import _percentiles

    p = _percentiles([100.0, 200.0, 300.0, 400.0, 500.0])
    assert p.count == 5
    assert p.p50 == 300.0


# ── Schema serialization ──────────────────────


def test_experiment_response_schema():
    from app.schemas.schemas import ExperimentResponse

    r = ExperimentResponse(
        id="e1",
        name="reranker on vs off",
        dataset_id="d1",
        status="completed",
        deltas={"faithfulness": {"delta": 0.1, "winner": "B"}},
        created_at="2026-07-18T00:00:00Z",
    )
    assert r.deltas is not None
    assert r.deltas["faithfulness"]["winner"] == "B"


def test_trace_detail_schema():
    from app.schemas.schemas import SpanResponse, TraceDetail

    trace = TraceDetail(
        id="t1",
        otel_trace_id="abc123",
        status="ok",
        span_count=1,
        created_at="2026-07-18T00:00:00Z",
        spans=[
            SpanResponse(
                id="s1",
                otel_span_id="span1",
                span_kind="LLM",
                name="llm_generate",
                status="ok",
            )
        ],
    )
    assert trace.spans[0].span_kind == "LLM"


# ── Router wiring ─────────────────────────────


def test_phase4_routes_mounted():
    from app.main import app

    paths = set(app.openapi()["paths"].keys())
    assert "/api/v1/experiments" in paths
    assert "/api/v1/traces" in paths
    assert "/api/v1/feedback" in paths
    assert "/api/v1/analytics/latency" in paths
    assert "/api/v1/analytics/cost" in paths
    assert "/api/v1/analytics/cache" in paths
    assert "/api/v1/analytics/throughput" in paths
