"""
RAGScope — Eval Smoke Tests

Tests that the eval infrastructure imports and metrics compute correctly.
These run in CI without Postgres/Redis/Ollama.
"""


def test_retrieval_metrics_perfect_recall():
    """Perfect retrieval should yield context_recall = 1.0."""
    from app.eval.metrics.retrieval_metrics import context_recall

    retrieved = [{"content": "The capital of France is Paris."}]
    gold = ["The capital of France is Paris."]
    assert context_recall(retrieved, gold) == 1.0


def test_retrieval_metrics_zero_recall():
    """Empty retrieval should yield context_recall = 0.0."""
    from app.eval.metrics.retrieval_metrics import context_recall

    retrieved: list[dict] = []
    gold = ["Some important fact."]
    assert context_recall(retrieved, gold) == 0.0


def test_retrieval_metrics_hit_rate_hit():
    """Hit rate = 1.0 when a relevant chunk exists."""
    from app.eval.metrics.retrieval_metrics import hit_rate

    retrieved = [{"content": "The capital of France is Paris."}]
    gold = ["The capital of France is Paris."]
    assert hit_rate(retrieved, gold) == 1.0


def test_retrieval_metrics_hit_rate_miss():
    """Hit rate = 0.0 when no relevant chunk exists."""
    from app.eval.metrics.retrieval_metrics import hit_rate

    retrieved = [{"content": "Completely irrelevant text about cooking."}]
    gold = ["The capital of France is Paris."]
    assert hit_rate(retrieved, gold) == 0.0


def test_retrieval_metrics_mrr():
    """MRR should be 0.5 when first relevant is at rank 2."""
    from app.eval.metrics.retrieval_metrics import mean_reciprocal_rank

    retrieved = [
        {"content": "Irrelevant stuff about cooking."},
        {"content": "The capital of France is Paris."},
    ]
    gold = ["The capital of France is Paris."]
    assert mean_reciprocal_rank(retrieved, gold) == 0.5


def test_retrieval_metrics_ndcg():
    """NDCG should be 1.0 for perfect ordering."""
    from app.eval.metrics.retrieval_metrics import ndcg_at_k

    retrieved = [{"content": "The capital of France is Paris."}]
    gold = ["The capital of France is Paris."]
    assert ndcg_at_k(retrieved, gold, k=1) == 1.0


def test_retrieval_metrics_compute_all():
    """compute_all_retrieval_metrics should return all expected keys."""
    from app.eval.metrics.retrieval_metrics import compute_all_retrieval_metrics

    retrieved = [{"content": "The capital of France is Paris."}]
    gold = ["The capital of France is Paris."]
    metrics = compute_all_retrieval_metrics(retrieved, gold, k=10)

    expected_keys = {
        "context_precision",
        "context_recall",
        "hit_rate",
        "mrr",
        "ndcg@10",
        "precision@10",
        "recall@10",
    }
    assert expected_keys == set(metrics.keys())
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_generation_metrics_faithfulness_grounded():
    """High faithfulness when answer is fully grounded in context."""
    from app.eval.metrics.generation_metrics import faithfulness_heuristic

    answer = "Paris is the capital of France."
    contexts = ["Paris is the capital of France."]
    score = faithfulness_heuristic(answer, contexts)
    assert score > 0.5


def test_generation_metrics_faithfulness_hallucinated():
    """Low faithfulness when answer has no context overlap."""
    from app.eval.metrics.generation_metrics import faithfulness_heuristic

    answer = "Tokyo is famous for sushi and technology."
    contexts = ["Paris is the capital of France."]
    score = faithfulness_heuristic(answer, contexts)
    assert score < 0.3


def test_generation_metrics_answer_correctness():
    """Answer correctness should be high for matching answers."""
    from app.eval.metrics.generation_metrics import answer_correctness

    generated = "Paris is the capital of France."
    gold = "The capital of France is Paris."
    score = answer_correctness(generated, gold)
    assert score > 0.5


def test_generation_metrics_hallucination_score():
    """Hallucination score = 1 - faithfulness."""
    from app.eval.metrics.generation_metrics import hallucination_score

    answer = "Completely made up stuff about aliens."
    contexts = ["Paris is the capital of France."]
    score = hallucination_score(answer, contexts)
    assert score > 0.5  # High hallucination


def test_generation_metrics_compute_all():
    """compute_all_generation_metrics should return all expected keys."""
    from app.eval.metrics.generation_metrics import compute_all_generation_metrics

    metrics = compute_all_generation_metrics(
        question="What is the capital of France?",
        generated_answer="Paris is the capital of France.",
        retrieved_contexts=["Paris is the capital of France."],
        gold_answer="Paris",
    )
    expected_keys = {
        "faithfulness",
        "answer_relevance",
        "hallucination_score",
        "answer_correctness",
    }
    assert expected_keys == set(metrics.keys())


def test_llm_judge_parse_score():
    """LLM Judge should parse SCORE: from response text."""
    from app.eval.llm_judge import LLMJudge

    judge = LLMJudge()
    score, reasoning = judge._parse_score("SCORE: 4\nREASONING: Good answer.")
    assert score == 4
    assert reasoning == "Good answer."


def test_llm_judge_parse_score_clamped():
    """LLM Judge should clamp invalid scores to 1-5 range."""
    from app.eval.llm_judge import LLMJudge

    judge = LLMJudge()
    score, _ = judge._parse_score("SCORE: 10\nREASONING: Overflow.")
    assert score == 5

    score, _ = judge._parse_score("SCORE: 0\nREASONING: Underflow.")
    assert score == 1


def test_regression_gate_schema():
    """RegressionGateResult schema should serialize properly."""
    from app.schemas.schemas import RegressionGateResult

    result = RegressionGateResult(
        passed=False,
        thresholds={"faithfulness": 0.80},
        actual={"faithfulness": 0.65},
        failures=["faithfulness: 0.6500 < 0.8000"],
    )
    assert result.passed is False
    assert len(result.failures) == 1
