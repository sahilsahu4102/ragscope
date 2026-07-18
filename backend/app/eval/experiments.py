"""
RAGScope — A/B Experiment Framework (Phase 4)

Runs the same golden dataset through two pipeline configurations (variant A vs
variant B) and reports per-metric deltas with a paired-bootstrap significance
test. This is how RAGScope answers "does the reranker actually help?" with a
number instead of a vibe.

Significance: for each metric we align per-sample scores across the two runs
(same questions, paired), then bootstrap-resample the paired differences to
estimate the probability the mean difference is truly non-zero. A metric is
flagged significant when that probability clears the configured confidence.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.eval.runner import EvalRunner
from app.models import Dataset, EvalSample, Experiment

logger = structlog.get_logger()

# Metrics where a LOWER value is better (so a negative delta is an improvement).
LOWER_IS_BETTER = {"hallucination_score", "hallucination_rate"}

# Bootstrap config
_BOOTSTRAP_ITERS = 1000
_CONFIDENCE = 0.95
_SEED = 42


def _winner(metric: str, a: float, b: float) -> str:
    """Return 'A', 'B', or 'tie' accounting for lower-is-better metrics."""
    if a == b:
        return "tie"
    b_is_better = (b < a) if metric in LOWER_IS_BETTER else (b > a)
    return "B" if b_is_better else "A"


def _bootstrap_significant(
    paired_a: list[float],
    paired_b: list[float],
    iters: int = _BOOTSTRAP_ITERS,
    confidence: float = _CONFIDENCE,
) -> tuple[bool, float]:
    """Paired bootstrap over per-sample differences.

    Returns (is_significant, p_two_sided). Deterministic via a fixed seed so
    experiment results are reproducible.
    """
    n = len(paired_a)
    if n < 2:
        return False, 1.0

    diffs = [b - a for a, b in zip(paired_a, paired_b, strict=True)]
    observed = sum(diffs) / n
    if observed == 0:
        # No measured effect — cannot be significant.
        return False, 1.0

    rng = random.Random(_SEED)
    means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)]

    # Bootstrap p-value: how often a resample lands on the opposite side of the
    # observed effect (contradicting it), doubled for a two-sided test.
    if observed > 0:
        opposite = sum(1 for m in means if m <= 0)
    else:
        opposite = sum(1 for m in means if m >= 0)

    p_two_sided = min(1.0, 2 * opposite / iters)
    is_significant = p_two_sided < (1 - confidence)
    return is_significant, round(p_two_sided, 4)


def diff_metrics(metrics_a: dict, metrics_b: dict) -> dict:
    """Compute per-metric deltas (no significance) — pure and unit-testable."""
    deltas: dict = {}
    for key in sorted(set(metrics_a) | set(metrics_b)):
        a = metrics_a.get(key)
        b = metrics_b.get(key)
        if a is None or b is None:
            continue
        delta = round(b - a, 6)
        pct = round((delta / a) * 100, 2) if a not in (0, None) else None
        deltas[key] = {
            "a": a,
            "b": b,
            "delta": delta,
            "pct_change": pct,
            "winner": _winner(key, a, b),
        }
    return deltas


class ExperimentRunner:
    """Executes an A/B experiment: two eval runs + delta analysis."""

    def __init__(self, db: AsyncSession, use_llm_judge: bool = False):
        self.db = db
        self.use_llm_judge = use_llm_judge

    async def run(self, experiment: Experiment) -> Experiment:
        """Run both variants and populate the experiment with deltas."""
        result = await self.db.execute(select(Dataset).where(Dataset.id == experiment.dataset_id))
        dataset = result.scalar_one_or_none()
        if dataset is None:
            experiment.status = "failed"
            experiment.error_message = "Dataset not found"
            await self.db.commit()
            return experiment

        experiment.status = "running"
        await self.db.flush()

        try:
            runner = EvalRunner(self.db, use_llm_judge=self.use_llm_judge)

            run_a = await runner.run(
                dataset=dataset,
                run_name=f"{experiment.name} · A",
                config_overrides=dict(experiment.config_a or {}),
            )
            run_b = await runner.run(
                dataset=dataset,
                run_name=f"{experiment.name} · B",
                config_overrides=dict(experiment.config_b or {}),
            )

            metrics_a = run_a.metrics or {}
            metrics_b = run_b.metrics or {}
            deltas = diff_metrics(metrics_a, metrics_b)

            # Add significance from paired per-sample scores.
            paired = await self._paired_samples(run_a.id, run_b.id)
            for metric, entry in deltas.items():
                pairs = paired.get(metric, [])
                a_vals = [pa for pa, _ in pairs]
                b_vals = [pb for _, pb in pairs]
                if len(a_vals) >= 2:
                    significant, p_value = _bootstrap_significant(a_vals, b_vals)
                    entry["significant"] = significant
                    entry["p_value"] = p_value
                    entry["n"] = len(a_vals)
                else:
                    entry["significant"] = False
                    entry["p_value"] = None
                    entry["n"] = len(a_vals)

            experiment.run_a_id = run_a.id
            experiment.run_b_id = run_b.id
            experiment.metrics_a = metrics_a
            experiment.metrics_b = metrics_b
            experiment.deltas = deltas
            experiment.status = "completed"
            experiment.completed_at = datetime.now(UTC)
            await self.db.commit()
            await self.db.refresh(experiment)

            logger.info(
                "Experiment completed",
                experiment_id=str(experiment.id),
                metrics_compared=len(deltas),
            )
            return experiment

        except Exception as e:
            logger.error("Experiment failed", error=str(e))
            experiment.status = "failed"
            experiment.error_message = str(e)
            await self.db.commit()
            return experiment

    async def _paired_samples(self, run_a_id, run_b_id) -> dict[str, list[tuple[float, float]]]:
        """Align per-sample metric values across both runs by sample_index."""
        rows_a = (
            (
                await self.db.execute(
                    select(EvalSample)
                    .where(EvalSample.eval_run_id == run_a_id)
                    .order_by(EvalSample.sample_index)
                )
            )
            .scalars()
            .all()
        )
        rows_b = (
            (
                await self.db.execute(
                    select(EvalSample)
                    .where(EvalSample.eval_run_id == run_b_id)
                    .order_by(EvalSample.sample_index)
                )
            )
            .scalars()
            .all()
        )

        by_index_b = {r.sample_index: (r.metrics or {}) for r in rows_b}
        paired: dict[str, list[tuple[float, float]]] = {}

        for ra in rows_a:
            mb = by_index_b.get(ra.sample_index)
            if mb is None:
                continue
            ma = ra.metrics or {}
            for metric, va in ma.items():
                vb = mb.get(metric)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    paired.setdefault(metric, []).append((float(va), float(vb)))

        return paired
