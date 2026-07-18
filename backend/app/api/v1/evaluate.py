"""
RAGScope — Evaluate API Router

Endpoints for triggering and viewing evaluation runs.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.eval.runner import EvalRunner
from app.models import Dataset, EvalRun, EvalSample
from app.schemas.schemas import (
    EvalRunDetailResponse,
    EvalRunRequest,
    EvalRunResponse,
    EvalSampleResponse,
    RegressionGateResult,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/evaluate", tags=["evaluation"])

# ── CI Regression Thresholds ──────────────────
DEFAULT_THRESHOLDS = {
    "faithfulness": 0.80,
    "context_recall": 0.70,
    "context_precision": 0.60,
}


@router.post("", response_model=EvalRunResponse)
async def trigger_eval_run(
    request: EvalRunRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger a full evaluation run on a dataset.

    Runs: retrieve → generate → compute metrics for each sample.
    Stores per-sample and aggregate results.
    """
    # Load dataset
    result = await db.execute(select(Dataset).where(Dataset.id == uuid.UUID(request.dataset_id)))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    runner = EvalRunner(db, use_llm_judge=request.use_llm_judge)

    try:
        eval_run = await runner.run(
            dataset=dataset,
            run_name=request.run_name,
            config_overrides=request.config_overrides,
        )
    except Exception as e:
        logger.error("Eval run failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Eval run failed: {e}")

    return EvalRunResponse(
        id=str(eval_run.id),
        dataset_id=str(eval_run.dataset_id),
        name=eval_run.name,
        status=eval_run.status,
        config_snapshot=eval_run.config_snapshot,
        metrics=eval_run.metrics,
        total_samples=eval_run.total_samples,
        passed_samples=eval_run.passed_samples,
        failed_samples=eval_run.failed_samples,
        total_latency_ms=eval_run.total_latency_ms,
        created_at=eval_run.created_at.isoformat(),
        completed_at=eval_run.completed_at.isoformat() if eval_run.completed_at else None,
    )


@router.get("", response_model=list[EvalRunResponse])
async def list_eval_runs(
    db: AsyncSession = Depends(get_db),
):
    """List all evaluation runs, most recent first."""
    result = await db.execute(select(EvalRun).order_by(EvalRun.created_at.desc()))
    runs = result.scalars().all()

    return [
        EvalRunResponse(
            id=str(run.id),
            dataset_id=str(run.dataset_id),
            name=run.name,
            status=run.status,
            config_snapshot=run.config_snapshot,
            metrics=run.metrics,
            total_samples=run.total_samples,
            passed_samples=run.passed_samples,
            failed_samples=run.failed_samples,
            total_latency_ms=run.total_latency_ms,
            created_at=run.created_at.isoformat(),
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
        )
        for run in runs
    ]


@router.get("/{run_id}", response_model=EvalRunDetailResponse)
async def get_eval_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed evaluation run results including per-sample metrics."""
    result = await db.execute(select(EvalRun).where(EvalRun.id == uuid.UUID(run_id)))
    eval_run = result.scalar_one_or_none()
    if not eval_run:
        raise HTTPException(status_code=404, detail="Eval run not found")

    # Load samples
    samples_result = await db.execute(
        select(EvalSample)
        .where(EvalSample.eval_run_id == eval_run.id)
        .order_by(EvalSample.sample_index)
    )
    samples = samples_result.scalars().all()

    return EvalRunDetailResponse(
        id=str(eval_run.id),
        dataset_id=str(eval_run.dataset_id),
        name=eval_run.name,
        status=eval_run.status,
        config_snapshot=eval_run.config_snapshot,
        metrics=eval_run.metrics,
        total_samples=eval_run.total_samples,
        passed_samples=eval_run.passed_samples,
        failed_samples=eval_run.failed_samples,
        total_latency_ms=eval_run.total_latency_ms,
        created_at=eval_run.created_at.isoformat(),
        completed_at=eval_run.completed_at.isoformat() if eval_run.completed_at else None,
        samples=[
            EvalSampleResponse(
                sample_index=s.sample_index,
                question=s.question,
                gold_answer=s.gold_answer,
                generated_answer=s.generated_answer,
                metrics=s.metrics,
                judge_reasoning=s.judge_reasoning,
                retrieval_latency_ms=s.retrieval_latency_ms,
                generation_latency_ms=s.generation_latency_ms,
            )
            for s in samples
        ],
    )


@router.get("/{run_id}/gate", response_model=RegressionGateResult)
async def check_regression_gate(
    run_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    CI Regression Gate — check if an eval run passes quality thresholds.

    Used by GitHub Actions to fail PRs if metrics drop below thresholds.
    Returns exit-code-friendly result (passed: true/false).
    """
    result = await db.execute(select(EvalRun).where(EvalRun.id == uuid.UUID(run_id)))
    eval_run = result.scalar_one_or_none()
    if not eval_run:
        raise HTTPException(status_code=404, detail="Eval run not found")

    if eval_run.status != "completed":
        raise HTTPException(status_code=400, detail=f"Eval run status: {eval_run.status}")

    metrics = eval_run.metrics or {}
    failures = []
    actual = {}

    for metric_name, threshold in DEFAULT_THRESHOLDS.items():
        value = metrics.get(metric_name, 0.0)
        actual[metric_name] = value
        if value < threshold:
            failures.append(f"{metric_name}: {value:.4f} < {threshold:.4f}")

    return RegressionGateResult(
        passed=len(failures) == 0,
        thresholds=DEFAULT_THRESHOLDS,
        actual=actual,
        failures=failures,
    )
