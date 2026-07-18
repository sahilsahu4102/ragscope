"""
RAGScope — Experiments API Router (Phase 4)

Create and inspect A/B experiments comparing two pipeline configurations
(e.g. reranker on vs off) on the same golden dataset.
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.eval.experiments import ExperimentRunner
from app.models import Dataset, Experiment
from app.schemas.schemas import ExperimentCreateRequest, ExperimentResponse

logger = structlog.get_logger()
router = APIRouter(prefix="/experiments", tags=["experiments"])


def _to_response(exp: Experiment) -> ExperimentResponse:
    return ExperimentResponse(
        id=str(exp.id),
        name=exp.name,
        description=exp.description,
        dataset_id=str(exp.dataset_id),
        status=exp.status,
        config_a=exp.config_a,
        config_b=exp.config_b,
        run_a_id=str(exp.run_a_id) if exp.run_a_id else None,
        run_b_id=str(exp.run_b_id) if exp.run_b_id else None,
        metrics_a=exp.metrics_a,
        metrics_b=exp.metrics_b,
        deltas=exp.deltas,
        error_message=exp.error_message,
        created_at=exp.created_at.isoformat(),
        completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
    )


@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create and run an A/B experiment. Runs both variants synchronously.

    Example config_b: {"use_reranker": false} to measure the reranker's impact.
    """
    dataset = (
        await db.execute(select(Dataset).where(Dataset.id == uuid.UUID(request.dataset_id)))
    ).scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    experiment = Experiment(
        name=request.name,
        description=request.description,
        dataset_id=dataset.id,
        config_a=request.config_a,
        config_b=request.config_b,
        status="pending",
    )
    db.add(experiment)
    await db.flush()

    runner = ExperimentRunner(db, use_llm_judge=request.use_llm_judge)
    experiment = await runner.run(experiment)

    if experiment.status == "failed":
        raise HTTPException(
            status_code=500,
            detail=f"Experiment failed: {experiment.error_message}",
        )

    return _to_response(experiment)


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(db: AsyncSession = Depends(get_db)):
    """List all experiments, most recent first."""
    experiments = (
        (await db.execute(select(Experiment).order_by(Experiment.created_at.desc())))
        .scalars()
        .all()
    )
    return [_to_response(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get one experiment with its variant metrics and deltas."""
    experiment = (
        await db.execute(select(Experiment).where(Experiment.id == uuid.UUID(experiment_id)))
    ).scalar_one_or_none()
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _to_response(experiment)
