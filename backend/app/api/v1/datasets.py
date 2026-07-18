"""
RAGScope — Datasets API Router

Endpoints for managing evaluation datasets (golden QA sets).
"""

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.eval.dataset_generator import DatasetGenerator
from app.models import Dataset
from app.schemas.schemas import (
    DatasetCreateRequest,
    DatasetDetailResponse,
    DatasetResponse,
    DatasetUploadRequest,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/datasets", tags=["evaluation"])


@router.get("", response_model=list[DatasetResponse])
async def list_datasets(
    db: AsyncSession = Depends(get_db),
):
    """List all evaluation datasets."""
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()

    return [
        DatasetResponse(
            id=str(ds.id),
            name=ds.name,
            version=ds.version,
            description=ds.description,
            sample_count=ds.sample_count,
            created_at=ds.created_at.isoformat(),
        )
        for ds in datasets
    ]


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a dataset with all its samples."""
    result = await db.execute(select(Dataset).where(Dataset.id == uuid.UUID(dataset_id)))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetDetailResponse(
        id=str(dataset.id),
        name=dataset.name,
        version=dataset.version,
        description=dataset.description,
        sample_count=dataset.sample_count,
        samples=dataset.samples or [],
        source_documents=dataset.source_documents,
        created_at=dataset.created_at.isoformat(),
    )


@router.post("/generate", response_model=DatasetResponse)
async def generate_dataset(
    request: DatasetCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a synthetic evaluation dataset from ingested documents.

    Uses LLM to create diverse QA pairs (factual, reasoning, analytical).
    """
    generator = DatasetGenerator(db)

    try:
        dataset = await generator.generate(
            name=request.name,
            document_ids=request.document_ids,
            num_questions=request.num_questions,
            version=request.version,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        version=dataset.version,
        description=dataset.description,
        sample_count=dataset.sample_count,
        created_at=dataset.created_at.isoformat(),
    )


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    request: DatasetUploadRequest,
    db: AsyncSession = Depends(get_db),
):
    """Upload a pre-built evaluation dataset (manual or external tool)."""
    dataset = Dataset(
        name=request.name,
        version=request.version,
        description=request.description,
        sample_count=len(request.samples),
        samples=request.samples,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        version=dataset.version,
        description=dataset.description,
        sample_count=dataset.sample_count,
        created_at=dataset.created_at.isoformat(),
    )
