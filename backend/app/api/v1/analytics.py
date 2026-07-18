"""
RAGScope — Analytics API Router (Phase 4)

Aggregate latency, cost, cache, and throughput analytics computed from the
persisted query history. Powers the cost/latency dashboard.
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta

import numpy as np
import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.caching.semantic_cache import SemanticCache
from app.db.session import get_db
from app.models import Query as QueryModel
from app.schemas.schemas import (
    CacheAnalytics,
    CostAnalytics,
    CostPoint,
    LatencyAnalytics,
    LatencyPoint,
    ModelCost,
    Percentiles,
    ThroughputAnalytics,
    ThroughputPoint,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/analytics", tags=["analytics"])

_semantic_cache = SemanticCache()


def _window_start(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


def _percentiles(values: list[float]) -> Percentiles:
    if not values:
        return Percentiles(count=0)
    arr = np.array(values, dtype=float)
    return Percentiles(
        p50=round(float(np.percentile(arr, 50)), 2),
        p95=round(float(np.percentile(arr, 95)), 2),
        p99=round(float(np.percentile(arr, 99)), 2),
        count=len(values),
    )


def _bucket(dt: datetime) -> str:
    return dt.date().isoformat()


async def _load_queries(db: AsyncSession, days: int) -> list[QueryModel]:
    stmt = (
        select(QueryModel)
        .where(QueryModel.created_at >= _window_start(days))
        .order_by(QueryModel.created_at)
    )
    return list((await db.execute(stmt)).scalars().all())


@router.get("/latency", response_model=LatencyAnalytics)
async def latency_analytics(
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """p50/p95/p99 latency overall and per day."""
    queries = await _load_queries(db, days)
    all_latencies = [q.latency_ms for q in queries if q.latency_ms is not None]

    by_bucket: dict[str, list[float]] = defaultdict(list)
    for q in queries:
        if q.latency_ms is not None:
            by_bucket[_bucket(q.created_at)].append(q.latency_ms)

    series = []
    for bucket in sorted(by_bucket):
        p = _percentiles(by_bucket[bucket])
        series.append(LatencyPoint(bucket=bucket, p50=p.p50, p95=p.p95, p99=p.p99, count=p.count))

    return LatencyAnalytics(overall=_percentiles(all_latencies), series=series)


@router.get("/cost", response_model=CostAnalytics)
async def cost_analytics(
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Total + per-query cost, broken down by model and over time."""
    queries = await _load_queries(db, days)

    total_cost = sum(q.total_cost_usd or 0.0 for q in queries)
    total_tokens = sum(q.total_tokens or 0 for q in queries)
    count = len(queries)
    avg_cost = round(total_cost / count, 8) if count else 0.0

    model_agg: dict[str, dict] = defaultdict(lambda: {"queries": 0, "tokens": 0, "cost": 0.0})
    bucket_agg: dict[str, dict] = defaultdict(lambda: {"cost": 0.0, "queries": 0})

    for q in queries:
        model = (q.config_snapshot or {}).get("model", "unknown")
        model_agg[model]["queries"] += 1
        model_agg[model]["tokens"] += q.total_tokens or 0
        model_agg[model]["cost"] += q.total_cost_usd or 0.0

        b = bucket_agg[_bucket(q.created_at)]
        b["cost"] += q.total_cost_usd or 0.0
        b["queries"] += 1

    by_model = [
        ModelCost(
            model=model,
            queries=agg["queries"],
            tokens=agg["tokens"],
            cost_usd=round(agg["cost"], 8),
        )
        for model, agg in sorted(model_agg.items(), key=lambda kv: kv[1]["cost"], reverse=True)
    ]
    series = [
        CostPoint(bucket=b, cost_usd=round(agg["cost"], 8), queries=agg["queries"])
        for b, agg in sorted(bucket_agg.items())
    ]

    return CostAnalytics(
        total_cost_usd=round(total_cost, 8),
        total_tokens=total_tokens,
        avg_cost_per_query=avg_cost,
        by_model=by_model,
        series=series,
    )


@router.get("/cache", response_model=CacheAnalytics)
async def cache_analytics(
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Cache hit rate and estimated cost savings."""
    stats = await _semantic_cache.stats()

    # Estimate savings: each hit avoids one generation of average cost.
    queries = await _load_queries(db, days)
    costs = [q.total_cost_usd for q in queries if q.total_cost_usd]
    avg_cost = (sum(costs) / len(costs)) if costs else 0.0
    estimated_savings = round(stats.get("hits", 0) * avg_cost, 8)

    return CacheAnalytics(
        entries=stats.get("entries", 0),
        hits=stats.get("hits", 0),
        misses=stats.get("misses", 0),
        hit_rate=stats.get("hit_rate", 0.0),
        threshold=stats.get("threshold", 0.0),
        ttl_seconds=stats.get("ttl_seconds", 0),
        estimated_savings_usd=estimated_savings,
    )


@router.get("/throughput", response_model=ThroughputAnalytics)
async def throughput_analytics(
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Query throughput over time (average QPS per daily bucket)."""
    queries = await _load_queries(db, days)
    total = len(queries)

    by_bucket: dict[str, int] = defaultdict(int)
    for q in queries:
        by_bucket[_bucket(q.created_at)] += 1

    seconds_per_bucket = 86_400  # daily buckets
    series = [
        ThroughputPoint(bucket=b, count=c, qps=round(c / seconds_per_bucket, 6))
        for b, c in sorted(by_bucket.items())
    ]
    peak_qps = round(max(by_bucket.values()) / seconds_per_bucket, 6) if by_bucket else 0.0
    window_seconds = max(1, days * seconds_per_bucket)
    avg_qps = round(total / window_seconds, 6)

    return ThroughputAnalytics(
        total_queries=total,
        avg_qps=avg_qps,
        peak_qps=peak_qps,
        series=series,
    )
