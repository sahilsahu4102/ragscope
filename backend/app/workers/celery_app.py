"""
RAGScope — Celery Application

Celery app configured with Redis as broker and result backend.
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "ragscope",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Auto-discover tasks from app.workers
celery_app.autodiscover_tasks(["app.workers"])
