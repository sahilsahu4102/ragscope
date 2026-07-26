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
    # Explicit imports — autodiscover_tasks() only looks for a module named
    # `tasks` inside each package, so it never picked up `ingest_task.py`
    # and the worker rejected every job as an unregistered task.
    include=["app.workers.ingest_task"],
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

# Also scan app.workers for any conventionally-named `tasks` modules added later.
celery_app.autodiscover_tasks(["app.workers"])
