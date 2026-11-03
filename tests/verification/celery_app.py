"""
Celery app for memory-safe verification parallelism.

Usage example:
    export VALIDATION_CELERY_BROKER_URL=redis://localhost:6379/0
    export VALIDATION_CELERY_RESULT_BACKEND=redis://localhost:6379/0
    export VALIDATION_CELERY_QUEUE=sai_verification  # optional; default below
    celery -A tests.verification.celery_app worker --loglevel=INFO --concurrency=2 -Q sai_verification

Each prefork child can use several GiB during GeoTIFF verification; size ``--concurrency``
from available RAM, not only CPU count, or the kernel OOM-killer may reap workers.
"""

from __future__ import annotations

from celery import Celery
from kombu import Queue

from tests.verification.env import env_int, env_str

_BROKER = env_str("VALIDATION_CELERY_BROKER_URL", "redis://localhost:6379/0")
_BACKEND = env_str("VALIDATION_CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
_TASK_QUEUE = env_str("VALIDATION_CELERY_QUEUE", "sai_verification")

celery_app = Celery(
    "sai_verification",
    broker=_BROKER,
    backend=_BACKEND,
    include=["tests.verification.celery_tasks"],
)

# Keep worker memory usage bounded and avoid prefetch spikes.
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=env_int("VALIDATION_CELERY_TASK_TIME_LIMIT", 3600),
    task_soft_time_limit=env_int("VALIDATION_CELERY_TASK_SOFT_TIME_LIMIT", 3300),
    # Dedicated queue so shared Redis instances are not drained by unrelated ``celery`` consumers.
    task_default_queue=_TASK_QUEUE,
    task_queues=(Queue(_TASK_QUEUE),),
    task_create_missing_queues=True,
)

# Ensure task registration also works when workers are started with this module.
from tests.verification import celery_tasks  # noqa: F401,E402
