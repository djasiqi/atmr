"""Smoke d'intégration : Celery natif (sans monkey-patch queue.py)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_CELERY_BROKER_INTEGRATION", "").lower() not in {"1", "true", "yes"},
    reason="Set RUN_CELERY_BROKER_INTEGRATION=1 with Redis + worker Celery running",
)


def test_celery_broker_connection_native():
    from celery_app import celery as celery_app

    celery_app.connection().ensure_connection(max_retries=1)


def test_run_dispatch_task_apply_async_roundtrip():
    from celery.result import AsyncResult

    from tasks.dispatch_tasks import run_dispatch_task

    async_result = run_dispatch_task.apply_async(
        kwargs={
            "company_id": int(os.getenv("TEST_DISPATCH_COMPANY_ID", "1")),
            "for_date": os.getenv("TEST_DISPATCH_FOR_DATE", "2026-06-02"),
            "mode": "auto",
        },
        queue="default",
    )
    assert async_result.id

    result = AsyncResult(async_result.id, app=run_dispatch_task.app)
    state = result.get(timeout=int(os.getenv("CELERY_INTEGRATION_TIMEOUT", "120")))
    assert isinstance(state, dict)
