"""Tâches Celery légères pour smoke tests broker/worker (runbook incident)."""

from __future__ import annotations

from celery import shared_task


@shared_task(name="tasks.health_tasks.celery_health_ping", bind=True)
def celery_health_ping(_self) -> str:
    """Ping minimal traversant Redis — vérifie worker + broker."""
    return "ok"
