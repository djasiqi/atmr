from __future__ import annotations

import logging
import os

from celery_app import celery
from services.demo.access_service import expire_due_demo_accesses
from services.demo.seed_service import (
    ensure_demo_reference_dataset,
    reset_and_seed_demo_dataset,
)

logger = logging.getLogger(__name__)


@celery.task(name="tasks.demo_access_tasks.expire_demo_accesses")
def expire_demo_accesses_task() -> dict[str, int]:
    expired_count = expire_due_demo_accesses()
    logger.info("[demo_access_task] expired_count=%s", expired_count)
    return {"expired_count": expired_count}


@celery.task(name="tasks.demo_access_tasks.ensure_demo_reference_dataset")
def ensure_demo_reference_dataset_task() -> dict[str, int]:
    """Vérifie et complète le socle démo partagé si incomplet."""
    profile = os.getenv("DEMO_SEED_PROFILE", "sales")
    summary = ensure_demo_reference_dataset(profile_name=profile)
    logger.info("[demo_seed_task] ensure profile=%s summary=%s", profile, summary)
    return summary


@celery.task(name="tasks.demo_access_tasks.reset_demo_reference_dataset")
def reset_demo_reference_dataset_task() -> dict[str, int]:
    """Réinitialise quotidiennement le dataset démo vivant."""
    profile = os.getenv("DEMO_SEED_PROFILE", "sales")
    summary = reset_and_seed_demo_dataset(profile_name=profile, reset=True)
    logger.info("[demo_seed_task] reset profile=%s summary=%s", profile, summary)
    return summary
