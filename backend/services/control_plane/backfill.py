"""Backfill idempotent control plane (CP-PR1)."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select

from ext import db
from models.company import Company
from models.driver import Driver
from models.institution import Institution
from models.user import User
from services.control_plane.classification import (
    apply_user_classification,
    classify_user_data_origin,
)
from services.control_plane.projector import get_projector
from services.control_plane.reconcile import reconcile_control_plane
from services.control_plane.seed import seed_control_plane_catalogs

logger = logging.getLogger(__name__)


def backfill_control_plane(*, dry_run: bool = False) -> dict[str, Any]:
    """Seed + projection complète + classification users + reconcile apply."""
    seed_control_plane_catalogs(commit=False)
    projector = get_projector()

    users = db.session.scalars(select(User)).all()
    for u in users:
        apply_user_classification(u, classify_user_data_origin(u))

    # Institutions d'abord
    for inst in db.session.scalars(select(Institution)).all():
        org = projector.ensure_institution_organization(inst)
        projector.ensure_shadow_entitlements_institution(org)
        for u in db.session.scalars(
            select(User).where(User.institution_id == inst.id)
        ).all():
            projector.sync_institution_user(u)

    # Companies (tenant only via classify)
    for company in db.session.scalars(select(Company)).all():
        projector.ensure_company_organization(company)

    for driver in db.session.scalars(select(Driver)).all():
        projector.sync_driver(driver)

    if dry_run:
        db.session.rollback()
        return {"dry_run": True}

    db.session.commit()
    # Reconcile persist anomalies
    return reconcile_control_plane(dry_run=False, apply_projection=True)
