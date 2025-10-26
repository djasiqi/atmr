from __future__ import annotations

from celery import shared_task

from ext import db
from models import DriverShift
from services.planning_service import serialize_shift


@shared_task(name="planning.autogen_from_templates")
def autogen_from_templates(company_id: int) -> int:  # noqa: ARG001
    """Génère N semaines de shifts à l'avance (squelette)."""
    return 0


@shared_task(name="planning.sync_status_from_assignments")
def sync_status_from_assignments(company_id: int) -> int:  # noqa: ARG001
    """Met à jour le statut des shifts selon l'activité chauffeur (squelette)."""
    return 0


@shared_task(name="planning.compliance_scan")
def compliance_scan(company_id: int) -> int:
    """Calcule et met à jour les compliance_flags pour les shifts (squelette)."""
    # Placeholder: ne fait rien de coûteux pour l'instant
    # Ex: marquer les shifts passés sans flags
    try:
        q = db.session.query(DriverShift).filter(
            DriverShift.company_id == company_id)
        count = 0
        for s in q.limit(1000):
            # no-op: serialize to ensure import path is valid
            _ = serialize_shift(s)
            count += 1
        return count
    except Exception:
        return 0
