from __future__ import annotations

from celery import shared_task  # pyright: ignore[reportMissingImports]

from ext import db
from models import Company, DriverShift
from services.dispatch.planning import serialize_shift


def _run_compliance_scan_for_company(company_id: int) -> int:
    """Scan shifts pour une entreprise (squelette compliance)."""
    try:
        q = db.session.query(DriverShift).filter(DriverShift.company_id == company_id)
        count = 0
        for s in q.limit(1000):
            _ = serialize_shift(s)
            count += 1
        return count
    except Exception:
        return 0


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
    return _run_compliance_scan_for_company(company_id)


@shared_task(name="planning.compliance_scan_all")
def compliance_scan_all() -> dict[str, int]:
    """Scan compliance pour toutes les entreprises (Beat sans argument).

    Exécuté sous app Flask via celery_app.ContextTask.
    """
    total_shifts = 0
    companies_processed = 0
    for row in db.session.query(Company.id).all():
        cid = int(row[0])
        total_shifts += _run_compliance_scan_for_company(cid)
        companies_processed += 1
    return {
        "companies_processed": companies_processed,
        "total_shifts_touched": total_shifts,
    }
