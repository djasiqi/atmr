"""Gouvernance tenant (Company) — preview blast radius, suspension, états desired/observed/effective."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from ext import db
from models.booking import Booking
from models.company import Company
from models.dispatch import DispatchRun
from models.driver import Driver
from models.enums import BookingStatus, DispatchStatus

from services.effective_state_resolver import resolve_effective_tenant_state


def _active_booking_statuses() -> tuple[str, ...]:
    return (
        BookingStatus.PENDING.value,
        BookingStatus.ACCEPTED.value,
        BookingStatus.ASSIGNED.value,
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )


def get_company_or_404(company_id: int) -> Company | None:
    return db.session.get(Company, company_id)


def count_drivers(company_id: int) -> int:
    return db.session.scalar(
        select(func.count()).select_from(Driver).where(Driver.company_id == company_id)
    ) or 0


def count_active_bookings(company_id: int) -> int:
    return db.session.scalar(
        select(func.count())
        .select_from(Booking)
        .where(
            Booking.company_id == company_id,
            Booking.status.in_(_active_booking_statuses()),
        )
    ) or 0


def count_running_dispatch_runs(company_id: int) -> int:
    return db.session.scalar(
        select(func.count())
        .select_from(DispatchRun)
        .where(
            DispatchRun.company_id == company_id,
            DispatchRun.status == DispatchStatus.RUNNING,
        )
    ) or 0


def build_observed_snapshot(company_id: int, *, include_runtime_hints: bool = True) -> dict[str, Any]:
    """Agrégats « observés » pour réconciliation (sessions/sockets/GPS : enrichissements futurs)."""
    out: dict[str, Any] = {
        "drivers_count": count_drivers(company_id),
        "active_bookings_count": count_active_bookings(company_id),
        "running_dispatch_runs": count_running_dispatch_runs(company_id),
        "active_sessions_estimate": None,
        "open_sockets_estimate": None,
        "gps_recent_activity_note": (
            "Agrégat positions récentes par tenant : LATER (Redis). "
            "Voir gps_pipeline (runtime) pour santé du pipeline."
        ),
    }
    if include_runtime_hints:
        try:
            from services.platform_runtime_hints import gps_pipeline_hint

            out["gps_pipeline"] = gps_pipeline_hint()
        except Exception:
            out["gps_pipeline"] = {"status": "unknown"}
    return out


def compute_reconciliation_and_effective(
    *,
    desired_suspended: bool,
    observed: dict[str, Any],
) -> tuple[str, str, list[str]]:
    """Retourne (reconciliation_status, effective_state, badges) via resolver unique."""
    active_bookings = int(observed.get("active_bookings_count") or 0)
    running_dispatch = int(observed.get("running_dispatch_runs") or 0)
    has_residual_activity = active_bookings > 0 or running_dispatch > 0

    if not desired_suspended:
        recon = "converged"
    elif has_residual_activity:
        recon = "drift"
    else:
        recon = "converged"

    desired_state = {"operational": "suspended" if desired_suspended else "active"}
    resolved = resolve_effective_tenant_state(
        desired_state=desired_state,
        observed_state=observed,
        reconciliation_status=recon,
    )
    eff = str(resolved.get("effective_state") or "active")
    badges = list(resolved.get("ui_badges") or [])

    return recon, eff, badges


def tenant_governance_payload(company: Company) -> dict[str, Any]:
    """Payload unique pour GET tenant et réponse suspend."""
    cid = company.id
    observed = build_observed_snapshot(cid)
    desired = {"operational": "suspended" if company.platform_suspended else "active"}
    recon, effective, badges = compute_reconciliation_and_effective(
        desired_suspended=bool(company.platform_suspended),
        observed=observed,
    )
    return {
        "tenant_id": cid,
        "tenant_type": "company",
        "name": company.name,
        "desired_state": desired,
        "observed_state": observed,
        "reconciliation_status": recon,
        "effective_state": effective,
        "effective_badges": badges,
    }


def suspend_preview(company_id: int) -> dict[str, Any] | None:
    company = get_company_or_404(company_id)
    if not company:
        return None
    observed = build_observed_snapshot(company_id)
    blast = {
        "drivers_affected": observed["drivers_count"],
        "active_bookings": observed["active_bookings_count"],
        "running_dispatch_runs": observed["running_dispatch_runs"],
        "notes": [
            "Estimation conservative : réservations actives et dispatch RUNNING.",
            observed["gps_recent_activity_note"],
        ],
    }
    return {
        "tenant_id": company_id,
        "blast_radius": blast,
        "current": tenant_governance_payload(company),
        "would_set_desired": {"operational": "suspended"},
    }


def apply_suspend(
    company: Company,
) -> tuple[dict[str, Any], str]:
    """Applique platform_suspended=True et retourne payload + statut résultat (applied | partially_applied)."""
    company.platform_suspended = True
    db.session.add(company)
    db.session.flush()
    payload = tenant_governance_payload(company)
    observed = payload["observed_state"]
    recon = payload["reconciliation_status"]
    if recon == "drift":
        return payload, "partially_applied"
    return payload, "applied"
