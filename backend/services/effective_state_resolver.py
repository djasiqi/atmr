"""Contrat unique desired / observed / reconciliation → effective_state (plateforme V1)."""

from __future__ import annotations

from typing import Any


def resolve_effective_tenant_state(
    *,
    desired_state: dict[str, Any],
    observed_state: dict[str, Any],
    reconciliation_status: str,
    incident_overlay: dict[str, Any] | None = None,
    context_type: str = "tenant",
) -> dict[str, Any]:
    """
    Entrées stables (versionnables). Sortie : effective_state, reasons, badges, severity.
    incident_overlay réservé pour extensions (incident, maintenance).
    """
    _ = context_type
    reasons: list[str] = []
    badges: list[str] = []
    severity: str = "ok"

    if incident_overlay:
        reasons.append("incident_overlay_present")

    desired_suspended = (desired_state.get("operational") or "") == "suspended"
    active_bookings = int(observed_state.get("active_bookings_count") or 0)
    running_dispatch = int(observed_state.get("running_dispatch_runs") or 0)
    has_residual = active_bookings > 0 or running_dispatch > 0

    if not desired_suspended:
        eff = "active"
        if reconciliation_status == "drift":
            badges.append("drift")
            eff = "degraded"
            severity = "warning"
            reasons.append("drift_while_active")
        return {
            "effective_state": eff,
            "reasons": reasons,
            "ui_badges": badges,
            "severity": severity,
        }

    if has_residual or reconciliation_status == "drift":
        badges.append("drift")
        reasons.append("residual_activity")
        return {
            "effective_state": "suspended_with_residual_activity",
            "reasons": reasons,
            "ui_badges": badges,
            "severity": "warning",
        }

    return {
        "effective_state": "suspended",
        "reasons": reasons,
        "ui_badges": badges,
        "severity": "info",
    }
