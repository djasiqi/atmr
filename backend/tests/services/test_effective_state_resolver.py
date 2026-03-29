"""Tests unitaires — effective_state_resolver (contrat unique)."""

from __future__ import annotations

from services.effective_state_resolver import resolve_effective_tenant_state


def test_active_converged():
    r = resolve_effective_tenant_state(
        desired_state={"operational": "active"},
        observed_state={"active_bookings_count": 0, "running_dispatch_runs": 0},
        reconciliation_status="converged",
    )
    assert r["effective_state"] == "active"


def test_suspended_residual():
    r = resolve_effective_tenant_state(
        desired_state={"operational": "suspended"},
        observed_state={"active_bookings_count": 1, "running_dispatch_runs": 0},
        reconciliation_status="drift",
    )
    assert r["effective_state"] == "suspended_with_residual_activity"
    assert "drift" in r["ui_badges"]


def test_suspended_clean():
    r = resolve_effective_tenant_state(
        desired_state={"operational": "suspended"},
        observed_state={"active_bookings_count": 0, "running_dispatch_runs": 0},
        reconciliation_status="converged",
    )
    assert r["effective_state"] == "suspended"
