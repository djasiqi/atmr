"""Tests sans DB — effective_state / reconciliation (DEC-002)."""

from __future__ import annotations

from services.platform_tenant_governance import compute_reconciliation_and_effective


def test_active_tenant_converged():
    r, e, b = compute_reconciliation_and_effective(
        desired_suspended=False,
        observed={"active_bookings_count": 5, "running_dispatch_runs": 1},
    )
    assert r == "converged"
    assert e == "active"
    assert b == []


def test_suspended_with_drift():
    r, e, b = compute_reconciliation_and_effective(
        desired_suspended=True,
        observed={"active_bookings_count": 2, "running_dispatch_runs": 0},
    )
    assert r == "drift"
    assert e == "suspended_with_residual_activity"
    assert "drift" in b


def test_suspended_converged_no_activity():
    r, e, b = compute_reconciliation_and_effective(
        desired_suspended=True,
        observed={"active_bookings_count": 0, "running_dispatch_runs": 0},
    )
    assert r == "converged"
    assert e == "suspended"
    assert b == []
