"""Tests unitaires — ordre d'évaluation policy (DEC-001)."""

from __future__ import annotations

from unittest.mock import patch

from services.platform_policy import evaluate_policy


def test_explicit_deny_short_circuits():
    r = evaluate_policy(
        action_type="governance.tenant.suspend",
        scope_type="tenant",
        scope_id="1",
        explicit_deny=True,
        is_admin=True,
    )
    assert r["decision"] == "deny"
    assert r["stopped_at"] == "explicit_deny"


def test_scope_mismatch_for_suspend():
    r = evaluate_policy(
        action_type="governance.tenant.suspend",
        scope_type="environment",
        scope_id="prod",
        is_admin=True,
    )
    assert r["decision"] == "deny"
    assert r["stopped_at"] == "scope_check"


def test_allow_suspend_tenant_admin():
    r = evaluate_policy(
        action_type="governance.tenant.suspend",
        scope_type="tenant",
        scope_id="42",
        requested_permission="governance.tenant.suspend",
        is_admin=True,
    )
    assert r["decision"] == "allow"
    assert r["stopped_at"] == "allow"


def test_emergency_override_requires_incident():
    r = evaluate_policy(
        action_type="governance.tenant.suspend",
        scope_type="tenant",
        scope_id="1",
        emergency_override=True,
        incident_id=None,
        is_admin=True,
    )
    assert r["decision"] == "deny"
    assert r["stopped_at"] == "emergency_override"


def test_allow_runbook_execute_admin():
    r = evaluate_policy(
        action_type="governance.runbook.execute",
        scope_type="tenant",
        scope_id="42",
        requested_permission="operate.runbooks.execute",
        is_admin=True,
    )
    assert r["decision"] == "allow"


def test_scope_mismatch_runbook():
    r = evaluate_policy(
        action_type="governance.runbook.execute",
        scope_type="environment",
        scope_id="prod",
        is_admin=True,
    )
    assert r["decision"] == "deny"
    assert r["stopped_at"] == "scope_check"


@patch("services.platform_policy.user_has_platform_permission", return_value=False)
def test_deny_when_user_id_lacks_permission(_mock_perm):
    r = evaluate_policy(
        action_type="governance.tenant.suspend",
        scope_type="tenant",
        scope_id="1",
        requested_permission="governance.tenant.suspend",
        is_admin=True,
        user_id=999,
    )
    assert r["decision"] == "deny"
    assert r["stopped_at"] == "policy_check"
