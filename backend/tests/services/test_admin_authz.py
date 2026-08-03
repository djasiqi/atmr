"""Tests PR2bis — admin_authz (flag + fallback legacy)."""

from __future__ import annotations

from unittest.mock import patch

from services.admin_authz import (
    CAP_BILLING_LOCK,
    CAP_LABS_EXECUTE,
    admin_capabilities_enforced,
    user_has_admin_capability,
)


def test_admin_capabilities_enforced_default_false(monkeypatch):
    monkeypatch.delenv("ADMIN_CAPABILITIES_ENFORCED", raising=False)
    assert admin_capabilities_enforced() is False


def test_admin_capabilities_enforced_true(monkeypatch):
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
    assert admin_capabilities_enforced() is True


def test_would_deny_allows_when_not_enforced(monkeypatch):
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "false")
    with (
        patch(
            "services.admin_authz.user_effective_admin_capabilities",
            return_value=frozenset(),
        ),
        patch("services.admin_authz.logger") as log,
    ):
        assert user_has_admin_capability(1, CAP_LABS_EXECUTE) is True
        log.info.assert_called()
        assert "admin_capability_would_deny" in log.info.call_args[0][0]


def test_denies_when_enforced_and_missing(monkeypatch):
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
    with patch(
        "services.admin_authz.user_effective_admin_capabilities",
        return_value=frozenset({CAP_BILLING_LOCK}),
    ):
        assert user_has_admin_capability(1, CAP_LABS_EXECUTE) is False
        assert user_has_admin_capability(1, CAP_BILLING_LOCK) is True


def test_none_user_denied():
    assert user_has_admin_capability(None, CAP_LABS_EXECUTE) is False
