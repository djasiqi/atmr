"""Tests admin_authz — compat vs enforced."""

from __future__ import annotations

from unittest.mock import patch

from services.admin_authz import (
    CAP_BILLING_LOCK,
    CAP_LABS_EXECUTE,
    admin_capabilities_enforced,
    user_effective_admin_capabilities,
    user_has_admin_capability,
)


def test_admin_capabilities_enforced_default_false(monkeypatch):
    monkeypatch.delenv("ADMIN_CAPABILITIES_ENFORCED", raising=False)
    assert admin_capabilities_enforced() is False


def test_admin_capabilities_enforced_true(monkeypatch):
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
    assert admin_capabilities_enforced() is True


def test_compat_allows_even_with_partial_policy(monkeypatch):
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "false")
    type("U", (), {"role": type("R", (), {"__eq__": lambda s, o: True})()})()
    # Simplify: patch role check path
    with (
        patch("services.admin_authz.db.session.get") as get_user,
        patch(
            "services.admin_authz.user_policy_admin_capabilities",
            return_value=frozenset({CAP_BILLING_LOCK}),
        ),
        patch("services.admin_authz.logger") as log,
    ):
        from models.enums import UserRole

        get_user.return_value = type("U", (), {"role": UserRole.ADMIN})()
        assert user_has_admin_capability(1, CAP_LABS_EXECUTE) is True
        assert "admin_capability_would_deny" in log.info.call_args[0][0]


def test_enforced_denies_without_grants(monkeypatch):
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
    from models.enums import UserRole

    with (
        patch("services.admin_authz.db.session.get") as get_user,
        patch(
            "services.admin_authz._admin_capability_grants",
            return_value=frozenset(),
        ),
    ):
        get_user.return_value = type("U", (), {"role": UserRole.ADMIN})()
        assert user_has_admin_capability(1, CAP_LABS_EXECUTE) is False
        assert user_effective_admin_capabilities(1) == frozenset()


def test_enforced_allows_granted_capability(monkeypatch):
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
    from models.enums import UserRole

    with (
        patch("services.admin_authz.db.session.get") as get_user,
        patch(
            "services.admin_authz._admin_capability_grants",
            return_value=frozenset({CAP_BILLING_LOCK}),
        ),
    ):
        get_user.return_value = type("U", (), {"role": UserRole.ADMIN})()
        assert user_has_admin_capability(1, CAP_BILLING_LOCK) is True
        assert user_has_admin_capability(1, CAP_LABS_EXECUTE) is False


def test_enforced_partners_alias_expands_effective(monkeypatch):
    """Grant partners.read ⇒ organizations/accounts présents dans capabilities_effective."""
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
    from models.enums import UserRole
    from services.admin_authz import (
        CAP_ACCOUNTS_READ,
        CAP_ORGANIZATIONS_READ,
        CAP_PARTNERS_READ,
    )

    with (
        patch("services.admin_authz.db.session.get") as get_user,
        patch(
            "services.admin_authz._admin_capability_grants",
            return_value=frozenset({CAP_PARTNERS_READ}),
        ),
    ):
        get_user.return_value = type("U", (), {"role": UserRole.ADMIN})()
        effective = user_effective_admin_capabilities(1)
        assert CAP_PARTNERS_READ in effective
        assert CAP_ORGANIZATIONS_READ in effective
        assert CAP_ACCOUNTS_READ in effective
        assert user_has_admin_capability(1, CAP_ORGANIZATIONS_READ) is True
        assert user_has_admin_capability(1, CAP_ACCOUNTS_READ) is True


def test_none_user_denied():
    assert user_has_admin_capability(None, CAP_LABS_EXECUTE) is False
