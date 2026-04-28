"""Règles bootstrap session — alignement table de vérité."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from application.auth_bootstrap import access_denied_codes as codes
from application.auth_bootstrap.session_access_rules import evaluate_access_denial
from application.auth_bootstrap.session_bootstrap_snapshot import (
    SessionBootstrapSnapshot,
)
from models.enums import UserRole


def _snap(**kwargs: object) -> SessionBootstrapSnapshot:
    fields = {
        "user_id": 1,
        "public_id": "pub",
        "username": "u",
        "email": "e@example.com",
        "role": UserRole.CLIENT,
        "account_status": None,
        "driver_id": None,
        "driver_company_id": None,
        "driver_is_active": None,
        "company_relation_id": None,
        "client_active_flags": (),
    }
    fields.update(kwargs)
    return SessionBootstrapSnapshot(
        user_id=int(fields["user_id"]),
        public_id=str(fields["public_id"]),
        username=str(fields["username"]),
        email=fields["email"],  # type: ignore[arg-type]
        role=fields["role"],  # type: ignore[arg-type]
        account_status=fields["account_status"],  # type: ignore[arg-type]
        driver_id=fields["driver_id"],  # type: ignore[arg-type]
        driver_company_id=fields["driver_company_id"],  # type: ignore[arg-type]
        driver_is_active=fields["driver_is_active"],  # type: ignore[arg-type]
        company_relation_id=fields["company_relation_id"],  # type: ignore[arg-type]
        client_active_flags=fields["client_active_flags"],  # type: ignore[arg-type]
    )


def test_pending_activation():
    s = _snap(account_status="pending_activation")
    d = evaluate_access_denial(s, MagicMock())
    assert d is not None
    assert d[0] == codes.PENDING_ACTIVATION


def test_driver_inactive():
    s = _snap(
        role=UserRole.DRIVER,
        driver_id=10,
        driver_company_id=1,
        driver_is_active=False,
    )
    d = evaluate_access_denial(s, MagicMock())
    assert d is not None
    assert d[0] == codes.DRIVER_PROFILE_INACTIVE


def test_driver_no_row_allowed_by_legacy():
    s = _snap(role=UserRole.DRIVER, driver_id=None)
    d = evaluate_access_denial(s, MagicMock())
    assert d is None


def test_client_all_inactive():
    s = _snap(role=UserRole.CLIENT, client_active_flags=(False, False))
    d = evaluate_access_denial(s, MagicMock())
    assert d is not None
    assert d[0] == codes.NO_ACTIVE_CLIENT_PROFILE


def test_client_no_clients_skips_client_block():
    s = _snap(role=UserRole.CLIENT, client_active_flags=())
    d = evaluate_access_denial(s, MagicMock())
    assert d is None


def test_institution_invited():
    s = _snap(role=UserRole.INSTITUTION, account_status="invited")
    d = evaluate_access_denial(s, MagicMock())
    assert d is not None
    assert d[0] == codes.INSTITUTION_INVITED


def test_institution_disabled():
    s = _snap(role=UserRole.INSTITUTION, account_status="disabled")
    d = evaluate_access_denial(s, MagicMock())
    assert d is not None
    assert d[0] == codes.INSTITUTION_DISABLED
