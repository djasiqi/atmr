"""Tests d'idempotence / classification erreurs consumer tracking."""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError, OperationalError

from services.tracking.db_error_classification import DbErrorAction, classify_db_error


def test_integrity_error_is_fail_stop_not_dlq():
    """UniqueViolation / IntegrityError → FAIL_STOP (idempotence via ON CONFLICT)."""
    exc = IntegrityError("statement", {}, Exception("duplicate key"))
    assert classify_db_error(exc) == DbErrorAction.FAIL_STOP


def test_operational_error_is_infrastructure_retry():
    exc = OperationalError("statement", {}, Exception("connection reset"))
    assert classify_db_error(exc) == DbErrorAction.INFRASTRUCTURE_RETRY


def test_unknown_error_returns_none():
    assert classify_db_error(RuntimeError("boom")) is None
