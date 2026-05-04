"""Verrou optimiste brouillon : expected_updated_at vs invoice.updated_at."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from application.invoices.edit_draft_invoice import _expected_updated_at_conflict
from models.enums import InvoiceStatus


def _inv(updated_at: datetime | None):
    m = MagicMock()
    m.updated_at = updated_at
    m.status = InvoiceStatus.DRAFT
    return m


def test_expected_updated_at_skipped_when_none():
    inv = _inv(datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC))
    assert _expected_updated_at_conflict(inv, None) is None
    assert _expected_updated_at_conflict(inv, "") is None


def test_expected_updated_at_invalid_format():
    inv = _inv(datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC))
    r = _expected_updated_at_conflict(inv, "not-a-date")
    assert r is not None
    assert r.status_code == 400


@pytest.mark.parametrize(
    "client_ts",
    [
        "2026-04-30T12:00:00+00:00",
        "2026-04-30T12:00:00Z",
    ],
)
def test_expected_updated_at_matches_within_tolerance(client_ts):
    inv = _inv(datetime(2026, 4, 30, 12, 0, 1, tzinfo=UTC))
    assert _expected_updated_at_conflict(inv, client_ts) is None


def test_expected_updated_at_conflict_returns_409():
    inv = _inv(datetime(2026, 4, 30, 14, 0, 0, tzinfo=UTC))
    r = _expected_updated_at_conflict(inv, "2026-04-30T12:00:00+00:00")
    assert r is not None
    assert r.status_code == 409
    assert r.error
    assert r.error.get("error_code") == "INVOICE_CONCURRENT_MODIFICATION"
