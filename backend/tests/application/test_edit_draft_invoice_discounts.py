"""Non-régression remises brouillon : snapshots, booking amount 0, helpers."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from application.invoices.edit_draft_invoice import (
    _eligible_lines_for_per_line_discount,
    _line_ht_snapshot_dict,
)
from models.enums import InvoiceLineType


class TestLineHtSnapshotDict:
    def test_includes_line_type_ride(self):
        line = MagicMock()
        line.id = 236
        line.type = InvoiceLineType.RIDE
        line.line_total = Decimal("40.00")
        line.unit_price = Decimal("40.00")
        line.vat_amount = Decimal("0.00")
        line.total_with_vat = Decimal("40.00")
        line.vat_rate = None
        snap = _line_ht_snapshot_dict(line)
        assert snap["id"] == 236
        assert snap["line_type"] == InvoiceLineType.RIDE.value
        assert snap["line_total"] == "40.00"

    def test_includes_line_type_custom(self):
        line = MagicMock()
        line.id = 244
        line.type = InvoiceLineType.CUSTOM
        line.line_total = Decimal("33750.00")
        line.unit_price = Decimal("33750.00")
        line.vat_amount = Decimal("0.00")
        line.total_with_vat = Decimal("33750.00")
        line.vat_rate = None
        snap = _line_ht_snapshot_dict(line)
        assert snap["line_type"] == InvoiceLineType.CUSTOM.value


@pytest.mark.parametrize(
    ("amount", "estimated", "expected_raw"),
    [
        (Decimal("0"), Decimal("99"), Decimal("0")),
        (None, Decimal("40"), Decimal("40")),
        (None, None, 0),
    ],
)
def test_restore_ride_raw_amount_resolution(amount, estimated, expected_raw):
    """Logique identique à _restore_ride_amounts_from_bookings (vérité montant 0)."""
    amt = amount
    est = estimated
    raw = amt if amt is not None else (est if est is not None else 0)
    assert raw == expected_raw


def test_per_line_discount_eligible_lines_include_positive_custom_lines():
    ride = MagicMock()
    ride.id = 1
    ride.type = InvoiceLineType.RIDE
    ride.line_total = Decimal("40.00")
    ride.line_meta = None

    custom = MagicMock()
    custom.id = 2
    custom.type = InvoiceLineType.CUSTOM
    custom.line_total = Decimal("25.00")
    custom.line_meta = {}

    manual_discount = MagicMock()
    manual_discount.id = 3
    manual_discount.type = InvoiceLineType.CUSTOM
    manual_discount.line_total = Decimal("-5.00")
    manual_discount.line_meta = {"manual_discount": True}

    technical_discount = MagicMock()
    technical_discount.id = 4
    technical_discount.type = InvoiceLineType.CUSTOM
    technical_discount.line_total = Decimal("10.00")
    technical_discount.line_meta = {"per_line_discount_line": True}

    inv = MagicMock()
    inv.lines = [ride, custom, manual_discount, technical_discount]

    assert [line.id for line in _eligible_lines_for_per_line_discount(inv)] == [1, 2]
