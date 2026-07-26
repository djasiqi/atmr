"""Exclusion d'une jambe A/R sur brouillon facture (deux lignes ou fusion unique)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from application.invoices.edit_draft_invoice import (
    _resolve_round_trip_leg_line_to_delete,
    remove_draft_invoice_line,
)
from models.enums import InvoiceLineType


def _line(lid: int, rid: int, meta: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=lid,
        type=InvoiceLineType.RIDE,
        reservation_id=rid,
        line_meta=meta or {},
        line_total=Decimal("80.00"),
        description="Trajet test",
        qty=Decimal("1"),
        unit_price=Decimal("80.00"),
        vat_rate=None,
        vat_amount=Decimal("0"),
        total_with_vat=Decimal("80.00"),
        adjustment_note=None,
    )


def _inv(*lines: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(id=1, lines=list(lines))


def test_resolve_exclude_return_on_primary():
    inv = _inv(
        _line(
            1,
            101,
            {
                "round_trip_merge_partner_reservation_id": 102,
                "is_round_trip_leg": True,
            },
        ),
        _line(2, 102, {"preview_hide_merged_round_trip": True}),
    )
    primary = inv.lines[0]
    target = _resolve_round_trip_leg_line_to_delete(inv, primary, exclude_leg="return")
    assert target is not None
    assert target.id == 2


def test_resolve_exclude_outbound_on_return_line():
    inv = _inv(
        _line(
            1,
            101,
            {
                "round_trip_merge_partner_reservation_id": 102,
            },
        ),
        _line(
            2,
            102,
            {
                "preview_hide_merged_round_trip": True,
                "round_trip_merge_primary_reservation_id": 101,
            },
        ),
    )
    secondary = inv.lines[1]
    target = _resolve_round_trip_leg_line_to_delete(
        inv, secondary, exclude_leg="outbound"
    )
    assert target is not None
    assert target.id == 1


@patch("application.invoices.edit_draft_invoice._recompute_totals_from_lines")
@patch("application.invoices.edit_draft_invoice._mark_pdf_stale")
@patch("application.invoices.edit_draft_invoice.db")
@patch("application.invoices.edit_draft_invoice._resolve_draft_invoice")
@patch("application.invoices.edit_draft_invoice.Booking")
def test_remove_exclude_return_keeps_primary(
    mock_booking_cls,
    mock_resolve,
    mock_db,
    _mark,
    _recompute,
):
    primary = _line(
        10,
        201,
        {"round_trip_merge_partner_reservation_id": 202, "is_round_trip_leg": True},
    )
    secondary = _line(
        11,
        202,
        {
            "preview_hide_merged_round_trip": True,
            "round_trip_merge_primary_reservation_id": 201,
        },
    )
    inv = _inv(primary, secondary)
    mock_resolve.return_value = (inv, None, None)
    mock_booking_cls.query.get.return_value = SimpleNamespace(
        invoice_line_id=11, updated_at=None
    )

    r = remove_draft_invoice_line(1, 1, 10, exclude_round_trip_leg="return")

    assert r.success is True
    mock_db.session.delete.assert_called_once_with(secondary)
    pri_meta = primary.line_meta or {}
    assert pri_meta.get("round_trip_merge_partner_reservation_id") is None
