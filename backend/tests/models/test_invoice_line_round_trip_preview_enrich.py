"""Enrichissement ``to_dict`` : meta A/R pour fusion dans l'apercu HTML."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from models.enums import InvoiceLineType
from models.invoice import _enrich_invoice_line_payloads_round_trip_merge


def _booking(
    bid: int,
    cid: int | None,
    t1: datetime,
    pu: str,
    do: str,
    amount: Decimal,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=bid,
        client_id=cid,
        scheduled_time=t1,
        pickup_location=pu,
        dropoff_location=do,
        amount=amount,
        estimated_amount=None,
        status="COMPLETED",
        parent_booking_id=None,
        is_return=False,
    )


def test_enrich_sets_primary_ar_and_hides_secondary():
    """Hub + montants ligne identiques : primaire [A/R], secondaire masque apercu."""
    with patch("models.booking.Booking") as _mock_booking_cls:
        day_am = datetime(2026, 3, 15, 10, 0, 0)
        day_pm = datetime(2026, 3, 15, 14, 0, 0)
        foyer = "Foyer test, Route 1, Anieres"
        coll = "Chemin des Ramiers 9, Collonge-Bellerive"
        clin = "Clinique test, Chemin 9, Anieres"
        b1 = _booking(101, 42, day_am, foyer, coll, Decimal("12.00"))
        b2 = _booking(102, 42, day_pm, clin, foyer, Decimal("78.00"))
        _mock_booking_cls.query.filter.return_value.all.return_value = [b1, b2]

        ln1 = SimpleNamespace(type=InvoiceLineType.RIDE, reservation_id=101)
        ln2 = SimpleNamespace(type=InvoiceLineType.RIDE, reservation_id=102)
        d1: dict[str, object] = {"reservation_id": 101, "line_total": 45.0, "line_meta": {}}
        d2: dict[str, object] = {"reservation_id": 102, "line_total": 45.0, "line_meta": {}}

        _enrich_invoice_line_payloads_round_trip_merge([ln1, ln2], [d1, d2])

        assert d1["line_meta"]["is_round_trip_leg"] is True
        assert d1["line_meta"]["transport_type"] == "A/R"
        assert d1["line_meta"]["round_trip_merge_partner_reservation_id"] == 102
        assert d2["line_meta"]["preview_hide_merged_round_trip"] is True
        assert d2["line_meta"]["round_trip_merge_primary_reservation_id"] == 101
