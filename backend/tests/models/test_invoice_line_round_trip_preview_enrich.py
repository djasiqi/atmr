"""Enrichissement ``to_dict`` : meta A/R pour fusion dans l'apercu HTML."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from models.enums import InvoiceLineType
from models.invoice import (
    _enrich_invoice_line_payloads_round_trip_merge,
    _enrich_invoice_line_payloads_single_round_trip,
    enrich_invoice_line_payloads_for_api,
)


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
        d1: dict[str, object] = {
            "reservation_id": 101,
            "line_total": 45.0,
            "line_meta": {},
        }
        d2: dict[str, object] = {
            "reservation_id": 102,
            "line_total": 45.0,
            "line_meta": {},
        }

        _enrich_invoice_line_payloads_round_trip_merge([ln1, ln2], [d1, d2])

        assert d1["line_meta"]["is_round_trip_leg"] is True
        assert d1["line_meta"]["transport_type"] == "A/R"
        assert d1["line_meta"]["round_trip_merge_partner_reservation_id"] == 102
        assert d2["line_meta"]["preview_hide_merged_round_trip"] is True
        assert d2["line_meta"]["round_trip_merge_primary_reservation_id"] == 101


def test_enrich_single_round_trip_booking_marks_billing_unit():
    """Réservation ``is_round_trip`` sur une seule ligne facture → billing_unit round_trip."""
    b1 = SimpleNamespace(
        id=501,
        is_return=False,
        is_round_trip=True,
        parent_booking_id=None,
    )
    ln1 = SimpleNamespace(type=InvoiceLineType.RIDE, reservation_id=501)
    d1: dict[str, object] = {
        "reservation_id": 501,
        "line_total": 80.0,
        "line_meta": {},
    }

    _enrich_invoice_line_payloads_single_round_trip(
        [ln1], [d1], bookings_by_id={501: b1}
    )

    assert d1["line_meta"]["billing_unit"] == "round_trip"
    assert d1["line_meta"]["transport_type"] == "A/R"


def test_enrich_for_api_skips_single_when_pair_merge_applies():
    """Paire deux lignes : merge prioritaire, pas de billing_unit mono-ligne."""
    with patch("models.booking.Booking") as _mock_booking_cls:
        day_am = datetime(2026, 3, 15, 10, 0, 0)
        day_pm = datetime(2026, 3, 15, 14, 0, 0)
        foyer = "Foyer test, Route 1, Anieres"
        coll = "Chemin des Ramiers 9, Collonge-Bellerive"
        b1 = SimpleNamespace(
            id=101,
            client_id=42,
            scheduled_time=day_am,
            pickup_location=foyer,
            dropoff_location=coll,
            amount=Decimal("40.00"),
            estimated_amount=None,
            status="COMPLETED",
            parent_booking_id=None,
            is_return=False,
            is_round_trip=True,
        )
        b2 = SimpleNamespace(
            id=102,
            client_id=42,
            scheduled_time=day_pm,
            pickup_location=coll,
            dropoff_location=foyer,
            amount=Decimal("40.00"),
            estimated_amount=None,
            status="COMPLETED",
            parent_booking_id=101,
            is_return=True,
            is_round_trip=False,
        )
        bookings_by_id = {101: b1, 102: b2}
        _mock_booking_cls.query.filter.return_value.all.return_value = [b1, b2]

        ln1 = SimpleNamespace(type=InvoiceLineType.RIDE, reservation_id=101)
        ln2 = SimpleNamespace(type=InvoiceLineType.RIDE, reservation_id=102)
        d1: dict[str, object] = {
            "reservation_id": 101,
            "line_total": 40.0,
            "line_meta": {},
        }
        d2: dict[str, object] = {
            "reservation_id": 102,
            "line_total": 40.0,
            "line_meta": {},
        }

        enrich_invoice_line_payloads_for_api(
            [ln1, ln2], [d1, d2], bookings_by_id=bookings_by_id
        )

        assert d1["line_meta"]["round_trip_merge_partner_reservation_id"] == 102
        assert d2["line_meta"]["preview_hide_merged_round_trip"] is True
        assert d1["line_meta"].get("billing_unit") != "round_trip"


def test_three_leg_route_stays_three_invoice_lines():
    """Étape 1, étape 2 et retour : pas de méta A/R qui masquerait une course au PDF."""
    with patch("models.booking.Booking") as mock_booking_cls:
        group = "18b0975a-221b-455c-a526-f5574b4122fc"
        day = datetime(2026, 9, 29, 22, 50, 0)
        later = datetime(2026, 9, 29, 23, 45, 0)
        pictet = "Avenue Ernest-Pictet 9, Genève"
        hug = "HUG, Genève"
        joli = "Clinique de Joli-Mont, Genève"
        ab = _booking(46797, 42, day, pictet, hug, Decimal("40.00"))
        bc = _booking(46798, 42, later, hug, joli, Decimal("40.00"))
        ca = _booking(46799, 42, later, joli, pictet, Decimal("40.00"))
        ca.parent_booking_id = 46798
        ca.is_return = True
        ab.is_round_trip = True
        for booking in (ab, bc, ca):
            booking.route_group_id = group
        mock_booking_cls.query.filter.return_value.all.return_value = [ab, bc, ca]
        lines = [
            SimpleNamespace(type=InvoiceLineType.RIDE, reservation_id=bid)
            for bid in (46797, 46798, 46799)
        ]
        payloads = [
            {"reservation_id": bid, "line_total": 40.0, "line_meta": {}}
            for bid in (46797, 46798, 46799)
        ]
        enrich_invoice_line_payloads_for_api(
            lines,
            payloads,
            bookings_by_id={46797: ab, 46798: bc, 46799: ca},
        )
        for payload in payloads:
            meta = payload["line_meta"]
            assert meta.get("billing_unit") != "round_trip"
            assert "round_trip_merge_partner_reservation_id" not in meta
            assert meta.get("preview_hide_merged_round_trip") is not True
