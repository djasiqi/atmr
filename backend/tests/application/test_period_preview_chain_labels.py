"""Aperçu période : une chaîne A→B, B→C, C→A n'est pas un A/R d'une course."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.period_invoice_preview import (
    PeriodPreviewLine,
    _chain_leg_labels,
    _round_trip_leg_by_booking_id,
    preview_line_to_dict,
)

_GROUP = "18b0975a-221b-455c-a526-f5574b4122fc"


def _bk(
    bid: int,
    *,
    sequence: int,
    is_return: bool = False,
    parent_booking_id: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=bid,
        route_group_id=_GROUP,
        route_sequence_number=sequence,
        is_return=is_return,
        parent_booking_id=parent_booking_id,
    )


def test_three_leg_route_is_not_flagged_as_round_trip():
    bookings = [
        _bk(46797, sequence=1),
        _bk(46798, sequence=2),
        _bk(46799, sequence=3, is_return=True, parent_booking_id=46798),
    ]
    flags = _round_trip_leg_by_booking_id(bookings)
    assert flags == {46797: False, 46798: False, 46799: False}
    assert _chain_leg_labels(bookings) == {
        46797: "Étape 1",
        46798: "Étape 2",
        46799: "Retour",
    }


def test_single_chain_leg_serializes_without_round_trip_badge():
    line = PeriodPreviewLine(
        booking_id=46798,
        scheduled_at="2026-09-29T23:45:00",
        amount_ht=40.0,
        origin_amount_ht=40.0,
        description="HUG → Joli-Mont",
        source_type="booking",
        is_locked=False,
        already_invoiced=False,
        is_round_trip_leg=True,
        leg_label="Étape 2",
    )
    payload = preview_line_to_dict(line)
    assert payload["unit_type"] == "single"
    assert payload["segments_count"] == 1
    assert payload["is_round_trip_leg"] is False
    assert payload["leg_label"] == "Étape 2"
    assert "round_trip_partner_booking_id" not in payload
