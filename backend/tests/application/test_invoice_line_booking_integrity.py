"""Tests BILLING-INTEGRITY-02 : rattachements InvoiceLine ↔ Booking."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from application.invoices.invoice_line_booking_integrity import (
    ERROR_CODE_INVOICE_LINE_LINK_INCOMPLETE,
    InvoiceBookingLinkIncompleteError,
    assert_invoice_booking_link_integrity,
    assert_invoice_lines_booking_link_integrity,
    check_invoice_booking_link_integrity,
    covered_booking_ids,
)
from models.enums import InvoiceLineType, InvoiceStatus


def _line(
    lid: int,
    *,
    reservation_id: int | None,
    meta: dict | None = None,
    line_type: InvoiceLineType = InvoiceLineType.RIDE,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=lid,
        type=line_type,
        reservation_id=reservation_id,
        line_meta=meta,
    )


def _booking(bid: int, invoice_line_id: int | None) -> SimpleNamespace:
    return SimpleNamespace(id=bid, invoice_line_id=invoice_line_id)


def _invoice(
    iid: int,
    lines: list[SimpleNamespace],
    *,
    status: InvoiceStatus = InvoiceStatus.DRAFT,
    sent_at=None,
) -> SimpleNamespace:
    return SimpleNamespace(id=iid, lines=lines, status=status, sent_at=sent_at)


# ---------------------------------------------------------------------------
# G1 / covered_booking_ids
# ---------------------------------------------------------------------------


def test_covered_booking_ids_legacy_merge_partner_rca_case():
    """Cas RCA 37127/37128 : reservation + merge_partner, sans booking_ids."""
    line = _line(
        4648,
        reservation_id=37127,
        meta={
            "is_round_trip_leg": True,
            "round_trip_merge_partner_reservation_id": 37128,
            "amount_overridden": True,
        },
    )
    assert covered_booking_ids(line) == {37127, 37128}


def test_covered_booking_ids_new_booking_ids_format():
    line = _line(
        10,
        reservation_id=101,
        meta={
            "billing_unit": "round_trip",
            "booking_ids": [101, 102],
            "round_trip_secondary_reservation_ids": [102],
            "round_trip_secondary_reservation_id": 102,
        },
    )
    assert covered_booking_ids(line) == {101, 102}


def test_covered_booking_ids_simple_ride():
    line = _line(1, reservation_id=55, meta={"patient_name": "X"})
    assert covered_booking_ids(line) == {55}


# ---------------------------------------------------------------------------
# T1–T6 checker
# ---------------------------------------------------------------------------


def test_t1_legacy_incomplete_fails():
    line = _line(
        4648,
        reservation_id=37127,
        meta={"round_trip_merge_partner_reservation_id": 37128},
    )
    inv = _invoice(1773, [line])
    bookings = {
        37127: _booking(37127, 4648),
        37128: _booking(37128, None),
    }
    result = check_invoice_booking_link_integrity(inv, bookings_by_id=bookings)
    assert result.ok is False
    assert len(result.line_issues) == 1
    issue = result.line_issues[0]
    assert set(issue.covered_booking_ids) == {37127, 37128}
    assert set(issue.null_link_booking_ids) == {37128}
    assert set(issue.linked_correctly) == {37127}


def test_t2_legacy_complete_passes():
    line = _line(
        4648,
        reservation_id=37127,
        meta={"round_trip_merge_partner_reservation_id": 37128},
    )
    inv = _invoice(1773, [line])
    bookings = {
        37127: _booking(37127, 4648),
        37128: _booking(37128, 4648),
    }
    result = check_invoice_booking_link_integrity(inv, bookings_by_id=bookings)
    assert result.ok is True
    assert result.line_issues == ()


def test_t3_new_format_complete_passes():
    line = _line(
        10,
        reservation_id=101,
        meta={"booking_ids": [101, 102], "billing_unit": "round_trip"},
    )
    inv = _invoice(1, [line])
    bookings = {101: _booking(101, 10), 102: _booking(102, 10)}
    assert check_invoice_booking_link_integrity(inv, bookings_by_id=bookings).ok


def test_t4_new_format_incomplete_fails():
    line = _line(
        10,
        reservation_id=101,
        meta={"booking_ids": [101, 102]},
    )
    inv = _invoice(1, [line])
    bookings = {101: _booking(101, 10), 102: _booking(102, None)}
    result = check_invoice_booking_link_integrity(inv, bookings_by_id=bookings)
    assert result.ok is False
    assert set(result.line_issues[0].null_link_booking_ids) == {102}


def test_t5_simple_line_passes():
    line = _line(3, reservation_id=77, meta=None)
    inv = _invoice(1, [line])
    bookings = {77: _booking(77, 3)}
    assert check_invoice_booking_link_integrity(inv, bookings_by_id=bookings).ok


def test_t6_intentional_single_leg_normalized_passes():
    """Après exclusion volontaire : covered={A} seulement, B NULL OK."""
    line = _line(
        10,
        reservation_id=101,
        meta={
            "intentional_single_leg": True,
            "intentional_single_leg_kept": "outbound",
            "released_round_trip_booking_ids": [102],
            "booking_ids": [101],
            "billing_unit": "single",
            "primary_booking_id": 101,
        },
    )
    inv = _invoice(1, [line])
    bookings = {
        101: _booking(101, 10),
        102: _booking(102, None),
    }
    result = check_invoice_booking_link_integrity(inv, bookings_by_id=bookings)
    assert result.ok is True
    assert covered_booking_ids(line) == {101}


# ---------------------------------------------------------------------------
# T7 émission / T8 génération (fail-closed)
# ---------------------------------------------------------------------------


def test_t7_emission_gate_rejects_incomplete_draft():
    line = _line(
        4648,
        reservation_id=37127,
        meta={"round_trip_merge_partner_reservation_id": 37128},
    )
    inv = _invoice(1773, [line], status=InvoiceStatus.DRAFT, sent_at=None)
    bookings = {
        37127: _booking(37127, 4648),
        37128: _booking(37128, None),
    }
    with pytest.raises(InvoiceBookingLinkIncompleteError) as exc_info:
        assert_invoice_booking_link_integrity(inv, bookings_by_id=bookings)
    payload = exc_info.value.to_error_payload()
    assert payload["error_code"] == ERROR_CODE_INVOICE_LINE_LINK_INCOMPLETE
    assert payload["invoice_id"] == 1773
    assert payload["invoice_line_id"] == 4648
    assert set(payload["expected_booking_ids"]) == {37127, 37128}
    assert set(payload["incorrect_booking_ids"]) == {37128}
    # Facture non mutée par le gate
    assert inv.status == InvoiceStatus.DRAFT
    assert inv.sent_at is None


def test_t8_generation_atomic_gate_raises_for_partial_link():
    line = _line(
        10,
        reservation_id=101,
        meta={"booking_ids": [101, 102], "billing_unit": "round_trip"},
    )
    bookings = {101: _booking(101, 10), 102: _booking(102, None)}
    with pytest.raises(InvoiceBookingLinkIncompleteError):
        assert_invoice_lines_booking_link_integrity(
            [line],
            bookings_by_id=bookings,
            invoice_id=99,
        )


def test_rca_case_37127_37128_detected():
    """Fixture de référence BILLING-RCA-RT-01."""
    line = _line(
        4648,
        reservation_id=37127,
        meta={
            "patient_id": 23,
            "patient_name": "MOLLET Emmanuel",
            "service_date": "2026-07-27",
            "amount_overridden": True,
            "is_round_trip_leg": True,
            "patient_client_id": 23,
            "round_trip_merge_partner_reservation_id": 37128,
        },
    )
    inv = _invoice(1773, [line], status=InvoiceStatus.PAID)
    bookings = {
        37127: _booking(37127, 4648),
        37128: _booking(37128, None),
    }
    result = check_invoice_booking_link_integrity(inv, bookings_by_id=bookings)
    assert result.ok is False
    issue = result.line_issues[0]
    assert set(issue.covered_booking_ids) == {37127, 37128}
    assert set(issue.null_link_booking_ids) == {37128}
    assert set(issue.linked_correctly) == {37127}


def test_non_ride_lines_ignored():
    line = _line(
        5,
        reservation_id=None,
        meta=None,
        line_type=InvoiceLineType.CUSTOM,
    )
    inv = _invoice(1, [line])
    assert check_invoice_booking_link_integrity(inv, bookings_by_id={}).ok
