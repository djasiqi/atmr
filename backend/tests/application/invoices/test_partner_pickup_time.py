"""Heure de prise en charge partenaire : snapshot, colonne, parité de tableau."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

from application.invoices.partner_pickup_time import (
    SOURCE_ACTUAL_BOARDED,
    SOURCE_SCHEDULED_FALLBACK,
    SOURCE_UNKNOWN,
    pickup_display_label,
    snapshot_pickup_from_booking,
)
from services.documents.invoice_pdf_columns import (
    LINE_TIME_NONE,
    LINE_TIME_PICKUP,
    detail_headers,
)
from services.documents.invoice_pdf_presentation import build_services_table


def _pdf_text(flowables) -> str:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    doc.build(flowables)
    raw = buffer.getvalue().decode("latin-1", errors="ignore")
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(buffer.getvalue()))
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        extracted = ""
    return f"{extracted}\n{raw}"


def test_institution_headers_without_pickup_column():
    assert detail_headers(show_date=True, line_time_mode=LINE_TIME_NONE) == (
        "Date",
        "Description",
        "Montant",
    )


def test_pickup_mode_adds_only_the_pickup_column():
    off = detail_headers(show_date=True, line_time_mode=LINE_TIME_NONE)
    on = detail_headers(show_date=True, line_time_mode=LINE_TIME_PICKUP)
    assert on == ("Date", "Prise en charge", "Description", "Montant")
    assert [label for label in on if label not in off] == ["Prise en charge"]


def test_boarded_time_wins_and_converts_to_zurich():
    booking = SimpleNamespace(
        scheduled_time=datetime(2026, 1, 15, 8, 15),
        boarded_at=datetime(2026, 1, 15, 7, 23, tzinfo=UTC),
        time_confirmed=True,
    )
    snap = snapshot_pickup_from_booking(booking)
    assert snap["pickup_time_source"] == SOURCE_ACTUAL_BOARDED
    assert snap["boarded_at"] == "08:23"
    assert snap["scheduled_pickup_at"] == "08:15"
    assert (
        pickup_display_label(
            snap["pickup_time_source"],
            snap["scheduled_pickup_at"],
            snap["boarded_at"],
        )
        == "08:23"
    )


def test_scheduled_fallback_is_marked_as_planned():
    booking = SimpleNamespace(
        scheduled_time=datetime(2026, 9, 8, 13, 30),
        boarded_at=None,
        time_confirmed=True,
    )
    snap = snapshot_pickup_from_booking(booking)
    assert snap["pickup_time_source"] == SOURCE_SCHEDULED_FALLBACK
    label = pickup_display_label(
        snap["pickup_time_source"],
        snap["scheduled_pickup_at"],
        snap["boarded_at"],
    )
    assert label == "13:30 (prévue)"


def test_unknown_or_unconfirmed_or_midnight_never_shows_0000():
    unconfirmed = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 9, 8, 13, 30),
            boarded_at=None,
            time_confirmed=False,
        )
    )
    midnight = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 9, 8, 0, 0),
            boarded_at=datetime(2026, 9, 7, 22, 0, tzinfo=UTC),
            time_confirmed=True,
        )
    )
    missing = snapshot_pickup_from_booking(None)
    for snap in (unconfirmed, midnight, missing):
        assert snap["pickup_time_source"] == SOURCE_UNKNOWN
        label = pickup_display_label(
            snap["pickup_time_source"],
            snap["scheduled_pickup_at"],
            snap["boarded_at"],
        )
        assert label == "—"
        assert "00:00" not in label


def test_snapshot_does_not_follow_a_later_booking_change():
    booking = SimpleNamespace(
        scheduled_time=datetime(2026, 9, 5, 8, 15),
        boarded_at=datetime(2026, 9, 5, 6, 23, tzinfo=UTC),
        time_confirmed=True,
    )
    snap = snapshot_pickup_from_booking(booking)
    booking.scheduled_time = datetime(2026, 9, 5, 18, 0)
    booking.boarded_at = datetime(2026, 9, 5, 16, 0, tzinfo=UTC)
    assert snap["boarded_at"] == "08:23"
    assert snap["scheduled_pickup_at"] == "08:15"


def test_round_trip_keeps_both_pickup_times():
    outbound = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 9, 12, 8, 15),
            boarded_at=datetime(2026, 9, 12, 6, 23, tzinfo=UTC),
            time_confirmed=True,
        )
    )
    inbound = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 9, 12, 11, 30),
            boarded_at=datetime(2026, 9, 12, 9, 47, tzinfo=UTC),
            time_confirmed=True,
        )
    )
    assert outbound["boarded_at"] == "08:23"
    assert inbound["boarded_at"] == "11:47"
    rows = [
        {
            "date": "12.09.2026",
            "pickup": pickup_display_label(
                outbound["pickup_time_source"],
                outbound["scheduled_pickup_at"],
                outbound["boarded_at"],
            ),
            "description": "Michelle BUSSARD — A → B",
            "amount": "40.00",
        },
        {
            "date": "12.09.2026",
            "pickup": pickup_display_label(
                inbound["pickup_time_source"],
                inbound["scheduled_pickup_at"],
                inbound["boarded_at"],
            ),
            "description": "Michelle BUSSARD — B → A",
            "amount": "40.00",
        },
    ]
    table = build_services_table(
        rows, line_time_mode=LINE_TIME_PICKUP, available_width_pt=460
    )
    assert table.repeatRows == 1
    text = _pdf_text([table])
    assert "08:23" in text
    assert "11:47" in text
    assert "A → B" in text or "A" in text
    assert "00:00" not in text


def test_option_off_table_has_no_pickup_header():
    table = build_services_table(
        [
            {
                "date": "05.09.2026",
                "description": "Eric DEMIERRE — A → B",
                "amount": "40.00",
            }
        ],
        line_time_mode=LINE_TIME_NONE,
        available_width_pt=460,
    )
    text = _pdf_text([table])
    assert "Description" in text
    assert "Prise en charge" not in text
