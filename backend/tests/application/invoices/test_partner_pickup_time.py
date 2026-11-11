"""Heure de prise en charge partenaire : snapshot et ligne de prestation."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

from application.invoices.partner_invoice_lines import partner_preview_line
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
from services.documents.invoice_pdf_presentation import (
    GRAND_TOTAL_LABEL,
    QR_BILL_UNAVAILABLE_LABEL,
    build_prestation_detail_table,
    client_meta_line_html,
)


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


def test_pickup_mode_does_not_add_a_column():
    off = detail_headers(show_date=True, line_time_mode=LINE_TIME_NONE)
    on = detail_headers(show_date=True, line_time_mode=LINE_TIME_PICKUP)
    assert on == off == ("Date", "Description", "Montant")
    assert "Client :" in client_meta_line_html("FAVRE Anne")
    assert QR_BILL_UNAVAILABLE_LABEL == "QR-Bill non disponible"


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


def test_unconfirmed_or_missing_time_stays_unknown():
    unconfirmed = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 9, 8, 0, 0),
            boarded_at=None,
            time_confirmed=False,
        )
    )
    missing = snapshot_pickup_from_booking(None)
    for snap in (unconfirmed, missing):
        assert snap["pickup_time_source"] == SOURCE_UNKNOWN
        label = pickup_display_label(
            snap["pickup_time_source"],
            snap["scheduled_pickup_at"],
            snap["boarded_at"],
        )
        assert label == "—"


def test_confirmed_local_midnight_is_a_real_pickup():
    planned = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 9, 8, 0, 0),
            boarded_at=None,
            time_confirmed=True,
        )
    )
    assert planned["pickup_time_source"] == SOURCE_SCHEDULED_FALLBACK
    assert planned["scheduled_pickup_at"] == "00:00"
    assert (
        pickup_display_label(
            planned["pickup_time_source"],
            planned["scheduled_pickup_at"],
            planned["boarded_at"],
        )
        == "00:00 (prévue)"
    )
    boarded = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 9, 8, 0, 0),
            boarded_at=datetime(2026, 9, 7, 22, 0, tzinfo=UTC),
            time_confirmed=True,
        )
    )
    assert boarded["boarded_at"] == "00:00"
    assert (
        pickup_display_label(
            boarded["pickup_time_source"],
            boarded["scheduled_pickup_at"],
            boarded["boarded_at"],
        )
        == "00:00"
    )


def test_zurich_offset_follows_the_transport_date():
    summer = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 7, 15, 22, 0, tzinfo=UTC),
            boarded_at=None,
            time_confirmed=True,
        )
    )
    winter = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 11, 15, 23, 0, tzinfo=UTC),
            boarded_at=None,
            time_confirmed=True,
        )
    )
    before_change = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 10, 25, 0, 30, tzinfo=UTC),
            boarded_at=None,
            time_confirmed=True,
        )
    )
    after_change = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 10, 25, 1, 30, tzinfo=UTC),
            boarded_at=None,
            time_confirmed=True,
        )
    )
    assert summer["scheduled_pickup_at"] == "00:00"
    assert winter["scheduled_pickup_at"] == "00:00"
    assert before_change["scheduled_pickup_at"] == "02:30"
    assert after_change["scheduled_pickup_at"] == "02:30"
    october_summer = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 10, 22, 0, 0, tzinfo=UTC),
            boarded_at=None,
            time_confirmed=True,
        )
    )
    november_winter = snapshot_pickup_from_booking(
        SimpleNamespace(
            scheduled_time=datetime(2026, 11, 22, 0, 0, tzinfo=UTC),
            boarded_at=None,
            time_confirmed=True,
        )
    )
    assert october_summer["scheduled_pickup_at"] == "02:00"
    assert november_winter["scheduled_pickup_at"] == "01:00"


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
            "client_name": "Michelle BUSSARD",
            "departure": "A",
            "arrival": "B",
            "pickup": pickup_display_label(
                outbound["pickup_time_source"],
                outbound["scheduled_pickup_at"],
                outbound["boarded_at"],
            ),
            "amount": "40.00",
        },
        {
            "date": "12.09.2026",
            "client_name": "Michelle BUSSARD",
            "departure": "B",
            "arrival": "A",
            "pickup": pickup_display_label(
                inbound["pickup_time_source"],
                inbound["scheduled_pickup_at"],
                inbound["boarded_at"],
            ),
            "amount": "40.00",
        },
    ]
    table = build_prestation_detail_table(
        rows, show_pickup=True, available_width_pt=460
    )
    assert table.repeatRows == 1
    text = _pdf_text([table])
    assert "Client :" in text
    assert "Prise en charge : 08:23" in text or "08:23" in text
    assert "Prise en charge : 11:47" in text or "11:47" in text
    assert "Trajet :" in text
    assert "40.00 CHF" in text
    assert "00:00" not in text
    assert "TOTAL À FACTURER" not in text


def test_partner_pdf_discount_caption_matches_preview():
    table = build_prestation_detail_table(
        [
            {
                "date": "06.10.2026",
                "client_name": "Claire Bonvin",
                "departure": "Chemin des Vignes 4, 1247 Anieres",
                "arrival": "Clinique les Hauts d'Anieres",
                "amount": "64.80",
                "catalog_amount": "72.00",
            },
            {
                "date": "08.10.2026",
                "client_name": "Marc Duval",
                "departure": "Route de Hermance 8",
                "arrival": "Hopitaux Universitaires de Geneve",
                "amount": "95.00",
            },
        ],
        show_pickup=False,
        available_width_pt=460,
    )
    text = " ".join(_pdf_text([table]).split())
    assert "Rabais 10 % · prix initial 72.00 CHF · -7.20 CHF" in text
    assert "64.80" in text
    assert "72.00 →" not in text
    assert text.count("Rabais") == 1


def test_preview_line_uses_institution_prestation_shape():
    booking = SimpleNamespace(
        scheduled_time=datetime(2026, 10, 6, 8, 15),
        boarded_at=datetime(2026, 10, 6, 6, 23, tzinfo=UTC),
        time_confirmed=True,
        customer_name="Anne Favre",
        client=None,
        pickup_location="Chemin des Vignes 4",
        dropoff_location="Hôpital de La Tour",
    )
    transfer = SimpleNamespace(id=1, booking=booking, partner_cost=70)
    line = partner_preview_line(transfer, amount=70)
    assert line["patient_name"] == "Anne Favre"
    assert line["description"].startswith("Trajet :")
    assert "Transfert partenaire" not in line["description"]
    assert line["pickup_label"] == "08:23"
    assert GRAND_TOTAL_LABEL == "TOTAL À FACTURER :"


def test_option_off_hides_pickup_line():
    table = build_prestation_detail_table(
        [
            {
                "date": "05.09.2026",
                "client_name": "Eric DEMIERRE",
                "departure": "A",
                "arrival": "B",
                "pickup": "08:23",
                "amount": "40.00",
            }
        ],
        show_pickup=False,
        available_width_pt=460,
    )
    text = _pdf_text([table])
    assert "Client :" in text
    assert "Trajet :" in text
    assert "Prise en charge" not in text
