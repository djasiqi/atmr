#!/usr/bin/env python3
"""Génère des PDF de comparaison pour revue visuelle (STOP GATE PDF-UX-01).

Usage (Docker) :
  docker compose exec atmr_api python scripts/generate_mission_pdf_review.py

Les fichiers sont écrits dans /tmp/mission_pdf_review/ et tmp_pdf_review/ux/
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.institutions.mission_report_context import collect_mission_report_context
from services.institutions.mission_report_pdf import (
    VoucherLayoutOptions,
    build_operational_voucher_pdf,
)

OUTPUT_DIR = Path("/tmp/mission_pdf_review")
UX_DIR = OUTPUT_DIR / "ux"

# Décision produit : un seul design cible (operational final) + legacy pour sécurité.
FINAL_OPTIONS = VoucherLayoutOptions(
    hero_style="inline", signature_style="confirmation_inline"
)


def _institution(long_name: bool = False):
    name = (
        "EMS Résidence Médico-Sociale Les Jardins du Léman et de la Rive Gauche"
        if long_name
        else "Clinique Les Hauts d'Anières"
    )
    return SimpleNamespace(
        id=1,
        name=name,
        contact_phone="+41 22 512 02 03",
        contact_email="admin@lha.ch",
        address="Chemin des Courbes 9, 1247 Anières",
    )


def _patient_osmani():
    return SimpleNamespace(
        first_name="Mirjete",
        last_name="OSMANI",
        dob=date(1997, 10, 4),
        external_reference="DPI-996",
        floor=3,
        residence_name="Unité Cardiologie",
        address="Chemin des Courbes 9",
        postal_code="1247",
        city="Anières",
    )


def _patient(long_name: bool = False):
    last = "JASIQI" + (" VON VERYLONGNAME" * 4 if long_name else "")
    return SimpleNamespace(
        first_name="Drin",
        last_name=last,
        dob=date(1993, 7, 24),
        external_reference="DPI-99",
        floor=3,
        residence_name="Unité Cardiologie"
        if not long_name
        else "Service de Cardiologie Interventionnelle et de Réadaptation",
        address="Chemin des Courbes 9",
        postal_code="1247",
        city="Anières",
    )


def _booking(status="COMPLETED", **kwargs):
    defaults = {
        "id": 31002,
        "status": status,
        "amount": 40.0,
        "boarded_at": datetime(2026, 6, 13, 11, 34, tzinfo=UTC),
        "completed_at": datetime(2026, 6, 13, 14, 7, tzinfo=UTC),
        "billed_to_type": "clinic",
        "invoice_line_id": None,
        "driver_id": None,
    }
    defaults.update(kwargs)
    b = SimpleNamespace(**defaults)
    b._get_route_journey = lambda: [
        {"type": "pickup", "date": "2026-06-13T11:34:00Z"},
        {"type": "dropoff", "date": "2026-06-13T14:07:00Z"},
    ]
    return b


def _tr(**kwargs):
    long_addr = kwargs.pop("long_address", False)
    long_name = kwargs.pop("long_name", False)
    long_inst = kwargs.pop("long_institution", False)
    defaults = {
        "id": 976,
        "public_id": "98ddbfc9-review-test",
        "institution_id": 1,
        "booking_id": kwargs.get("booking", _booking()).id
        if kwargs.get("booking")
        else None,
        "created_at": datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
        "accepted_at": datetime(2026, 6, 12, 21, 18, tzinfo=UTC),
        "mission_date": date(2026, 6, 13),
        "mission_type": "patient_transport",
        "billing_intent": "institution",
        "status": "CONVERTED",
        "pickup_location": "Chemin des Courbes 9, 1247 Anières"
        + (" " + "x" * 180 if long_addr else ""),
        "dropoff_location": "Centre d'Imagerie Rive Gauche, Route de Thonon 61, 1222 Vésenaz"
        if long_inst
        else "Clinique Générale Beaulieu",
        "is_round_trip": False,
        "multi_stop": False,
        "return_to_institution": False,
        "pickup_time_confirmed": True,
        "scheduled_time": datetime(2026, 6, 13, 11, 30),
        "return_time_confirmed": False,
        "return_time": None,
        "return_date": None,
        "legs": [],
        "external_reference": None,
        "contact_on_site": {
            "requester_service": "Admissions",
            "requester_name": "Marc Mouchet",
        },
        "notes": kwargs.pop("notes", "Note courte"),
        "floor_elevator_info": "3e étage, ascenseur",
        "mobility": {"wheelchair": True},
        "carrier_source": "lirie",
        "external_carrier_name": None,
        "external_carrier_phone": None,
        "external_carrier_reference": None,
        "external_carrier_reason": None,
        "assigned_externally_at": None,
        "executed_externally_at": None,
        "external_execution_notes": None,
        "externalized_by": None,
        "executed_externally_by": None,
    }
    defaults.update(kwargs)
    tr = SimpleNamespace(**defaults)
    tr.institution = _institution(long_name=long_inst)
    tr.patient = (
        kwargs.get("patient") if "patient" in kwargs else _patient(long_name=long_name)
    )
    tr.get_mobility = lambda: defaults.get("mobility") or {}
    tr._get_creator_name = lambda: "Marc Mouchet"
    tr._serialize_booking_summary = lambda: {
        "is_invoiced": False,
        "is_cancellation_billable": False,
    }
    if "accepted_by_company" in kwargs:
        tr.accepted_by_company = kwargs["accepted_by_company"]
    else:
        tr.accepted_by_company = SimpleNamespace(
            name="Emmenez Moi",
            contact_phone="022 512 02 03",
            contact_email="khalid.alaoui@outlook.com",
        )
    return tr


def _pdf_page_count(pdf_bytes: bytes) -> int | None:
    try:
        from pypdf import PdfReader

        return len(PdfReader(BytesIO(pdf_bytes)).pages)
    except Exception:
        return None


def _write(name: str, pdf_bytes: bytes, *, ux: bool = False) -> None:
    base = UX_DIR if ux else OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    path.write_bytes(pdf_bytes)
    pages = _pdf_page_count(pdf_bytes)
    page_info = f", {pages} page(s)" if pages is not None else ""
    print(f"  ✓ {path} ({len(pdf_bytes)} octets{page_info})")


def _write_final(prefix: str, ctx, *, with_legacy: bool = True) -> None:
    """Écrit le bon final (operational) + legacy pour comparaison/sécurité."""
    _write(
        f"{prefix}_operational_final.pdf",
        build_operational_voucher_pdf(ctx, layout="operational", options=FINAL_OPTIONS),
        ux=True,
    )
    if with_legacy:
        _write(
            f"{prefix}_legacy.pdf",
            build_operational_voucher_pdf(ctx, layout="legacy"),
            ux=True,
        )


def main() -> None:
    from app import create_app

    app = create_app()
    with app.app_context():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        UX_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Génération PDF revue visuelle → {OUTPUT_DIR}")
        print(f"Maquettes UX → {UX_DIR}")
        _generate_all()
        print("Terminé.")


def _generate_all() -> None:
    """Génère les bons finaux (operational) pour les scénarios obligatoires."""
    inst = _institution()

    # 01 — Mission simple (aller simple)
    tr = _tr(booking=_booking(), booking_id=31002)
    ctx = collect_mission_report_context(tr, inst, variant="operational")
    _write_final("01_simple", ctx)

    # 03 — Aller-retour institution
    tr = _tr(
        is_round_trip=True,
        return_to_institution=True,
        return_time_confirmed=True,
        return_time=datetime(2026, 6, 13, 15, 0),
        return_date=date(2026, 6, 13),
        booking=_booking(status="RETURN_COMPLETED"),
        booking_id=31002,
    )
    ctx = collect_mission_report_context(tr, inst, variant="operational")
    _write_final("03_roundtrip", ctx)

    # 05 — Contenu long (noms / adresses / notes très longs)
    long_notes = "\n\n".join(
        [f"Paragraphe médical {i} " + ("détail clinique " * 40) for i in range(5)]
    )
    tr = _tr(
        booking=_booking(),
        booking_id=31002,
        long_name=True,
        long_address=True,
        long_institution=True,
        notes=long_notes,
    )
    ctx = collect_mission_report_context(tr, inst, variant="operational")
    _write_final("05_longcontent", ctx)

    # 08 — Scénario critique chauffeur : imaging A-R, 09:00 / 23:00, besoins
    tr_crit = _tr(
        id=996,
        public_id="driver-critique-996",
        patient=_patient_osmani(),
        booking=_booking(),
        booking_id=31002,
        is_round_trip=True,
        return_to_institution=True,
        return_time_confirmed=True,
        return_time=datetime(2026, 6, 13, 23, 0),
        return_date=date(2026, 6, 13),
        scheduled_time=datetime(2026, 6, 13, 9, 0),
        pickup_time_confirmed=True,
        dropoff_location="Centre d'Imagerie Rive Gauche, Route de Thonon 61, 1222 Vésenaz",
        notes="Transport couchée — prévoir accompagnement",
        mobility={"assisted": True, "needs_assistance": True},
        contact_on_site={
            "requester_service": "Admissions",
            "requester_name": "Marc Mouchet",
            "requester_phone": "+41 22 512 02 03",
        },
    )
    ctx_crit = collect_mission_report_context(tr_crit, inst, variant="operational")
    _write_final("08_driver_critique", ctx_crit)

    # 09 — Multi-étapes 5 arrêts (lisibilité parcours chauffeur, aucun « Étape N »)
    legs = [
        SimpleNamespace(
            sequence_index=0,
            pickup_location="Chemin des Courbes 9, 1247 Anières",
            dropoff_location="HUG",
            dropoff_establishment="HUG",
            scheduled_time=datetime(2026, 6, 13, 10, 0),
            time_confirmed=True,
        ),
        SimpleNamespace(
            sequence_index=1,
            pickup_location="HUG",
            dropoff_location="Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
            dropoff_establishment="Centre d'Imagerie Rive Gauche",
            scheduled_time=datetime(2026, 6, 13, 12, 0),
            time_confirmed=True,
        ),
        SimpleNamespace(
            sequence_index=2,
            pickup_location="Centre d'Imagerie Rive Gauche",
            dropoff_location="Route de Thonon 61, 1222 Vésenaz",
            dropoff_establishment="Clinique Générale-Beaulieu",
            scheduled_time=datetime(2026, 6, 13, 14, 0),
            time_confirmed=True,
        ),
        SimpleNamespace(
            sequence_index=3,
            pickup_location="Clinique Générale-Beaulieu",
            dropoff_location="Chemin Beau-Soleil 20, 1206 Genève",
            dropoff_establishment="Laboratoire Unilabs",
            scheduled_time=datetime(2026, 6, 13, 16, 0),
            time_confirmed=True,
        ),
    ]
    booking5 = _booking()
    booking5._get_route_journey = lambda: [
        {"type": "pickup", "date": "2026-06-13T10:00:00Z"},
        {"type": "dropoff", "date": "2026-06-13T11:30:00Z"},
        {"type": "dropoff", "date": "2026-06-13T13:30:00Z"},
        {"type": "dropoff", "date": "2026-06-13T15:30:00Z"},
        {"type": "dropoff", "date": "2026-06-13T17:30:00Z"},
    ]
    tr5 = _tr(
        id=997,
        public_id="multistep-5-997",
        patient=_patient_osmani(),
        multi_stop=True,
        is_round_trip=True,
        return_to_institution=True,
        legs=legs,
        booking=booking5,
        booking_id=31002,
        scheduled_time=datetime(2026, 6, 13, 9, 0),
        pickup_time_confirmed=True,
    )
    ctx5 = collect_mission_report_context(tr5, inst, variant="operational")
    _write_final("09_multistep_5", ctx5, with_legacy=False)

    print(
        "  → Bon final (operational) : 01_simple / 03_roundtrip / 05_longcontent / "
        "08_driver_critique / 09_multistep_5 (_operational_final.pdf)"
    )


if __name__ == "__main__":
    main()
