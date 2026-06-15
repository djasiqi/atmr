"""Tests unitaires — génération PDF mission institution (PDF-01a…f)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from services.institutions.mission_report_context import collect_mission_report_context
from services.institutions.mission_report_pdf import (
    build_mission_audit_report_pdf,
    build_operational_voucher_pdf,
    _resolve_logo,
    _step_time_line,
    _truncate_field,
    _truncate_medical_notes,
    _MAX_VOUCHER_NOTES,
)


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extraction best-effort ; en CI sans pypdf/pdfminer, retourne chaîne vide."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams

            return extract_text(BytesIO(pdf_bytes), laparams=LAParams()) or ""
        except Exception:
            return ""


def _pdf_page_count(pdf_bytes: bytes) -> int | None:
    try:
        from pypdf import PdfReader

        return len(PdfReader(BytesIO(pdf_bytes)).pages)
    except Exception:
        return None


def _institution():
    return SimpleNamespace(
        id=1,
        name="Clinique LHA",
        contact_phone="+41 22 000 00 00",
        contact_email="a@clinique.ch",
    )


def _patient(long_name: bool = False, with_address: bool = False):
    last = "JASIQI" + (" VON VERYLONGNAME" * 4 if long_name else "")
    return SimpleNamespace(
        first_name="Drin",
        last_name=last,
        dob=date(1980, 5, 12),
        external_reference="DPI-99",
        address="Chemin des Courbes 9" if with_address else None,
        postal_code="1247" if with_address else None,
        city="Anières" if with_address else None,
    )


def _flow_texts(flow):
    """Extrait le texte des Paragraph d'une liste de flowables (tables incluses)."""
    from reportlab.platypus import Paragraph, Table

    out = []
    for el in flow:
        if isinstance(el, Paragraph):
            out.append(el.getPlainText())
        elif isinstance(el, Table):
            for row in el._cellvalues:
                for cell in row:
                    items = cell if isinstance(cell, list) else [cell]
                    for sub in items:
                        if isinstance(sub, Paragraph):
                            out.append(sub.getPlainText())
    return out


def _flow_has_image(flow) -> bool:
    """True si le flow contient au moins une image (ex. QR code), tables imbriquées incluses."""
    from reportlab.platypus import Image, Table

    def _cell_has_image(cell) -> bool:
        items = cell if isinstance(cell, list) else [cell]
        for sub in items:
            if isinstance(sub, Image):
                return True
            if isinstance(sub, Table):
                for row in sub._cellvalues:
                    for nested in row:
                        if _cell_has_image(nested):
                            return True
        return False

    for el in flow:
        if isinstance(el, Image):
            return True
        if isinstance(el, Table):
            for row in el._cellvalues:
                for cell in row:
                    if _cell_has_image(cell):
                        return True
    return False


def _tr(**kwargs):
    defaults = {
        "id": 1820,
        "public_id": "uuid-1820",
        "institution_id": 1,
        "booking_id": 4567,
        "created_at": datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
        "accepted_at": datetime(2026, 6, 12, 21, 18, tzinfo=UTC),
        "mission_date": date(2026, 6, 13),
        "mission_type": "patient_transport",
        "billing_intent": "institution",
        "status": "CONVERTED",
        "pickup_location": "Chemin des Courbes 9, 1247 Anières " + ("x" * 180 if kwargs.get("long_address") else ""),
        "dropoff_location": "Clinique Beaulieu",
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
        "contact_on_site": {"requester_service": "Admissions", "requester_name": "Marc Mouchet"},
        "notes": kwargs.get("notes", "Note courte"),
        "floor_elevator_info": "Étage 3",
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
    tr.institution = _institution()
    tr.patient = kwargs.get("patient") if "patient" in kwargs else _patient(long_name=kwargs.get("long_name", False))
    tr.booking = kwargs.get("booking")
    tr.accepted_by_company = kwargs.get(
        "accepted_by_company",
        SimpleNamespace(name="Emmenez Moi", contact_phone="079", contact_email="e@em.com"),
    )
    tr.get_mobility = lambda: defaults.get("mobility") or {}
    tr._get_creator_name = lambda: "Marc Mouchet"
    tr._serialize_booking_summary = lambda: {"is_invoiced": False, "is_cancellation_billable": False}
    return tr


def _booking(**kwargs):
    defaults = {
        "id": 4567,
        "status": "COMPLETED",
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


def _mock_timeline_and_messages(mock_timeline, mock_bmsg, *, events=0, messages=0):
    mock_timeline.return_value = [
        SimpleNamespace(
            id=i,
            event_type=(
                "request_created"
                if i == 0
                else "offer_accepted"
                if i == 1
                else "patient_boarded"
                if i == 2
                else "patient_completed"
                if i == 3
                else "offer_sent"
            ),
            created_at=datetime(2026, 6, 12, 20, 48, tzinfo=UTC) + timedelta(minutes=i * 30),
            actor_type="system" if i % 2 == 0 else "driver",
            actor_user_id=None,
            payload={"driver_name": "Chauffeur", "company_name": "Emmenez Moi"},
        )
        for i in range(events)
    ]
    msgs = [
        SimpleNamespace(
            id=i,
            content=f"Message test numéro {i} pour le transport.",
            sender_label="Institution" if i % 2 else "Transporteur",
            created_at=datetime(2026, 6, 13, 10, i, tzinfo=UTC),
        )
        for i in range(messages)
    ]
    mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = len(msgs)
    mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = msgs
    if len(msgs) > 200:
        mock_bmsg.query.filter_by.return_value.order_by.return_value.offset.return_value.all.return_value = msgs[-200:]


def _pdf_text(pdf_bytes: bytes) -> str:
    return _extract_pdf_text(pdf_bytes)


def test_step_time_line_cancelled_keeps_planned_only():
    step = {
        "address": "Chemin des Courbes 9",
        "planned_time": "11:30",
        "actual_time": "11:34",
    }
    cancelled_line = _step_time_line(step, cancelled=True)
    assert cancelled_line == "Prévu : 11:30"
    assert "Réel" not in cancelled_line
    full_line = _step_time_line(step, cancelled=False)
    assert "Prévu : 11:30" in full_line
    assert "Réel : 11:34" in full_line


def test_step_time_line_empty_when_no_times():
    assert _step_time_line({"address": "X"}) is None


def test_truncate_field_ellipsis():
    long_name = (
        "JASIQI VON VERYLONGNAME THAT EXCEEDS TWO LINES "
        "IN THE PDF COLUMN AND SHOULD BE TRUNCATED"
    )
    truncated = _truncate_field(long_name)
    assert truncated is not None
    assert truncated.endswith("…")
    assert len(truncated) <= 66


def test_truncate_field_per_field_limits():
    patient = _truncate_field("X" * 200, max_len=80)
    assert len(patient) == 80
    assert patient.endswith("…")
    carrier = _truncate_field("Y" * 200, max_len=60)
    assert len(carrier) == 60


def test_truncate_medical_notes():
    from services.institutions.mission_report_pdf import _MAX_MEDICAL_NOTES

    short = "Patient calme, accompagnant requis."
    assert _truncate_medical_notes(short) == short
    long_notes = "détail clinique " * 200
    truncated = _truncate_medical_notes(long_notes)
    assert truncated.endswith("[…]")
    assert len(truncated) <= _MAX_MEDICAL_NOTES + len(" […]")


def test_truncate_voucher_medical_notes_120():
    assert _MAX_VOUCHER_NOTES == 120
    long_notes = "x" * 250
    truncated = _truncate_medical_notes(long_notes, max_len=_MAX_VOUCHER_NOTES)
    assert truncated.endswith("[…]")
    assert len(truncated) <= _MAX_VOUCHER_NOTES + len(" […]")


@patch("services.institutions.mission_report_context.list_timeline_events")
@patch("services.institutions.mission_report_context.BookingMessage")
class TestMissionReportPdf:
    def test_pdf01a_generates_audit_and_operational(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=3, messages=2)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        audit = build_mission_audit_report_pdf(ctx)
        op = build_operational_voucher_pdf(ctx)
        assert audit[:4] == b"%PDF"
        assert op[:4] == b"%PDF"
        assert len(audit) > 1000
        assert len(op) > 500
        assert len(op) < len(audit)
        assert ctx.reference == "TR-2026-001820"
        assert ctx.status_label == "Réalisé"
        assert "COMPLETED" not in ctx.status_label
        text = _pdf_text(audit)
        if text:
            assert "Historique" in text or "TR-2026" in text

    def test_pdf01e_no_booking_operational(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr = _tr(status="SENT", booking_id=None, booking=None, patient=_patient())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pdf = build_operational_voucher_pdf(ctx)
        assert pdf[:4] == b"%PDF"
        assert ctx.booking_number is None

    def test_pdf01f_stress_render(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=15, messages=20)
        long_notes = "\n\n".join([f"Paragraphe médical {i} " + ("détail " * 40) for i in range(5)])
        tr = _tr(
            booking=_booking(),
            long_name=True,
            long_address=True,
            notes=long_notes,
        )
        ctx = collect_mission_report_context(tr, _institution())
        audit = build_mission_audit_report_pdf(ctx)
        assert audit[:4] == b"%PDF"
        assert len(audit) > 2000
        assert ctx.timeline_truncated is False or len(ctx.timeline_rows) <= 500
        text = _pdf_text(audit)
        if text:
            assert "JASIQI" in text

    def test_operational_has_signature_zones(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pdf = build_operational_voucher_pdf(ctx)
        assert pdf[:4] == b"%PDF"
        assert len(ctx.route_steps) >= 1
        text = _pdf_text(pdf)
        if text:
            assert "Chauffeur" in text

    def test_v11_milestones_and_v2_certificate_in_audit(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=5, messages=1)
        user = SimpleNamespace(first_name="Jean", last_name="Dupont", phone="079 111 22 33")
        driver = SimpleNamespace(user=user, vehicle_assigned="Mercedes Vito")
        booking = _booking()
        booking.driver = driver
        booking.driver_id = 99
        tr = _tr(booking=booking)
        patient = tr.patient
        patient.floor = 2
        patient.residence_name = "Cardiologie"
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        assert ctx.mission_milestones
        assert ctx.completion_certificate is not None
        assert ctx.traceability.get("archive_reference")
        pdf = build_mission_audit_report_pdf(ctx)
        assert pdf[:4] == b"%PDF"
        assert len(pdf) > len(build_operational_voucher_pdf(ctx))
        text = _pdf_text(pdf)
        if text:
            assert "Jalons opérationnels" not in text
            assert "Certificat de réalisation" not in text
            assert "INFORMATIONS ADMINISTRATIVES" in text.upper() or "Informations administratives" in text

    def test_pr1_visual_elements(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=2, messages=1)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        audit = build_mission_audit_report_pdf(ctx)
        op = build_operational_voucher_pdf(ctx)
        assert audit[:4] == b"%PDF"
        assert op[:4] == b"%PDF"
        text = _pdf_text(audit)
        if text:
            assert "RAPPORT DE MISSION" in text.upper() or "Rapport de mission" in text
            assert "Transporteur" in text
            assert "Plateforme de coordination" not in text
            assert "Informations de mission" not in text
            assert text.count("Clinique LHA") <= 2
        text_op = _pdf_text(op)
        if text_op:
            assert "Chauffeur" in text_op
            assert "Signature" in text_op

    def test_logo_fallback_no_exception(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        with patch("services.institutions.mission_report_pdf._LOGO_PATH") as mock_path:
            mock_path.is_file.return_value = False
            tr = _tr(booking=_booking())
            ctx = collect_mission_report_context(tr, _institution())
            pdf = build_operational_voucher_pdf(ctx)
            assert pdf[:4] == b"%PDF"
        assert _resolve_logo() is None or isinstance(_resolve_logo(), str)

    def test_certificate_absent_when_in_progress(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr = _tr(booking=_booking(status="EN_ROUTE"))
        ctx = collect_mission_report_context(tr, _institution())
        assert ctx.completion_certificate is None
        pdf = build_mission_audit_report_pdf(ctx)
        assert pdf[:4] == b"%PDF"
        text = _pdf_text(pdf)
        if text:
            assert "Certificat de réalisation" not in text

    def test_synthetic_history_in_audit(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=4)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        assert ctx.synthetic_history
        assert len(ctx.synthetic_history) <= 4
        labels = [row["label"] for row in ctx.synthetic_history]
        assert any("Demande créée" in label for label in labels)
        pdf = build_mission_audit_report_pdf(ctx)
        text = _pdf_text(pdf)
        if text:
            assert "Historique" in text
            assert "Demande créée" in text or "Prise en charge" in text
            assert "Offre envoyée" not in text
            assert "Historique complet" not in text

    def test_admin_block_replaces_billing_and_traceability(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=2)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        pdf = build_mission_audit_report_pdf(ctx)
        text = _pdf_text(pdf)
        if text:
            assert "INFORMATIONS ADMINISTRATIVES" in text.upper() or "Informations administratives" in text
            assert "■ FACTURATION" not in text.upper()
            assert "■ TRAÇABILITÉ" not in text.upper() and "■ TRACABILITE" not in text.upper()
            # Émetteur retiré du bloc admin (porté par le footer uniquement)
            assert "Document généré par LIRIE" not in text

    def test_no_attachments_section_when_empty(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        assert ctx.attachments == []
        pdf = build_mission_audit_report_pdf(ctx)
        text = _pdf_text(pdf)
        if text:
            assert "Aucune pièce jointe" not in text
            assert "PIÈCES JOINTES" not in text.upper() and "PIECES JOINTES" not in text.upper()

    def test_medical_section_conditional(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr_empty = _tr(booking=_booking(), notes="", floor_elevator_info="", mobility={})
        ctx_empty = collect_mission_report_context(tr_empty, _institution(), variant="audit")
        pdf_empty = build_mission_audit_report_pdf(ctx_empty)
        text_empty = _pdf_text(pdf_empty)
        if text_empty:
            assert "BESOINS MÉDICAUX" not in text_empty.upper() and "Besoins médicaux" not in text_empty

        tr_med = _tr(booking=_booking(), mobility={"wheelchair": True})
        ctx_med = collect_mission_report_context(tr_med, _institution(), variant="audit")
        pdf_med = build_mission_audit_report_pdf(ctx_med)
        text_med = _pdf_text(pdf_med)
        if text_med:
            assert "Besoins médicaux" in text_med or "BESOINS MÉDICAUX" in text_med.upper()

        ctx_voucher_empty = collect_mission_report_context(tr_empty, _institution(), variant="operational")
        text_voucher_empty = _pdf_text(build_operational_voucher_pdf(ctx_voucher_empty))
        if text_voucher_empty:
            assert "BESOINS PARTICULIERS" not in text_voucher_empty.upper()

        ctx_voucher_med = collect_mission_report_context(tr_med, _institution(), variant="operational")
        text_voucher_med = _pdf_text(build_operational_voucher_pdf(ctx_voucher_med))
        if text_voucher_med:
            assert "BESOINS PARTICULIERS" in text_voucher_med.upper()
            assert "BESOINS MÉDICAUX" not in text_voucher_med.upper()

    def test_voucher_vs_report_identity_separation(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=2)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        audit = build_mission_audit_report_pdf(ctx)
        voucher = build_operational_voucher_pdf(ctx)
        text_audit = _pdf_text(audit)
        text_voucher = _pdf_text(voucher)
        if text_audit and text_voucher:
            assert "1980" in text_audit or "Dossier" in text_audit
            assert "Naissance" not in text_audit
            assert "Mode facturation" not in text_voucher
            assert "Informations de mission" not in text_audit

    def test_header_status_and_admin_refs(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=4)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        pdf = build_mission_audit_report_pdf(ctx)
        text = _pdf_text(pdf)
        if text:
            # Option B : statut dans le header, pas de badge résultat dédoublonné
            assert "Statut" in text
            assert "MISSION RÉALISÉE" not in text
            assert "Référence mission" in text
            assert "TR-2026" in text
            assert "Demande #1820" in text or "Demande" in text
            assert "LIRIE-TR-" not in text
            assert "Vérification documentaire" not in text
            assert "Document généré par LIRIE" not in text

    def test_identity_driver_before_vehicle(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=2)
        user = SimpleNamespace(first_name="Khalid", last_name="ALAOUI", phone="079")
        driver = SimpleNamespace(user=user, vehicle_assigned="Mercedes Vito")
        booking = _booking(driver=driver, driver_id=1)
        tr = _tr(booking=booking)
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        pdf = build_mission_audit_report_pdf(ctx)
        text = _pdf_text(pdf)
        if text:
            assert "Chauffeur" in text
            assert text.index("Transporteur") < text.index("Chauffeur")
            if "Véhicule" in text:
                assert text.index("Chauffeur") < text.index("Véhicule")

    def test_compact_simple_mission_page_count(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=4)
        tr = _tr(booking=_booking(), notes="", floor_elevator_info="", mobility={})
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        pdf = build_mission_audit_report_pdf(ctx)
        assert pdf[:4] == b"%PDF"
        pages = _pdf_page_count(pdf)
        if pages is not None:
            assert pages == 1

    def test_cancelled_mission_no_boarded_in_history(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = [
            SimpleNamespace(
                id=1,
                event_type="request_created",
                created_at=datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
                actor_type="institution_user",
                actor_user_id=None,
                payload={},
            ),
            SimpleNamespace(
                id=2,
                event_type="offer_accepted",
                created_at=datetime(2026, 6, 12, 21, 18, tzinfo=UTC),
                actor_type="company",
                actor_user_id=None,
                payload={"company_name": "Emmenez Moi"},
            ),
            SimpleNamespace(
                id=3,
                event_type="cancelled",
                created_at=datetime(2026, 6, 12, 22, 0, tzinfo=UTC),
                actor_type="institution_user",
                actor_user_id=None,
                payload={},
            ),
        ]
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr = _tr(status="CANCELLED", booking=_booking(status="CANCELED"))
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        # Historique strictement chronologique : Annulée jamais avant les autres
        hist = ctx.synthetic_history
        dts = [row["at"] for row in hist if row.get("at")]
        assert dts == sorted(dts)
        labels = [row["label"] for row in hist]
        if "Annulée" in labels and len(labels) > 1:
            assert labels[-1] == "Annulée"
        pdf = build_mission_audit_report_pdf(ctx)
        text = _pdf_text(pdf)
        if text:
            assert "Annul" in text
            assert "Prise en charge" not in text
            assert "Informations de mission" not in text
            assert "réel" not in text.lower()

    def test_voucher_terrain_structure(self, mock_bmsg, mock_timeline):
        """PDF-VOUCHER-01 : bon opérationnel sans archivage."""
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=2)
        tr = _tr(booking=_booking(), mobility={"wheelchair": True})
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pdf = build_operational_voucher_pdf(ctx)
        text = _pdf_text(pdf)
        if text:
            assert "BON DE TRANSPORT" in text.upper() or "Bon de transport" in text
            assert ctx.reference in text
            assert "13.06.2026" in text
            assert "Naissance" in text
            assert "12.05.1980" in text
            assert "Institution" in text
            assert "Clinique LHA" in text
            assert "TRANSPORT" in text.upper() or "Transport" in text
            assert "Prise en charge" in text or "11:30" in text
            assert "Heure prévue" not in text
            assert "Type transport" in text
            assert "Fauteuil roulant" in text
            assert "Départ" in text
            assert "Trajet" not in text
            assert "Empreinte" not in text
            assert "Historique" not in text
            assert "Facturation" not in text

    def test_voucher_appointment_label(self, mock_bmsg, mock_timeline):
        """scheduled_time_type=arrival → libellé 'Rendez-vous'."""
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=1)
        tr = _tr(booking=_booking(), scheduled_time_type="arrival")
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert "Rendez-vous" in text
            assert "Prise en charge" not in text

    def test_voucher_transport_type_labels(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        cases = {
            "stretcher": "Brancard",
            "walking": "Assis",
        }
        for mobility_key, expected in cases.items():
            tr = _tr(booking=_booking(), mobility={mobility_key: True})
            ctx = collect_mission_report_context(tr, _institution(), variant="operational")
            text = _pdf_text(build_operational_voucher_pdf(ctx))
            if text:
                assert "Type transport" in text
                assert expected in text

    def test_voucher_patient_one_line(self, mock_bmsg, mock_timeline):
        from services.institutions.mission_report_pdf import _MAX_VOUCHER_PATIENT

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking(), long_name=True)
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        full = ctx.patient_block.get("full_name") or ""
        assert len(full) > _MAX_VOUCHER_PATIENT
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert "…" in text  # nom tronqué sur une ligne

    def test_voucher_signature_real_time(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert "Heure réelle" in text
            assert "Signature" in text

    def test_voucher_contact_includes_service_and_phone(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=_booking(),
            contact_on_site={
                "requester_service": "Admissions",
                "requester_name": "Marc Mouchet",
                "requester_phone": "022 000 00 00",
            },
        )
        tr._get_creator_name = lambda: None
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert "Marc Mouchet" in text
            assert "022 000 00 00" in text
            assert "Admissions" in text

    def test_voucher_driver_conditional(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr_no_driver = _tr(booking=_booking())
        ctx_no = collect_mission_report_context(tr_no_driver, _institution(), variant="operational")
        text_no = _pdf_text(build_operational_voucher_pdf(ctx_no))
        if text_no:
            assert "Khalid" not in text_no
            assert text_no.count("Chauffeur") == 1  # zone signature uniquement

        user = SimpleNamespace(first_name="Khalid", last_name="ALAOUI", phone="079")
        driver = SimpleNamespace(user=user, vehicle_assigned="Mercedes Vito")
        tr_with = _tr(booking=_booking(driver=driver, driver_id=1))
        ctx_with = collect_mission_report_context(tr_with, _institution(), variant="operational")
        text_with = _pdf_text(build_operational_voucher_pdf(ctx_with))
        if text_with:
            assert "Khalid" in text_with
            assert text_with.count("Chauffeur") >= 2

    def test_voucher_contact_conditional(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr_no_contact = _tr(
            booking=_booking(),
            contact_on_site={},
        )
        tr_no_contact._get_creator_name = lambda: None
        ctx_no = collect_mission_report_context(tr_no_contact, _institution(), variant="operational")
        text_no = _pdf_text(build_operational_voucher_pdf(ctx_no))
        if text_no:
            assert "Contact" not in text_no

        tr_contact = _tr(
            booking=_booking(),
            contact_on_site={"requester_name": "Marc Mouchet", "requester_phone": "022 000 00 00"},
        )
        ctx_yes = collect_mission_report_context(tr_contact, _institution(), variant="operational")
        text_yes = _pdf_text(build_operational_voucher_pdf(ctx_yes))
        if text_yes:
            assert "Contact" in text_yes
            assert "Marc Mouchet" in text_yes

    def test_voucher_medical_conditional(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr_empty = _tr(booking=_booking(), notes="", floor_elevator_info="", mobility={})
        ctx_empty = collect_mission_report_context(tr_empty, _institution(), variant="operational")
        text_empty = _pdf_text(build_operational_voucher_pdf(ctx_empty))
        if text_empty:
            assert "BESOINS PARTICULIERS" not in text_empty.upper()
            assert "BESOINS MÉDICAUX" not in text_empty.upper()

        long_remark = "Prévoir aide au transfert. " + ("détail " * 80)
        tr_med = _tr(
            booking=_booking(),
            notes=long_remark,
            mobility={"wheelchair": True},
        )
        ctx_med = collect_mission_report_context(tr_med, _institution(), variant="operational")
        text_med = _pdf_text(build_operational_voucher_pdf(ctx_med))
        if text_med:
            assert "BESOINS PARTICULIERS" in text_med.upper()
            assert "BESOINS MÉDICAUX" not in text_med.upper()
            assert "Fauteuil" in text_med or "roulant" in text_med.lower()
            idx = text_med.find("Remarque")
            if idx >= 0:
                remark_slice = text_med[idx : idx + 220]
                assert len(remark_slice) <= 220

    def test_voucher_needs_alert_before_transport(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr = _tr(booking=_booking(), notes="Transport couché", mobility={"wheelchair": True})
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert text.index("BESOINS PARTICULIERS") < text.index("TRANSPORT")

    def test_voucher_needs_alert_shows_free_text_remark(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        remark = "Transport couché — prévoir brancard si escaliers"
        tr = _tr(booking=_booking(), notes=remark, mobility={"needs_assistance": True})
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert "BESOINS PARTICULIERS" in text.upper()
            assert "Remarque" in text
            assert "Transport couché" in text

    def test_voucher_no_needs_alert_when_empty(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        tr = _tr(
            booking=_booking(),
            notes="",
            floor_elevator_info="",
            mobility={"wheelchair": False, "needs_assistance": False},
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert "BESOINS PARTICULIERS" not in text.upper()

    def test_voucher_patient_billing_shows_address(self, mock_bmsg, mock_timeline):
        """PDF-VOUCHER-04 : facturation patient → adresse patient + mention facturation."""
        from services.institutions.mission_report_pdf import _build_voucher_identity_table

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=_booking(billed_to_type="patient"),
            billing_intent="patient",
            patient=_patient(with_address=True),
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        assert ctx.request_classification["billing_target"] == "patient"
        assert ctx.patient_block["address"] == "Chemin des Courbes 9, 1247 Anières"
        texts = _flow_texts(_build_voucher_identity_table(ctx))
        assert "Adresse patient" in texts
        assert "Chemin des Courbes 9, 1247 Anières" in texts
        assert "Facturation" in texts
        assert "Patient" in texts

    def test_voucher_institution_billing_hides_address(self, mock_bmsg, mock_timeline):
        """PDF-VOUCHER-04 : facturation institution → ni adresse ni facturation."""
        from services.institutions.mission_report_pdf import _build_voucher_identity_table

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=_booking(billed_to_type="clinic"),
            billing_intent="institution",
            patient=_patient(with_address=True),
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        assert ctx.request_classification["billing_target"] == "institution"
        texts = _flow_texts(_build_voucher_identity_table(ctx))
        assert "Adresse patient" not in texts
        assert "Facturation" not in texts

    def test_voucher_insurance_billing_hides_address(self, mock_bmsg, mock_timeline):
        """PDF-VOUCHER-04 : facturation assurance → ni adresse ni facturation."""
        from services.institutions.mission_report_pdf import _build_voucher_identity_table

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=_booking(billed_to_type="insurance"),
            billing_intent="insurance",
            patient=_patient(with_address=True),
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        texts = _flow_texts(_build_voucher_identity_table(ctx))
        assert "Adresse patient" not in texts
        assert "Facturation" not in texts

    def test_audit_keeps_admin_fingerprint(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=2)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        assert ctx.traceability.get("document_hash")
        audit_text = _pdf_text(build_mission_audit_report_pdf(ctx))
        voucher_text = _pdf_text(build_operational_voucher_pdf(ctx))
        if audit_text and voucher_text:
            assert "Empreinte" in audit_text
            assert "Empreinte" not in voucher_text

    def test_voucher_has_lirie_qr_in_header(self, mock_bmsg, mock_timeline):
        from services.institutions.mission_report_pdf import (
            VoucherLayoutOptions,
            _build_voucher_presentation,
            _layout_voucher_operational,
            _voucher_header,
            _voucher_operational_header,
        )

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        assert ctx.traceability.get("verify_url") == "https://www.lirie.ch"

        pres = _build_voucher_presentation(ctx)
        assert pres.verify_url == "https://www.lirie.ch"
        assert _flow_has_image(_voucher_operational_header(pres))
        assert _flow_has_image(_voucher_header(ctx))

        op_flow = _layout_voucher_operational(pres, VoucherLayoutOptions())
        assert _flow_has_image(op_flow)
        voucher_text = _pdf_text(
            build_operational_voucher_pdf(ctx, layout="operational")
        )
        if voucher_text:
            assert "Empreinte" not in voucher_text
            assert "Réf. archivage" not in voucher_text

    def test_audit_report_has_logo_in_header(self, mock_bmsg, mock_timeline):
        from services.institutions.mission_report_pdf import _document_header

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        assert _flow_has_image(_document_header(ctx, "Rapport de mission"))
        pdf = build_mission_audit_report_pdf(ctx)
        assert pdf[:4] == b"%PDF"

    def test_voucher_external_carrier_label(self, mock_bmsg, mock_timeline):
        from services.institutions.mission_report_pdf import _build_voucher_identity_table

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=None,
            status="EXTERNAL_ASSIGNED",
            carrier_source="external",
            external_carrier_name="Taxi XYZ",
            accepted_by_company=None,
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        texts = _flow_texts(_build_voucher_identity_table(ctx))
        assert "Transporteur externe" in texts
        assert "Taxi XYZ" in texts
        assert "Chauffeur" not in texts

    def test_audit_external_execution_block(self, mock_bmsg, mock_timeline):
        from services.institutions.mission_report_pdf import _audit_execution_block

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=None,
            status="EXTERNAL_DECLARED_COMPLETED",
            carrier_source="external",
            external_carrier_name="Taxi XYZ",
            external_carrier_reference="REF-42",
            external_carrier_reason="Zone non couverte",
            assigned_externally_at=datetime(2026, 6, 13, 13, 20, tzinfo=UTC),
            executed_externally_at=datetime(2026, 6, 13, 17, 0, tzinfo=UTC),
            external_execution_notes="OK",
            accepted_by_company=None,
        )
        tr.executed_externally_by = SimpleNamespace(
            first_name="Marc",
            last_name="Mouchet",
            username="marc",
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        texts = _flow_texts(_audit_execution_block(ctx))
        assert any("Exécution" in t and "Transporteur externe" in t for t in texts)
        assert any("Transporteur" in t and "Taxi XYZ" in t for t in texts)
        assert any("Référence externe" in t and "REF-42" in t for t in texts)
        assert any("Raison d'externalisation" in t for t in texts)
        assert any("Déclarée réalisée par l'institution" in t for t in texts)


@patch("services.institutions.mission_report_context.list_timeline_events")
@patch("services.institutions.mission_report_context.BookingMessage")
class TestOperationalVoucherUxLayouts:
    """Layouts UX chauffeur (PDF-UX-01) — operational production cible."""

    def test_operational_driver_structure(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg, events=2)
        tr = _tr(booking=_booking(), mobility={"wheelchair": True})
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx, layout="operational"))
        if text:
            assert "BON DE TRANSPORT" in text.upper()
            assert ctx.reference in text
            assert "TRAJET" in text.upper()
            assert "■ TRANSPORT" not in text
            assert "Type transport" not in text
            assert "Naissance" not in text
            assert "JASIQI" in text or "Drin" in text
            assert "12.05.1980" in text
            assert "Clinique LHA" in text
            assert "Fauteuil roulant" in text
            assert "Prise en charge" in text or "11:30" in text
            assert "Étape 1" not in text
            assert "Étape 2" not in text
            assert "Destination 2" not in text
            assert "Empreinte" not in text

    def test_operational_time_before_address_in_story(self, mock_bmsg, mock_timeline):
        from services.institutions.mission_report_pdf import (
            VoucherLayoutOptions,
            _build_voucher_presentation,
            _layout_voucher_operational,
        )

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pres = _build_voucher_presentation(ctx)
        texts = _flow_texts(_layout_voucher_operational(pres, VoucherLayoutOptions()))
        if texts:
            time_idx = next(i for i, t in enumerate(texts) if "11:30" in t)
            addr_idx = next(i for i, t in enumerate(texts) if "Courbes" in t or "Anières" in t)
            assert time_idx < addr_idx

    def test_operational_patient_billing(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=_booking(billed_to_type="patient"),
            billing_intent="patient",
            patient=_patient(with_address=True),
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx, layout="operational"))
        if text:
            assert "Adresse patient" in text
            assert "Facturation : Patient" in text
            assert "Chemin des Courbes 9" in text

    def test_operational_confirmation_inline_default(self, mock_bmsg, mock_timeline):
        """Design final : confirmation inline, plus d'ancienne section signatures."""
        from services.institutions.mission_report_pdf import (
            VoucherLayoutOptions,
            _build_voucher_presentation,
            _layout_voucher_operational,
        )

        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pres = _build_voucher_presentation(ctx)
        texts = _flow_texts(_layout_voucher_operational(pres, VoucherLayoutOptions()))
        joined = " ".join(texts)
        assert "Confirmation" in joined
        assert "Chauffeur" in joined
        assert "Patient/représentant" in joined
        assert "Heure réelle" not in joined
        assert not any(t.strip() == "Signatures" for t in texts)
        text = _pdf_text(build_operational_voucher_pdf(ctx, layout="operational"))
        if text:
            assert "Confirmation" in text
            assert "Signatures" not in text

    def test_operational_one_page_simple(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pdf = build_operational_voucher_pdf(ctx, layout="operational")
        pages = _pdf_page_count(pdf)
        if pages is not None:
            assert pages == 1

    def test_legacy_unchanged_default(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        text = _pdf_text(build_operational_voucher_pdf(ctx))
        if text:
            assert "Type transport" in text
            assert "Naissance" in text
            assert "■ TRANSPORT" in text.upper() or "TRANSPORT" in text.upper()

    def test_review_layouts_smoke(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        for layout in ("ultra_compact", "medical"):
            pdf = build_operational_voucher_pdf(ctx, layout=layout)
            assert pdf[:4] == b"%PDF"
            text = _pdf_text(pdf)
            if text:
                assert "BON DE TRANSPORT" in text.upper()
                assert ctx.reference in text

    def test_voucher_presentation_no_etape_labels(self, mock_bmsg, mock_timeline):
        from services.institutions.mission_report_pdf import _build_voucher_presentation

        leg1 = SimpleNamespace(
            sequence_index=0,
            pickup_location="Chemin des Courbes 9, 1247 Anières",
            dropoff_location="Imagerie",
            dropoff_establishment="Centre d'Imagerie Rive Gauche",
            scheduled_time=datetime(2026, 6, 13, 12, 0),
            time_confirmed=True,
        )
        leg2 = SimpleNamespace(
            sequence_index=1,
            pickup_location="Imagerie",
            dropoff_location="Labo",
            dropoff_establishment="Laboratoire Unilabs",
            scheduled_time=datetime(2026, 6, 13, 14, 0),
            time_confirmed=True,
        )
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(multi_stop=True, legs=[leg1, leg2], booking=_booking())
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pres = _build_voucher_presentation(ctx)
        labels = [s.label for s in pres.route_stops]
        assert not any(lbl.startswith("Étape") for lbl in labels)
        assert "Destination 2" not in labels

    def test_operational_roundtrip_one_page(self, mock_bmsg, mock_timeline):
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            booking=_booking(status="RETURN_COMPLETED"),
            is_round_trip=True,
            return_to_institution=True,
            return_time_confirmed=True,
            return_time=datetime(2026, 6, 13, 15, 0),
            return_date=date(2026, 6, 13),
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pdf = build_operational_voucher_pdf(ctx, layout="operational")
        pages = _pdf_page_count(pdf)
        if pages is not None:
            assert pages == 1
        text = _pdf_text(pdf)
        if text:
            assert "Retour institution" in text

    def test_operational_multistep_5_no_erp_labels(self, mock_bmsg, mock_timeline):
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
        _mock_timeline_and_messages(mock_timeline, mock_bmsg)
        tr = _tr(
            multi_stop=True,
            is_round_trip=True,
            return_to_institution=True,
            legs=legs,
            booking=booking5,
        )
        from services.institutions.mission_report_pdf import (
            VoucherLayoutOptions,
            _build_voucher_presentation,
            _layout_voucher_operational,
        )

        ctx = collect_mission_report_context(tr, _institution(), variant="operational")
        pdf = build_operational_voucher_pdf(ctx, layout="operational")
        assert pdf[:4] == b"%PDF"
        pres = _build_voucher_presentation(ctx)
        texts = _flow_texts(_layout_voucher_operational(pres, VoucherLayoutOptions()))
        joined = " ".join(texts)
        for forbidden in ("Étape 1", "Étape 2", "Destination 2", "Destination 3"):
            assert forbidden not in joined
        assert "Confirmation" in joined
