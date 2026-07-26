"""Tests unitaires — contexte export PDF mission institution (STOP GATE PDF-01)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.institutions.mission_report_context import (
    MAX_MESSAGES,
    MAX_TIMELINE_EVENTS,
    build_carrier_block,
    build_client_identity_block,
    build_completion_certificate,
    build_gps_proof,
    build_mission_milestones,
    build_mission_status_label,
    build_patient_block,
    build_request_classification,
    build_synthetic_history,
    collect_mission_report_context,
    compute_document_hash,
    format_booking_number,
    format_request_number,
    format_transport_reference,
    resolve_institution_snapshot,
    resolve_timeline_channel,
)


def _institution(**kwargs):
    defaults = {
        "id": 1,
        "name": "Clinique LHA",
        "contact_phone": "+41 22 000 00 00",
        "contact_email": "a@clinique.ch",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _patient(**kwargs):
    defaults = {
        "first_name": "Drin",
        "last_name": "JASIQI",
        "dob": date(1980, 5, 12),
        "external_reference": "DPI-99",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _tr(**kwargs):
    defaults = {
        "id": 1820,
        "public_id": "uuid-tr-1820",
        "institution_id": 1,
        "booking_id": None,
        "created_at": datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
        "accepted_at": datetime(2026, 6, 12, 21, 18, tzinfo=UTC),
        "mission_date": date(2026, 6, 13),
        "mission_type": "patient_transport",
        "billing_intent": "institution",
        "status": "SENT",
        "pickup_location": "Chemin des Courbes 9, Anières",
        "dropoff_location": "Clinique Beaulieu, Genève",
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
        "notes": "Patient désorienté",
        "floor_elevator_info": "3e étage, ascenseur",
        "mobility": {"wheelchair": True, "needs_assistance": True},
    }
    defaults.update(kwargs)
    tr = SimpleNamespace(**defaults)
    tr.institution = kwargs.get("institution") or _institution()
    tr.patient = kwargs.get("patient") if "patient" in kwargs else _patient()
    tr.booking = kwargs.get("booking")
    tr.accepted_by_company = kwargs.get("accepted_by_company")
    tr.get_mobility = lambda: defaults.get("mobility") or {"wheelchair": False}
    tr._get_creator_name = lambda: "Marc Mouchet"
    tr._serialize_booking_summary = lambda: kwargs.get("booking_summary") or None
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
    }
    defaults.update(kwargs)
    b = SimpleNamespace(**defaults)
    b._get_route_journey = lambda: kwargs.get("route_journey") or [
        {"event": "Prise en charge", "date": "2026-06-13T11:34:00Z", "type": "pickup"},
        {"event": "Dépose", "date": "2026-06-13T14:07:00Z", "type": "dropoff"},
    ]
    return b


class TestReferencesAndStatus:
    def test_format_transport_reference(self):
        tr = _tr(id=123, created_at=datetime(2026, 1, 1, tzinfo=UTC))
        assert format_transport_reference(tr) == "TR-2026-000123"

    def test_format_request_and_booking_numbers(self):
        tr = _tr(id=1820)
        assert format_request_number(tr) == "#1820"
        assert format_booking_number(_booking()) == "#4567"
        assert format_booking_number(None) is None

    def test_build_mission_status_label_completed(self):
        tr = _tr(is_round_trip=False)
        assert build_mission_status_label(tr, _booking(status="COMPLETED")) == "Réalisé"

    def test_build_mission_status_label_return_completed(self):
        tr = _tr(is_round_trip=True)
        assert (
            build_mission_status_label(tr, _booking(status="RETURN_COMPLETED"))
            == "Réalisé (aller-retour)"
        )

    def test_build_mission_status_label_cancelled(self):
        tr = _tr(status="CANCELLED")
        assert build_mission_status_label(tr, _booking(status="CANCELED")) == "Annulé"

    def test_build_mission_status_label_no_booking_sent(self):
        tr = _tr(status="SENT")
        assert build_mission_status_label(tr, None) == "Envoyée"

    def test_no_raw_status_codes_in_label(self):
        label = build_mission_status_label(_tr(), _booking(status="COMPLETED"))
        assert "COMPLETED" not in label


class TestClientIdentityAndClassification:
    def test_client_identity_institution_patient(self):
        tr = _tr()
        block = build_client_identity_block(tr, None)
        assert "Patient" in block["headline"]
        assert block["display_category"] == "institution_patient"

    def test_request_classification_one_way(self):
        tr = _tr()
        cls = build_request_classification(tr, None)
        assert cls["trip_type"] == "one_way"
        assert cls["billing_target"] == "institution"
        assert cls["mobility_level"] == "wheelchair"


class TestTimelineChannel:
    def test_resolve_timeline_channel_driver(self):
        ev = SimpleNamespace(actor_type="driver")
        assert resolve_timeline_channel(ev) == "Mobile chauffeur"

    def test_resolve_timeline_channel_system(self):
        ev = SimpleNamespace(actor_type="system")
        assert resolve_timeline_channel(ev) == "Automatique"


class TestCollectContextScenarios:
    @patch("services.institutions.mission_report_context.list_timeline_events")
    @patch("services.institutions.mission_report_context.BookingMessage")
    def test_pdf01a_simple_mission(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []

        tr = _tr(status="CONVERTED", booking_id=4567, booking=_booking())
        tr.accepted_by_company = SimpleNamespace(
            name="Emmenez Moi", contact_phone="079", contact_email="e@em.com"
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        assert ctx.reference == "TR-2026-001820"
        assert ctx.booking_number == "#4567"
        assert ctx.status_label == "Réalisé"
        assert len(ctx.route_steps) >= 2

    @patch("services.institutions.mission_report_context.list_timeline_events")
    @patch("services.institutions.mission_report_context.BookingMessage")
    def test_pdf01b_multi_stop(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []

        leg1 = SimpleNamespace(
            sequence_index=0,
            route_sequence_number=1,
            pickup_location="A",
            dropoff_location="B",
            dropoff_establishment="Imagerie",
            scheduled_time=datetime(2026, 6, 13, 12, 0),
            time_confirmed=True,
        )
        leg2 = SimpleNamespace(
            sequence_index=1,
            route_sequence_number=2,
            pickup_location="B",
            dropoff_location="C",
            dropoff_establishment="Labo",
            scheduled_time=datetime(2026, 6, 13, 14, 0),
            time_confirmed=True,
        )
        booking = _booking(
            route_journey=[
                {"type": "pickup", "date": "2026-06-13T11:34:00Z"},
                {"type": "dropoff", "date": "2026-06-13T12:30:00Z"},
                {"type": "dropoff", "date": "2026-06-13T14:15:00Z"},
            ]
        )
        tr = _tr(
            multi_stop=True,
            legs=[leg1, leg2],
            booking_id=4567,
            booking=booking,
            status="CONVERTED",
        )
        ctx = collect_mission_report_context(tr, _institution())
        destinations = [s for s in ctx.route_steps if s["kind"] == "destination"]
        assert len(destinations) == 2

    @patch("services.institutions.mission_report_context.list_timeline_events")
    @patch("services.institutions.mission_report_context.BookingMessage")
    def test_pdf01c_round_trip(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []

        tr = _tr(
            is_round_trip=True,
            return_time_confirmed=True,
            return_time=datetime(2026, 6, 13, 15, 0),
            return_date=date(2026, 6, 13),
            booking_id=4567,
            booking=_booking(status="RETURN_COMPLETED"),
            status="CONVERTED",
        )
        ctx = collect_mission_report_context(tr, _institution())
        assert ctx.status_label == "Réalisé (aller-retour)"
        assert any(s["kind"] == "return" for s in ctx.route_steps)

    @patch("services.institutions.mission_report_context.list_timeline_events")
    @patch("services.institutions.mission_report_context.BookingMessage")
    def test_pdf01d_cancelled(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []

        tr = _tr(
            status="CANCELLED", booking_id=4567, booking=_booking(status="CANCELED")
        )
        ctx = collect_mission_report_context(tr, _institution())
        assert ctx.status_label == "Annulé"

    @patch("services.institutions.mission_report_context.list_timeline_events")
    @patch("services.institutions.mission_report_context.BookingMessage")
    def test_pdf01e_no_booking(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = [
            SimpleNamespace(
                id=1,
                event_type="request_created",
                created_at=datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
                actor_type="system",
                actor_user_id=None,
                payload={"company_name": "LIRIE"},
            )
        ]
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0

        tr = _tr(status="SENT", booking_id=None, booking=None, patient=None)
        ctx = collect_mission_report_context(tr, _institution())
        assert ctx.booking_number is None
        assert ctx.messages == []
        assert len(ctx.timeline_rows) == 1

    def test_compute_document_hash_deterministic(self):
        tr = _tr()
        with (
            patch(
                "services.institutions.mission_report_context.list_timeline_events",
                return_value=[],
            ),
            patch(
                "services.institutions.mission_report_context.BookingMessage"
            ) as mock_bmsg,
        ):
            mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
            mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []
            ctx = collect_mission_report_context(tr, _institution())
        h1 = compute_document_hash(ctx)
        h2 = compute_document_hash(ctx)
        assert h1 == h2
        assert len(h1) == 16

    def test_attachments_empty_v1(self):
        tr = _tr()
        with (
            patch(
                "services.institutions.mission_report_context.list_timeline_events",
                return_value=[],
            ),
            patch(
                "services.institutions.mission_report_context.BookingMessage"
            ) as mock_bmsg,
        ):
            mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
            mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []
            ctx = collect_mission_report_context(tr, _institution())
        assert ctx.attachments == []


class TestVolumeLimits:
    def test_constants(self):
        assert MAX_TIMELINE_EVENTS == 500
        assert MAX_MESSAGES == 200


class TestV11DriverVehicleMilestones:
    def test_carrier_block_includes_driver_and_vehicle(self):
        user = SimpleNamespace(
            first_name="Jean", last_name="Dupont", phone="079 111 22 33"
        )
        driver = SimpleNamespace(user=user, vehicle_assigned="Mercedes Vito AB-123-CD")
        booking = _booking(driver=driver, driver_id=42)
        tr = _tr(booking=booking, booking_id=4567)
        tr.accepted_by_company = SimpleNamespace(
            name="Emmenez Moi", contact_phone="079", contact_email="e@em.com"
        )
        block = build_carrier_block(tr, booking)
        assert block["driver_name"] == "Jean Dupont"
        assert block["driver_phone"] == "079 111 22 33"
        assert block["vehicle"] == "Mercedes Vito AB-123-CD"

    def test_build_mission_milestones_from_timeline(self):
        events = [
            SimpleNamespace(
                event_type="offer_accepted",
                created_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
                payload={},
            ),
            SimpleNamespace(
                event_type="patient_boarded",
                created_at=datetime(2026, 6, 13, 11, 34, tzinfo=UTC),
                payload={},
            ),
        ]
        tr = _tr()
        booking = _booking()
        milestones = build_mission_milestones(tr, booking, events)
        labels = [m["milestone"] for m in milestones]
        assert "Acceptation transporteur" in labels
        assert "Patient embarqué" in labels


class TestSyntheticHistory:
    def test_build_synthetic_history_standard_order(self):
        events = [
            SimpleNamespace(
                event_type="request_created",
                created_at=datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
                payload={},
            ),
            SimpleNamespace(
                event_type="offer_sent",
                created_at=datetime(2026, 6, 12, 21, 0, tzinfo=UTC),
                payload={},
            ),
            SimpleNamespace(
                event_type="offer_accepted",
                created_at=datetime(2026, 6, 12, 21, 18, tzinfo=UTC),
                payload={"company_name": "Emmenez Moi"},
            ),
            SimpleNamespace(
                event_type="patient_boarded",
                created_at=datetime(2026, 6, 13, 11, 34, tzinfo=UTC),
                payload={},
            ),
            SimpleNamespace(
                event_type="patient_completed",
                created_at=datetime(2026, 6, 13, 14, 7, tzinfo=UTC),
                payload={},
            ),
        ]
        tr = _tr()
        booking = _booking()
        history = build_synthetic_history(tr, booking, events)
        labels = [row["label"] for row in history]
        assert len(history) <= 4
        assert labels[0] == "Demande créée"
        assert any(label.startswith("Acceptée par") for label in labels)
        assert "Prise en charge" in labels
        assert "Mission terminée" in labels
        assert "Offre envoyée" not in labels

    def test_build_synthetic_history_cancelled(self):
        events = [
            SimpleNamespace(
                event_type="request_created",
                created_at=datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
                payload={},
            ),
            SimpleNamespace(
                event_type="offer_accepted",
                created_at=datetime(2026, 6, 12, 21, 18, tzinfo=UTC),
                payload={"company_name": "Emmenez Moi"},
            ),
            SimpleNamespace(
                event_type="patient_boarded",
                created_at=datetime(2026, 6, 13, 11, 34, tzinfo=UTC),
                payload={},
            ),
            SimpleNamespace(
                event_type="cancelled",
                created_at=datetime(2026, 6, 12, 22, 0, tzinfo=UTC),
                payload={},
            ),
        ]
        tr = _tr(status="CANCELLED")
        booking = _booking(status="CANCELED")
        history = build_synthetic_history(tr, booking, events)
        labels = [row["label"] for row in history]
        assert "Annulée" in labels
        assert "Prise en charge" not in labels
        assert "Mission terminée" not in labels
        assert all("at" in row for row in history)
        dts = [row["at"] for row in history]
        assert dts == sorted(dts)

    def test_build_synthetic_history_cancelled_sorted_without_timestamp(self):
        """Annulation sans timestamp réel : placée après les autres événements."""
        tr = _tr(
            status="CANCELLED",
            created_at=datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
            accepted_at=datetime(2026, 6, 12, 21, 18, tzinfo=UTC),
        )
        tr.accepted_by_company = SimpleNamespace(name="Emmenez Moi")
        booking = _booking(status="CANCELED")
        booking.updated_at = None
        history = build_synthetic_history(tr, booking, [])
        labels = [row["label"] for row in history]
        dts = [row["at"] for row in history]
        assert dts == sorted(dts)
        assert labels[-1] == "Annulée"

    @patch("services.institutions.mission_report_context.list_timeline_events")
    @patch("services.institutions.mission_report_context.BookingMessage")
    def test_synthetic_history_in_collected_context(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = [
            SimpleNamespace(
                id=1,
                event_type="request_created",
                created_at=datetime(2026, 6, 12, 20, 48, tzinfo=UTC),
                actor_type="system",
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
        ]
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []
        tr = _tr(status="CONVERTED", booking_id=4567, booking=_booking())
        tr.accepted_by_company = SimpleNamespace(
            name="Emmenez Moi", contact_phone="079", contact_email="e@em.com"
        )
        ctx = collect_mission_report_context(tr, _institution(), variant="audit")
        assert ctx.synthetic_history
        assert len(ctx.synthetic_history) <= 4


class TestV2PatientRoomAndCertificate:
    def test_patient_room_from_floor_and_residence(self):
        patient = _patient()
        patient.floor = 3
        patient.residence_name = "Unité A"
        tr = _tr(patient=patient)
        block = build_patient_block(tr)
        assert "Étage 3" in block["room"]
        assert "Unité A" in block["room"]

    def test_completion_certificate_only_when_realized(self):
        cert = build_completion_certificate(
            reference="TR-2026-001820",
            status_label="Réalisé",
            patient_block={"full_name": "JASIQI Drin"},
            institution_snapshot={"name": "Clinique LHA"},
            carrier_block={"name": "Emmenez Moi"},
            mission_info={"mission_date": "13.06.2026"},
            document_hash="abc123",
            public_id="uuid-1",
            generated_at=datetime(2026, 6, 13, 15, 0, tzinfo=UTC),
        )
        assert cert is not None
        assert cert["title"] == "Certificat de réalisation"
        assert (
            build_completion_certificate(
                reference="TR-2026-001820",
                status_label="Envoyée",
                patient_block={},
                institution_snapshot={},
                carrier_block={},
                mission_info={},
                document_hash="x",
                public_id="y",
                generated_at=datetime(2026, 6, 13, 15, 0, tzinfo=UTC),
            )
            is None
        )

    @patch("services.institutions.transport_timeline_service.find_latest_event")
    def test_resolve_institution_snapshot_persisted(self, mock_find):
        mock_find.return_value = SimpleNamespace(
            payload={
                "institution_snapshot": {"name": "Snapshot figé", "service": "Urgences"}
            }
        )
        tr = _tr()
        snap = resolve_institution_snapshot(tr, _institution())
        assert snap["name"] == "Snapshot figé"
        assert snap["source"] == "persisted"


class TestV3GpsAndArchiving:
    def test_gps_proof_fallback_without_driver(self):
        proof = build_gps_proof(None)
        assert proof["available"] is False
        assert "LIRIE" in proof["message"]

    @patch("services.institutions.mission_report_context.list_timeline_events")
    @patch("services.institutions.mission_report_context.BookingMessage")
    def test_archive_reference_in_context(self, mock_bmsg, mock_timeline):
        mock_timeline.return_value = []
        mock_bmsg.query.filter_by.return_value.order_by.return_value.count.return_value = 0
        mock_bmsg.query.filter_by.return_value.order_by.return_value.all.return_value = []
        ctx = collect_mission_report_context(_tr(), _institution())
        assert ctx.traceability["archive_reference"].startswith("LIRIE-TR-")
        assert ctx.gps_proof is not None
        assert ctx.traceability.get("verify_url") == "https://www.lirie.ch"
        assert ctx.traceability.get("verify_label") == "Document généré via LIRIE"
        assert ctx.traceability.get("edition_date")
