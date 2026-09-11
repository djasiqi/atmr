"""Changement de RDV institution : détection + invalidation départ transporteur."""

from __future__ import annotations

import uuid
from datetime import date, datetime

import pytest

from models import Booking, Institution, TransportRequest, TransportRequestLeg
from models.enums import BookingStatus, InstitutionRole, RequestStatus
from services.institutions.booking_change_service import (
    InstitutionBookingContext,
    update_institution_booking,
)


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = "Clinique les Hauts d'Anières"
    inst.institution_type = "clinic"
    db.session.add(inst)
    db.session.flush()
    return inst


@pytest.fixture
def accepted_world(db, test_company, test_client, institution):
    if not test_company or not test_client:
        pytest.skip("test_company and test_client required")

    pickup_at = datetime(2026, 9, 12, 13, 15)
    rdv_at = datetime(2026, 9, 12, 14, 0)

    booking = Booking()
    booking.user_id = test_client.user_id
    booking.company_id = test_company.id
    booking.client_id = test_client.id
    booking.customer_name = "Charlotte CAVADINI"
    booking.pickup_location = "Chemin des Courbes 9, 1247, Anières"
    booking.dropoff_location = (
        "Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4"
    )
    booking.scheduled_time = pickup_at
    booking.time_confirmed = True
    booking.status = BookingStatus.ACCEPTED
    booking.hospital_service = "Radiologie"
    booking.amount = 40.0
    booking.edit_version = 1
    db.session.add(booking)
    db.session.flush()

    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.pickup_location = booking.pickup_location
    tr.dropoff_location = booking.dropoff_location
    tr.scheduled_time = pickup_at
    tr.mission_date = date(2026, 9, 12)
    tr.pickup_time_confirmed = True
    tr.return_to_institution = True
    tr.status = RequestStatus.CONVERTED.value
    tr.booking_id = booking.id
    tr.accepted_by_company_id = test_company.id
    tr.created_by_display_name = "Admin LHA"
    tr.billing_intent = "patient"
    db.session.add(tr)
    db.session.flush()

    dest = TransportRequestLeg(
        transport_request_id=tr.id,
        sequence_index=0,
        route_sequence_number=1,
        pickup_location=booking.pickup_location,
        dropoff_location=booking.dropoff_location,
        scheduled_time=rdv_at,
        time_confirmed=True,
        booking_id=booking.id,
    )
    ret = TransportRequestLeg(
        transport_request_id=tr.id,
        sequence_index=1,
        route_sequence_number=2,
        pickup_location=booking.dropoff_location,
        dropoff_location=booking.pickup_location,
        scheduled_time=None,
        time_confirmed=False,
        is_return_stop=True,
    )
    db.session.add_all([dest, ret])
    db.session.flush()
    return {
        "booking": booking,
        "transport_request": tr,
        "dest_leg": dest,
        "return_leg": ret,
        "institution": institution,
    }


def _ctx(world) -> InstitutionBookingContext:
    return InstitutionBookingContext(
        booking=world["booking"],
        transport_request=world["transport_request"],
        institution_id=world["institution"].id,
    )


def _base_payload(**overrides):
    payload = {
        "version": 1,
        "pickup_location": "Chemin des Courbes 9, 1247, Anières",
        "dropoff_location": (
            "Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4"
        ),
        "scheduled_time": "2026-09-12T13:15:00",
        "hospital_service": "Radiologie",
        "appointment_time": "2026-09-12T14:00:00",
        "leg_appointments": [{"index": 0, "scheduled_time": "2026-09-12T14:00:00"}],
    }
    payload.update(overrides)
    return payload


@pytest.mark.usefixtures("requires_postgresql")
class TestAppointmentReconfirmation:
    def test_rdv_14_to_13_invalidates_pickup(self, db, accepted_world):
        body, code = update_institution_booking(
            _ctx(accepted_world),
            payload=_base_payload(
                appointment_time="2026-09-12T13:00:00",
                leg_appointments=[
                    {"index": 0, "scheduled_time": "2026-09-12T13:00:00"}
                ],
            ),
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin LHA",
        )

        assert code == 200, body
        assert "Aucun champ modifié" not in str(body)
        booking = accepted_world["booking"]
        tr = accepted_world["transport_request"]
        dest = accepted_world["dest_leg"]
        db.session.refresh(booking)
        db.session.refresh(tr)
        db.session.refresh(dest)
        assert booking.status == BookingStatus.ACCEPTED
        assert dest.scheduled_time == datetime(2026, 9, 12, 13, 0)
        assert dest.time_confirmed is True
        assert booking.time_confirmed is False
        assert tr.pickup_time_confirmed is False
        assert body.get("pickup_reconfirmation_required") is True

    def test_rdv_14_to_15_also_invalidates(self, db, accepted_world):
        body, code = update_institution_booking(
            _ctx(accepted_world),
            payload=_base_payload(
                appointment_time="2026-09-12T15:00:00",
                leg_appointments=[
                    {"index": 0, "scheduled_time": "2026-09-12T15:00:00"}
                ],
            ),
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin LHA",
        )
        assert code == 200, body
        booking = accepted_world["booking"]
        db.session.refresh(booking)
        assert booking.status == BookingStatus.ACCEPTED
        assert booking.time_confirmed is False

    def test_access_note_keeps_pickup_confirmed(self, db, accepted_world):
        body, code = update_institution_booking(
            _ctx(accepted_world),
            payload=_base_payload(pickup_access_notes="Code porte 1234"),
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin LHA",
        )
        assert code == 202, body
        booking = accepted_world["booking"]
        tr = accepted_world["transport_request"]
        db.session.refresh(booking)
        db.session.refresh(tr)
        assert booking.time_confirmed is True
        assert tr.pickup_time_confirmed is True

    def test_schema_then_service_same_as_browser_payload(self, db, accepted_world):
        """Même contrat que PATCH /institution/bookings/:id (schema + service)."""
        from routes.institution_bookings import InstitutionBookingPatchSchema

        raw = {
            "version": 1,
            "pickup_location": "Chemin des Courbes 9, 1247, Anières",
            "dropoff_location": (
                "Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4"
            ),
            "scheduled_time": "2026-09-12T13:15:00",
            "hospital_service": "Radiologie",
            "doctor_name": "",
            "destination_type": "medical",
            "appointment_time": "2026-09-12T13:00:00",
            "leg_appointments": [{"index": 0, "scheduled_time": "2026-09-12T13:00:00"}],
            "return_appointment_time": None,
            "medical_facility": None,
            "pickup_floor": None,
            "notes_medical": None,
            "wheelchair_need": False,
            "wheelchair_client_has": False,
        }
        validated = InstitutionBookingPatchSchema().load(raw)
        assert validated["appointment_time"] == "2026-09-12T13:00:00"
        assert validated["leg_appointments"][0]["scheduled_time"] == (
            "2026-09-12T13:00:00"
        )

        body, code = update_institution_booking(
            _ctx(accepted_world),
            payload=validated,
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin LHA",
        )
        assert code == 200, body
        dest = accepted_world["dest_leg"]
        booking = accepted_world["booking"]
        tr = accepted_world["transport_request"]
        db.session.refresh(dest)
        db.session.refresh(booking)
        db.session.refresh(tr)
        assert dest.scheduled_time == datetime(2026, 9, 12, 13, 0)
        assert booking.status == BookingStatus.ACCEPTED
        assert booking.time_confirmed is False
        assert tr.pickup_time_confirmed is False

    def test_service_only_does_not_invalidate(self, db, accepted_world):
        body, code = update_institution_booking(
            _ctx(accepted_world),
            payload=_base_payload(hospital_service="Cardiologie"),
            actor_user_id=None,
            actor_role=InstitutionRole.ADMIN.value,
            actor_display_name="Admin LHA",
        )
        assert code == 202, body
        booking = accepted_world["booking"]
        db.session.refresh(booking)
        assert booking.time_confirmed is True
        assert booking.hospital_service == "Radiologie"

    def test_http_patch_same_endpoint_as_request_detail_panel(
        self, db, client, accepted_world
    ):
        """PATCH /institutions/bookings/:id — même endpoint que Enregistrer."""
        from models import User, UserRole
        from tests.helpers.institution_auth import institution_bearer_headers

        institution = accepted_world["institution"]
        booking = accepted_world["booking"]
        dest = accepted_world["dest_leg"]
        tr = accepted_world["transport_request"]

        user = User()
        user.email = f"admin_{uuid.uuid4().hex[:8]}@lha.ch"
        user.username = user.email
        user.password = "test"
        user.role = UserRole.INSTITUTION.value
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.ADMIN.value
        user.public_id = str(uuid.uuid4())
        db.session.add(user)
        db.session.flush()

        headers = institution_bearer_headers(
            db,
            user,
            institution,
            institution_role=InstitutionRole.ADMIN.value,
        )
        payload = {
            "version": 1,
            "pickup_location": booking.pickup_location,
            "dropoff_location": booking.dropoff_location,
            "scheduled_time": "2026-09-12T13:15:00",
            "hospital_service": "Radiologie",
            "doctor_name": None,
            "destination_type": "medical",
            "appointment_time": "2026-09-12T13:00:00",
            "leg_appointments": [{"index": 0, "scheduled_time": "2026-09-12T13:00:00"}],
            "return_appointment_time": None,
        }
        response = client.patch(
            f"/api/v1/institutions/bookings/{booking.id}",
            json=payload,
            headers=headers,
        )
        assert response.status_code == 200, response.get_json()
        body = response.get_json()
        assert "Aucun champ modifié" not in str(body)
        db.session.refresh(dest)
        db.session.refresh(booking)
        db.session.refresh(tr)
        assert dest.scheduled_time == datetime(2026, 9, 12, 13, 0)
        assert booking.status == BookingStatus.ACCEPTED
        assert booking.time_confirmed is False
        assert tr.pickup_time_confirmed is False
