"""Parité portail particulier : le contrat doit rester exploitable sans rappel.

Cas couverts : dès que possible, rendez-vous, heure de départ, aller-retour
avec et sans heure, fauteuils exclusifs, assistance seule ou combinée,
destination médicale, accès départ/destination, récurrence.
"""

from __future__ import annotations

import pytest
from marshmallow import ValidationError

from schemas.booking_schemas import BookingCreateSchema
from schemas.validation_utils import validate_request
from shared.client_portal_notes import compose_client_portal_notes_medical
from shared.portal_client_booking_contract import (
    classify_destination_contact,
    decide_portal_schedule,
    decide_return_schedule,
    scrub_access_from_free_note,
)

_BASE = {
    "customer_name": "Ada Martin",
    "pickup_location": "Rue du Test 1",
    "dropoff_location": "HUG",
    "amount": 40.0,
}


def _load(**overrides):
    data = {**_BASE, **overrides}
    return validate_request(BookingCreateSchema(), data)


def test_asap_does_not_invent_a_clock():
    loaded = _load(asap=True, scheduled_time="2026-09-30T09:00:00")
    decision = decide_portal_schedule(loaded)
    assert loaded["scheduled_time"] is None
    assert loaded["is_urgent"] is True
    assert decision["kind"] == "asap"
    assert decision["scheduled_time_raw"] is None
    assert decision["time_confirmed"] is False


def test_arrival_keeps_appointment_unconfirmed_as_pickup():
    loaded = _load(
        scheduled_time="2026-09-30T09:00:00",
        scheduled_time_type="arrival",
    )
    decision = decide_portal_schedule(loaded)
    assert decision["kind"] == "arrival"
    assert decision["scheduled_time_raw"] == "2026-09-30T09:00:00"
    assert decision["time_confirmed"] is False
    assert decision["is_urgent"] is False


def test_departure_confirms_pickup_time():
    loaded = _load(
        scheduled_time="2026-09-30T08:15:00",
        scheduled_time_type="departure",
    )
    decision = decide_portal_schedule(loaded)
    assert decision["kind"] == "departure"
    assert decision["time_confirmed"] is True
    assert decision["scheduled_time_raw"] == "2026-09-30T08:15:00"


def test_round_trip_with_return_time_is_exact():
    loaded = _load(
        scheduled_time="2026-09-30T08:15:00",
        is_round_trip=True,
        return_date="2026-09-30",
        return_time="2026-09-30T16:30:00",
    )
    plan = decide_return_schedule(loaded)
    assert plan["return_time_exact"] is True
    assert plan["return_time_raw"] == "2026-09-30T16:30:00"


def test_round_trip_without_return_time_stays_null():
    loaded = _load(
        scheduled_time="2026-09-30T08:15:00",
        is_round_trip=True,
        return_date="2026-09-30",
        return_time="   ",
    )
    assert loaded["return_time"] is None
    plan = decide_return_schedule(loaded)
    assert plan["return_time_raw"] is None
    assert plan["return_time_exact"] is False
    assert plan["return_date"] == "2026-09-30"


def test_personal_and_provided_wheelchair_are_exclusive():
    with pytest.raises(ValidationError):
        _load(
            scheduled_time="2026-09-30T08:15:00",
            wheelchair_client_has=True,
            wheelchair_need=True,
        )


def test_assistance_alone_and_with_wheelchair():
    alone = _load(
        scheduled_time="2026-09-30T08:15:00",
        needs_assistance=True,
    )
    assert alone["needs_assistance"] is True
    assert alone["wheelchair_client_has"] is False
    assert alone["wheelchair_need"] is False

    with_chair = _load(
        scheduled_time="2026-09-30T08:15:00",
        wheelchair_need=True,
        needs_assistance=True,
    )
    assert with_chair["wheelchair_need"] is True
    assert with_chair["needs_assistance"] is True


def test_medical_destination_contact_is_structured():
    classified = classify_destination_contact("Radiologie – Dr Martin")
    assert classified["destination_contact_detail"] == "Radiologie – Dr Martin"
    assert classified["hospital_service"] == "Radiologie"
    assert classified["doctor_name"] == "Dr Martin"

    ambiguous = classify_destination_contact("Bâtiment B – étage 3")
    assert ambiguous["hospital_service"] == "Bâtiment B – étage 3"
    assert ambiguous["doctor_name"] == ""
    assert ambiguous["destination_contact_detail"] == "Bâtiment B – étage 3"


def test_access_stays_out_of_the_free_note():
    loaded = _load(
        scheduled_time="2026-09-30T09:00:00",
        scheduled_time_type="arrival",
        pickup_access_notes="Code 1234",
        dropoff_access_notes="Accueil radiologie",
        client_note="Prise en charge : Code 1234\nDestination : Accueil radiologie",
    )
    note = scrub_access_from_free_note(loaded)
    composed = compose_client_portal_notes_medical({**loaded, "client_note": note})
    assert "Code 1234" not in (composed or "")
    assert "Accueil radiologie" not in (composed or "")
    assert loaded["pickup_access_notes"] == "Code 1234"
    assert loaded["dropoff_access_notes"] == "Accueil radiologie"


def test_recurrence_remains_a_secondary_note():
    loaded = _load(
        scheduled_time="2026-09-30T08:15:00",
        is_recurring=True,
        recurrence_type="weekly",
        recurrence_series_length=4,
    )
    composed = compose_client_portal_notes_medical(loaded) or ""
    assert "Récurrence demandée" in composed
    assert loaded["is_recurring"] is True


def test_company_reading_ignores_notes():
    from shared.portal_client_booking_contract import read_company_portal_schedule

    misleading = "Horaire souhaité : rendez-vous à destination. La prise en charge est à proposer."
    asap = read_company_portal_schedule(
        {
            "is_urgent": True,
            "time_confirmed": False,
            "scheduled_time": None,
            "notes_medical": misleading,
        }
    )
    assert asap["kind"] == "asap"
    assert asap["label"] == "Dès que possible"
    assert asap["scheduled_time"] is None
    assert asap["time_confirmed"] is False

    appointment = read_company_portal_schedule(
        {
            "is_urgent": False,
            "is_return": False,
            "time_confirmed": False,
            "scheduled_time": "2026-09-30T07:00:00Z",
            "notes_medical": "note libre",
        }
    )
    assert appointment["kind"] == "appointment"
    assert appointment["label"] == "Rendez-vous — prise en charge à proposer"
    assert appointment["scheduled_time"] == "2026-09-30T07:00:00Z"
    assert appointment["time_confirmed"] is False

    departure = read_company_portal_schedule(
        {
            "is_urgent": False,
            "time_confirmed": True,
            "scheduled_time": "2026-09-30T06:15:00Z",
            "notes_medical": misleading,
        }
    )
    assert departure["kind"] == "departure"
    assert departure["scheduled_time"] == "2026-09-30T06:15:00Z"


def test_manual_and_institution_clocks_are_not_rewritten():
    from schemas.company_schemas import ManualBookingCreateSchema
    from schemas.institution_schemas import (
        TransportRequestCreateSchema,
        normalize_transport_request_schedule_payload,
    )

    manual = validate_request(
        ManualBookingCreateSchema(),
        {
            "client_id": 123,
            "pickup_location": "Rue du Test 1",
            "dropoff_location": "HUG",
            "scheduled_time": "2026-09-30T08:15:00Z",
            "is_urgent": True,
        },
    )
    assert manual["scheduled_time"] == "2026-09-30T08:15:00Z"

    with pytest.raises(ValidationError) as exc:
        validate_request(
            ManualBookingCreateSchema(),
            {
                "client_id": 123,
                "pickup_location": "Rue du Test 1",
                "dropoff_location": "HUG",
            },
        )
    assert "scheduled_time" in str(exc.value.messages)

    institution = TransportRequestCreateSchema().load(
        normalize_transport_request_schedule_payload(
            {
                "mission_date": "2026-09-30",
                "pickup_location": "Clinique",
                "dropoff_location": "HUG",
                "scheduled_time": "2026-09-30T08:15:00+02:00",
                "scheduled_time_type": "departure",
                "pickup_time_confirmed": True,
                "asap": True,
            }
        )
    )
    assert institution["scheduled_time"] == "2026-09-30T08:15:00+02:00"
    assert institution["scheduled_time_type"] == "departure"
    assert "asap" not in institution


def test_medical_destination_requires_establishment_and_filled_text():
    with pytest.raises(ValidationError):
        _load(
            scheduled_time="2026-09-30T09:00:00",
            medical_destination=True,
            medical_facility="HUG",
        )
    loaded = _load(
        scheduled_time="2026-09-30T09:00:00",
        scheduled_time_type="arrival",
        medical_destination=True,
        medical_facility="HUG",
        medical_destination_detail="Bâtiment B – étage 3",
    )
    classified = classify_destination_contact(loaded["medical_destination_detail"])
    assert classified["doctor_name"] == ""
    assert classified["hospital_service"] == "Bâtiment B – étage 3"
    assert classified["destination_contact_detail"] == "Bâtiment B – étage 3"


def test_medical_detail_splits_only_a_real_doctor():
    classified = classify_destination_contact("Radiologie – Dr Martin")
    assert classified["hospital_service"] == "Radiologie"
    assert classified["doctor_name"] == "Dr Martin"
    assert classified["destination_contact_detail"] == "Radiologie – Dr Martin"


def test_extra_stop_reuses_the_schedule_contract_without_inventing_a_clock():
    from shared.portal_client_booking_contract import prepare_portal_extra_stop

    asap = prepare_portal_extra_stop(
        {
            "address": "Clinique La Colline",
            "asap": True,
            "medical_destination": True,
            "medical_facility": "Clinique La Colline",
            "medical_destination_detail": "Dr Martin",
        }
    )
    assert asap["schedule"]["kind"] == "asap"
    assert asap["schedule"]["scheduled_time_raw"] is None
    assert asap["schedule"]["time_confirmed"] is False
    assert asap["doctor_name"] == "Dr Martin"

    with pytest.raises(ValueError, match="date et une heure"):
        prepare_portal_extra_stop(
            {
                "address": "HUG",
                "scheduled_time_type": "arrival",
                "medical_destination": True,
                "medical_facility": "HUG",
                "medical_destination_detail": "Radiologie",
            }
        )
