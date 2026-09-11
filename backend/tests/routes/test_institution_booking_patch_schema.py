"""Validation PATCH booking institution — leg_appointments + dest. médicale."""

import pytest
from marshmallow import ValidationError

from application.institutions.destination_details_rules import (
    MEDICAL_DESTINATION_OR_ERROR,
)
from routes.institution_bookings import InstitutionBookingPatchSchema


def test_leg_appointments_scheduled_time_null_allowed():
    """RDV « À définir » : scheduled_time null doit passer la validation."""
    payload = {
        "version": 1,
        "pickup_location": "Chemin des Courbes 9, 1247, Anières",
        "dropoff_location": "HUG, Genève",
        "scheduled_time": "2026-07-22T10:00:00",
        "leg_appointments": [{"index": 0, "scheduled_time": None}],
        "return_appointment_time": None,
    }
    result = InstitutionBookingPatchSchema().load(payload)
    assert result["leg_appointments"][0]["index"] == 0
    assert result["leg_appointments"][0]["scheduled_time"] is None


def test_leg_appointments_scheduled_time_iso_ok():
    payload = {
        "version": 1,
        "leg_appointments": [
            {"index": 0, "scheduled_time": "2026-07-22T11:30:00+02:00"},
        ],
    }
    result = InstitutionBookingPatchSchema().load(payload)
    assert (
        result["leg_appointments"][0]["scheduled_time"] == "2026-07-22T11:30:00+02:00"
    )


def test_other_destination_allows_both_empty():
    loaded = InstitutionBookingPatchSchema().load(
        {
            "version": 1,
            "destination_type": "other",
            "hospital_service": "",
            "doctor_name": "",
        }
    )
    assert loaded["destination_type"] == "other"


def test_omitted_destination_type_allows_both_empty():
    loaded = InstitutionBookingPatchSchema().load(
        {
            "version": 1,
            "hospital_service": "",
            "doctor_name": "",
        }
    )
    assert loaded["hospital_service"] == ""


def test_medical_destination_rejects_both_empty():
    schema = InstitutionBookingPatchSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(
            {
                "version": 1,
                "destination_type": "medical",
                "hospital_service": "",
                "doctor_name": "",
            }
        )
    assert MEDICAL_DESTINATION_OR_ERROR in str(exc.value)


def test_medical_destination_accepts_service_only():
    loaded = InstitutionBookingPatchSchema().load(
        {
            "version": 1,
            "destination_type": "medical",
            "hospital_service": "Radiologie",
            "doctor_name": "",
        }
    )
    assert loaded["hospital_service"] == "Radiologie"
    assert loaded["doctor_name"] == ""


def test_medical_destination_accepts_doctor_only():
    loaded = InstitutionBookingPatchSchema().load(
        {
            "version": 1,
            "destination_type": "medical",
            "hospital_service": "",
            "doctor_name": "Dr Martin",
        }
    )
    assert loaded["doctor_name"] == "Dr Martin"


def test_medical_destination_accepts_both():
    loaded = InstitutionBookingPatchSchema().load(
        {
            "version": 1,
            "destination_type": "medical",
            "hospital_service": "Cardiologie",
            "doctor_name": "Dr Martin",
        }
    )
    assert loaded["hospital_service"] == "Cardiologie"
    assert loaded["doctor_name"] == "Dr Martin"


def test_notes_only_does_not_require_service_or_doctor():
    loaded = InstitutionBookingPatchSchema().load(
        {"version": 1, "notes_medical": "RAS"}
    )
    assert loaded["notes_medical"] == "RAS"
