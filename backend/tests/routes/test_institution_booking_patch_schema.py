"""Validation PATCH booking institution — leg_appointments."""

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
    assert result["leg_appointments"][0]["scheduled_time"] == "2026-07-22T11:30:00+02:00"
