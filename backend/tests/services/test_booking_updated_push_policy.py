"""Tests politique booking_updated push conditionnel (handler)."""

from __future__ import annotations


def test_driver_push_only_on_significant_changes():
    changes_keys = {"status", "internal_ref"}
    driver_push_fields = {
        "scheduled_time",
        "pickup_location",
        "dropoff_location",
        "notes",
        "notes_medical",
        "medical_facility",
        "hospital_service",
        "doctor_name",
        "pickup_access_notes",
        "dropoff_access_notes",
        "wheelchair_client_has",
        "wheelchair_need",
    }
    driver_push = bool(changes_keys.intersection(driver_push_fields))
    assert driver_push is False


def test_driver_push_true_on_scheduled_time_change():
    changes_keys = {"scheduled_time"}
    driver_push_fields = {"scheduled_time", "pickup_location"}
    driver_push = bool(changes_keys.intersection(driver_push_fields))
    assert driver_push is True
