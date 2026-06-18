"""Tests normalisation horaires — schemas institution."""

from schemas.institution_schemas import (
    TransportRequestCreateSchema,
    normalize_transport_request_schedule_payload,
)


def test_normalize_empty_scheduled_time_to_none():
    raw = normalize_transport_request_schedule_payload(
        {
            "mission_date": "2026-06-17",
            "scheduled_time": "",
            "return_time": "",
        }
    )
    assert raw["scheduled_time"] is None
    assert raw["return_time"] is None


def test_normalize_hhmm_stop_with_mission_date():
    raw = normalize_transport_request_schedule_payload(
        {
            "mission_date": "2026-06-17",
            "intermediate_stops": [
                {"dropoff_location": "HUG", "scheduled_time": "09:00"},
            ],
        }
    )
    assert raw["intermediate_stops"][0]["scheduled_time"] == "2026-06-17T09:00:00"


def test_create_schema_accepts_payload_after_normalization():
    schema = TransportRequestCreateSchema()
    data = {
        "mission_date": "2026-06-17",
        "pickup_location": "Clinique",
        "dropoff_location": "HUG",
        "multi_stop": True,
        "return_to_institution": True,
        "intermediate_stops": [{"dropoff_location": "HUG", "scheduled_time": ""}],
        "return_time": "",
    }
    normalized = normalize_transport_request_schedule_payload(data)
    loaded = schema.load(normalized)
    assert loaded.get("scheduled_time") is None
    assert loaded.get("return_time") is None
