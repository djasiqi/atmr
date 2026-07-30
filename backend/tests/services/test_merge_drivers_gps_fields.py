"""Propagation des champs GPS dans merge_drivers_with_locations."""

from __future__ import annotations

from services.company_driver_locations import merge_drivers_with_locations


def test_merge_drivers_with_locations_propagates_gps_truth_fields():
    drivers = [{"id": 3, "status": "available", "full_name": "Jozsef"}]
    locations = [
        {
            "driver_id": 3,
            "lat": 46.19,
            "lon": 6.14,
            "location_status": "last_known",
            "presence_status": "offline",
            "tracking_display_status": "stale",
            "position_source": "db_fallback",
            "offline_reason": "location_stale",
            "recorded_at": "2026-07-29T09:47:47Z",
            "received_at": "2026-07-29T09:47:48Z",
            "timestamp": "2026-07-29T09:47:47Z",
            "status": "busy",
        }
    ]
    merged = merge_drivers_with_locations(drivers, locations)
    assert len(merged) == 1
    row = merged[0]
    assert row["position_source"] == "db_fallback"
    assert row["offline_reason"] == "location_stale"
    assert row["tracking_display_status"] == "stale"
    assert row["location_status"] == "last_known"
    assert row["presence_status"] == "offline"
    assert row["recorded_at"] == "2026-07-29T09:47:47Z"
    assert row["received_at"] == "2026-07-29T09:47:48Z"
    assert row["timestamp"] == "2026-07-29T09:47:47Z"
    assert row["status"] == "busy"
