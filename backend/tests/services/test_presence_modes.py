from services.geolocation.presence import (
    compute_last_seen_seconds,
    compute_location_status,
    normalize_location_mode,
    presence_status_from_location_status,
)


def test_normalize_location_mode_defaults_to_mission_live():
    assert normalize_location_mode(None) == "mission_live"
    assert normalize_location_mode("unknown") == "mission_live"


def test_compute_location_status_for_mission_live_thresholds():
    assert compute_location_status(mode="mission_live", last_seen_seconds=10) == "live"
    assert compute_location_status(mode="mission_live", last_seen_seconds=50) == "recent"
    assert compute_location_status(mode="mission_live", last_seen_seconds=150) == "stale"
    assert compute_location_status(mode="mission_live", last_seen_seconds=301) == "offline"


def test_compute_location_status_for_availability_presence_thresholds():
    assert (
        compute_location_status(mode="availability_presence", last_seen_seconds=70)
        == "live"
    )
    assert (
        compute_location_status(mode="availability_presence", last_seen_seconds=180)
        == "recent"
    )
    assert (
        compute_location_status(mode="availability_presence", last_seen_seconds=700)
        == "stale"
    )
    assert (
        compute_location_status(mode="availability_presence", last_seen_seconds=901)
        == "offline"
    )


def test_presence_status_mapping_is_canonical():
    assert presence_status_from_location_status("live") == "online"
    assert presence_status_from_location_status("recent") == "online"
    assert presence_status_from_location_status("stale") == "degraded"
    assert presence_status_from_location_status("offline") == "offline"


def test_compute_last_seen_seconds_handles_invalid_iso():
    assert compute_last_seen_seconds("invalid-date") is None
