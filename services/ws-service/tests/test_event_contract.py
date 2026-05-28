from event_contract import CRITICAL_EVENT_TYPES, event_criticality


def test_critical_events():
    assert event_criticality("booking_updated") == "critical"
    assert event_criticality("team_chat_message") == "critical"
    assert "booking_updated" in CRITICAL_EVENT_TYPES


def test_gps_not_critical():
    assert event_criticality("driver_location_update") == "high"
