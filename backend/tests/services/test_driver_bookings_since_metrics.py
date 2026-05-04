"""Métriques GET /driver/me/bookings/since (trigger header)."""

from services.monitoring import driver_booking_metrics as m


def test_normalize_bookings_since_trigger_unknown_when_missing() -> None:
    assert m.normalize_bookings_since_trigger(None) == "unknown"
    assert m.normalize_bookings_since_trigger("") == "unknown"


def test_normalize_bookings_since_trigger_known() -> None:
    assert m.normalize_bookings_since_trigger("socket_connect") == "socket_connect"


def test_normalize_bookings_since_trigger_garbage_to_unknown() -> None:
    assert m.normalize_bookings_since_trigger("not_a_real_trigger") == "unknown"


def test_observe_accepts_normalized_trigger() -> None:
    m.observe_driver_bookings_since_request(
        trigger_reason="foreground", duration_seconds=0.01
    )
