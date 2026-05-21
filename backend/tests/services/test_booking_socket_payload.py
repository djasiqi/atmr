from __future__ import annotations

from unittest.mock import patch

from services.events.booking_socket_payload import maybe_shrink_booking_socket_payload


def test_lite_payload_filters_extra_keys_when_enabled():
    data = {
        "id": 1,
        "status": "ASSIGNED",
        "updated_at": "2026-05-20T10:00:00Z",
        "heavy_nested": {"foo": "bar"},
        "notes": "x" * 5000,
    }
    with patch(
        "services.events.booking_socket_payload.BOOKING_SOCKET_LITE_PAYLOAD",
        True,
    ):
        out = maybe_shrink_booking_socket_payload(data, "booking_updated")
    assert "heavy_nested" not in out
    assert "notes" not in out
    assert out["id"] == 1


def test_full_payload_when_disabled():
    data = {"id": 1, "notes": "keep"}
    out = maybe_shrink_booking_socket_payload(data, "booking_updated")
    assert out == data
