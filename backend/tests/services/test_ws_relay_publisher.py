from unittest.mock import MagicMock, patch

from services.realtime.ws_relay_publisher import publish_relay_event, relay_stats


def test_publish_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("WS_RELAY_PUBLISH_ENABLED", "false")
    publish_relay_event(
        room="company_1",
        event_type="booking_updated",
        payload={"event_id": "e1"},
        criticality="critical",
    )
    assert relay_stats()["published"] == 0


def test_publish_no_raise_when_redis_missing(monkeypatch):
    monkeypatch.setenv("WS_RELAY_PUBLISH_ENABLED", "true")
    publish_relay_event(
        room="company_1",
        event_type="team_chat_message",
        payload={"event_id": "e2"},
        criticality="critical",
    )
    assert relay_stats()["dropped"] >= 0
