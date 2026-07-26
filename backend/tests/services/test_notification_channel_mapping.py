"""Tests mapping canaux Android push chauffeur."""

from __future__ import annotations

import pytest

from services.events.fanout import _get_notification_channel

LEGACY_CHANNELS = frozenset({"missions", "missions_v2", "critical", "messages", "info"})


@pytest.mark.parametrize(
    ("notification_type", "expected"),
    [
        ("booking", "mission_updates"),
        ("booking_assigned", "mission_updates"),
        ("booking_updated", "mission_updates"),
        ("booking_reassigned", "mission_updates"),
        ("delay", "mission_updates"),
        ("booking_cancelled", "urgent"),
        ("urgent_alert", "urgent"),
        ("accident", "urgent"),
        ("message", "chat"),
        ("team_chat_message", "chat"),
        ("dispatch_completed", "default"),
        ("stats", "default"),
        ("info", "default"),
        ("unknown_type", "mission_updates"),
    ],
)
def test_get_notification_channel_aligns_with_mobile(
    notification_type: str, expected: str
) -> None:
    assert _get_notification_channel(notification_type) == expected
    assert _get_notification_channel(notification_type) not in LEGACY_CHANNELS


def test_get_notification_channel_production_no_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.delenv("PUSH_PROOF", raising=False)
    for notification_type in (
        "booking",
        "booking_assigned",
        "booking_updated",
        "booking_cancelled",
    ):
        channel = _get_notification_channel(notification_type)
        assert channel not in LEGACY_CHANNELS
        assert channel in {"mission_updates", "urgent", "chat", "default"}
