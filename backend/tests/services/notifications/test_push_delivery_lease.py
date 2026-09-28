"""DELIVERY-01 à 05 : le claim n'est pas un envoi accepté."""

from __future__ import annotations

from unittest.mock import patch

from services.notifications.push_driver_booking_dedup import (
    begin_driver_booking_push,
    driver_booking_push_already_sent,
    mark_driver_booking_push_sent,
    release_driver_booking_push,
)


class _Redis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def set(
        self, key: str, value: str, nx: bool = False, ex: int | None = None
    ) -> bool:
        del ex
        if nx and key in self.store:
            return False
        self.store[key] = value
        return True

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


def _patch(redis: _Redis):
    return patch(
        "services.notifications.push_driver_booking_dedup._redis",
        return_value=redis,
    )


def test_delivery_01_failure_before_send_is_not_sent() -> None:
    redis = _Redis()
    with _patch(redis):
        assert begin_driver_booking_push(20135, 40500) == "claimed"
        release_driver_booking_push(20135, 40500)
        assert driver_booking_push_already_sent(20135, 40500) is False
        assert begin_driver_booking_push(20135, 40500) == "claimed"


def test_delivery_02_retry_after_presend_failure_can_send() -> None:
    redis = _Redis()
    with _patch(redis):
        assert begin_driver_booking_push(20135, 40500) == "claimed"
        release_driver_booking_push(20135, 40500)
        assert begin_driver_booking_push(20135, 40500) == "claimed"
        mark_driver_booking_push_sent(20135, 40500)
        assert driver_booking_push_already_sent(20135, 40500) is True


def test_delivery_03_provider_accept_marks_sent() -> None:
    redis = _Redis()
    with _patch(redis):
        assert begin_driver_booking_push(20135, 40500) == "claimed"
        mark_driver_booking_push_sent(20135, 40500)
        assert driver_booking_push_already_sent(20135, 40500) is True


def test_delivery_04_same_event_after_sent_is_skipped() -> None:
    redis = _Redis()
    with _patch(redis):
        mark_driver_booking_push_sent(20135, 40500)
        assert begin_driver_booking_push(20135, 40500) == "sent"
        assert driver_booking_push_already_sent(20135, 40500) is True


def test_delivery_05_concurrent_claims_one_winner() -> None:
    redis = _Redis()
    with _patch(redis):
        assert begin_driver_booking_push(20135, 40500) == "claimed"
        assert begin_driver_booking_push(20135, 40500) == "busy"
        mark_driver_booking_push_sent(20135, 40500)
        assert begin_driver_booking_push(20135, 40500) == "sent"


def test_delivery_fail_open_without_redis() -> None:
    with patch(
        "services.notifications.push_driver_booking_dedup._redis",
        return_value=None,
    ):
        assert begin_driver_booking_push(20135, 40500) == "open"
        assert driver_booking_push_already_sent(20135, 40500) is False
