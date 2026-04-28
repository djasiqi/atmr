"""Tests unitaires promotion guest Saferpay (idempotence, notify_key) sans appel API."""

from __future__ import annotations

import pytest

from services.saferpay.assert_response_status import (
    SAFERPAY_FINALIZE_ALREADY_COMPLETED,
)


def test_promote_guest_redis_already_promoted(app):
    calls: list[tuple[str, int, str]] = []

    def fake_setex(key: str, ttl: int, value: str) -> None:
        calls.append((key, ttl, value))

    with app.app_context():
        from services.guest_saferpay import promote_guest_booking_after_saferpay

        out = promote_guest_booking_after_saferpay(
            guest_booking_id="gb_unit_1",
            payload={
                "promoted_booking_id": 99,
                "public_status_token": "tok_x",
            },
            redis_setex=fake_setex,
        )

    assert out["status"] == SAFERPAY_FINALIZE_ALREADY_COMPLETED
    assert out["booking_id"] == 99
    assert out["public_status_token"] == "tok_x"
    assert calls == []


def test_promote_guest_bad_notify_key(app):
    def fake_setex(*_a, **_k):
        raise AssertionError("redis_setex ne doit pas être appelée")

    with app.app_context():
        from services.guest_saferpay import promote_guest_booking_after_saferpay

        out = promote_guest_booking_after_saferpay(
            guest_booking_id="gb_unit_2",
            payload={
                "saferpay_notify_key": "expected",
                "saferpay_token": "sess",
            },
            redis_setex=fake_setex,
            notify_key="wrong",
        )

    assert out["status"] == "forbidden"
    assert out.get("detail") == "bad_notify_key"


def test_promote_guest_missing_session_token(app):
    with app.app_context():
        from services.guest_saferpay import promote_guest_booking_after_saferpay

        with pytest.raises(ValueError, match="Session Saferpay"):
            promote_guest_booking_after_saferpay(
                guest_booking_id="gb_unit_3",
                payload={},
                redis_setex=lambda *_a, **_k: None,
            )
