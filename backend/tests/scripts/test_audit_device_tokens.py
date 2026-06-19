"""Tests classification tokens push (audit_device_tokens)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.audit_device_tokens import (
    _provider_mismatch,
    classify_token,
)
from services.notifications.push_token_platform import looks_like_fcm_token


class _TokenStub:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", 1)
        self.driver_id = kwargs.get("driver_id", 3)
        self.provider = kwargs.get("provider", "expo")
        self.platform = kwargs.get("platform", "android")
        self.token = kwargs.get("token", "ExponentPushToken[abc]")
        self.is_active = kwargs.get("is_active", True)
        self.last_seen_at = kwargs.get("last_seen_at")
        self.last_push_success_at = kwargs.get("last_push_success_at")
        self.last_push_failure_at = kwargs.get("last_push_failure_at")
        self.consecutive_push_failures = kwargs.get("consecutive_push_failures", 0)


def test_classify_inactive_token() -> None:
    token = _TokenStub(is_active=False)
    assert classify_token(token) == "INACTIVE"


def test_classify_mismatch_provider_fcm_as_expo() -> None:
    token = _TokenStub(
        provider="expo",
        token="APA91bFakeFcmTokenValue",
    )
    assert looks_like_fcm_token(token.token) is True
    assert _provider_mismatch(token) is True
    assert classify_token(token) == "MISMATCH_PROVIDER"


def test_modern_fcm_token_detected() -> None:
    token = "ewCrKUSCKU5bvnMqoZemWw:APA91bENAkBia"
    assert looks_like_fcm_token(token) is True


def test_classify_healthy_recent_push() -> None:
    now = datetime.now(UTC)
    token = _TokenStub(last_push_success_at=now - timedelta(days=1))
    assert classify_token(token, now=now) == "HEALTHY"


def test_classify_stale_without_recent_activity() -> None:
    now = datetime.now(UTC)
    token = _TokenStub(
        last_push_success_at=now - timedelta(days=30),
        last_seen_at=now - timedelta(days=30),
    )
    assert classify_token(token, now=now) == "STALE"
