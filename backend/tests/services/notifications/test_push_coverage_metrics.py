"""Tests classification et couverture push chauffeur."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

from services.notifications.push_coverage_service import _resolve_push_status
from services.notifications.push_token_classification import classify_token


def _token(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "is_active": True,
        "provider": "expo",
        "token": "ExponentPushToken[abc]",
        "last_push_success_at": datetime.now(UTC),
        "last_seen_at": datetime.now(UTC),
        "last_push_error_code": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_classify_token_stale_when_no_success_and_no_seen() -> None:
    token = _token(last_push_success_at=None, last_seen_at=None)
    assert classify_token(token) == "STALE"


def test_resolve_push_status_no_token() -> None:
    assert _resolve_push_status(has_active_token=False, active_tokens=[]) == "no_token"


def test_resolve_push_status_token_invalid() -> None:
    tokens = [_token(last_push_error_code="token_unregistered")]
    assert _resolve_push_status(has_active_token=True, active_tokens=tokens) == "token_invalid"


def test_resolve_push_status_operational() -> None:
    tokens = [_token()]
    assert _resolve_push_status(has_active_token=True, active_tokens=tokens) == "operational"


def test_resolve_push_status_android_expo_only_unreliable() -> None:
    tokens = [
        _token(
            provider="expo",
            token="ExponentPushToken[abc]",
            platform="android",
        )
    ]
    assert (
        _resolve_push_status(has_active_token=True, active_tokens=tokens)
        == "expo_fallback_unreliable"
    )


def test_resolve_push_status_android_fcm_operational() -> None:
    tokens = [
        _token(
            provider="fcm",
            token="abc:APA91bNative",
            platform="android",
        )
    ]
    assert _resolve_push_status(has_active_token=True, active_tokens=tokens) == "operational"


def test_refresh_push_active_owners_gauges_noop_when_prometheus_disabled() -> None:
    from services.monitoring.prometheus import refresh_push_active_owners_gauges

    with patch("services.monitoring.prometheus.PROMETHEUS_AVAILABLE", False):
        refresh_push_active_owners_gauges()
