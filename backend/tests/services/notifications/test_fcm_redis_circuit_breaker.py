"""Tests circuit breaker FCM Redis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.notifications.fcm_redis_circuit_breaker import (
    allow_fcm_request,
    record_fcm_retryable_failure,
    record_fcm_success,
)


def test_allow_fcm_request_fail_open_without_redis():
    with patch("services.notifications.fcm_redis_circuit_breaker._get_redis", return_value=None):
        assert allow_fcm_request() is True


def test_allow_fcm_request_blocked_when_open():
    mock_redis = MagicMock()
    mock_redis.get.return_value = b"1"

    with patch("services.notifications.fcm_redis_circuit_breaker._get_redis", return_value=mock_redis):
        assert allow_fcm_request() is False


def test_record_fcm_success_clears_failures():
    mock_redis = MagicMock()

    with patch("services.notifications.fcm_redis_circuit_breaker._get_redis", return_value=mock_redis):
        record_fcm_success()

    mock_redis.delete.assert_called_once()


def test_record_fcm_retryable_failure_opens_circuit():
    mock_redis = MagicMock()
    mock_redis.incr.return_value = 5

    with patch(
        "services.notifications.fcm_redis_circuit_breaker._get_redis",
        return_value=mock_redis,
    ), patch(
        "services.notifications.fcm_redis_circuit_breaker.FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        5,
    ):
        record_fcm_retryable_failure()

    mock_redis.setex.assert_called_once()
