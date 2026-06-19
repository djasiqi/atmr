"""Tests verify_fcm_token_coverage."""

from __future__ import annotations

from scripts.verify_fcm_token_coverage import (
    FCM_COVERAGE_ANDROID_EXPO_ONLY,
    FCM_COVERAGE_FCM_NATIVE_OK,
    FCM_COVERAGE_IOS_ONLY,
    FCM_COVERAGE_NO_ACTIVE_TOKEN,
    resolve_fcm_coverage,
)


class _TokenStub:
    def __init__(self, **kwargs):
        self.provider = kwargs.get("provider", "expo")
        self.platform = kwargs.get("platform", "android")
        self.token = kwargs.get("token", "ExponentPushToken[abc]")
        self.is_active = kwargs.get("is_active", True)
        self.id = kwargs.get("id", 1)
        self.driver_id = kwargs.get("driver_id", 7514)
        self.last_seen_at = kwargs.get("last_seen_at")
        self.last_push_success_at = kwargs.get("last_push_success_at")
        self.last_push_failure_at = kwargs.get("last_push_failure_at")
        self.consecutive_push_failures = kwargs.get("consecutive_push_failures", 0)
        self.updated_at = kwargs.get("updated_at")


def test_resolve_fcm_coverage_no_active_token() -> None:
    assert resolve_fcm_coverage([]) == FCM_COVERAGE_NO_ACTIVE_TOKEN


def test_resolve_fcm_coverage_android_expo_only() -> None:
    tokens = [
        _TokenStub(provider="expo", platform="android", token="ExponentPushToken[xyz]"),
    ]
    assert resolve_fcm_coverage(tokens) == FCM_COVERAGE_ANDROID_EXPO_ONLY


def test_resolve_fcm_coverage_fcm_native_ok() -> None:
    tokens = [
        _TokenStub(
            provider="fcm",
            platform="android",
            token="FakeFcmInstanceId:APA91bTestRegistrationToken",
        ),
    ]
    assert resolve_fcm_coverage(tokens) == FCM_COVERAGE_FCM_NATIVE_OK


def test_resolve_fcm_coverage_ios_only() -> None:
    tokens = [
        _TokenStub(provider="fcm", platform="ios", token="ios-apns-like-token"),
    ]
    assert resolve_fcm_coverage(tokens) == FCM_COVERAGE_IOS_ONLY


def test_fcm_android_preferred_over_expo() -> None:
    tokens = [
        _TokenStub(provider="expo", platform="android", token="ExponentPushToken[xyz]"),
        _TokenStub(
            provider="fcm",
            platform="android",
            token="FakeFcmInstanceId:APA91bTestRegistrationToken",
        ),
    ]
    assert resolve_fcm_coverage(tokens) == FCM_COVERAGE_FCM_NATIVE_OK
