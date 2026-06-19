"""Tests push_token_platform et push_device_selection."""

from __future__ import annotations

from types import SimpleNamespace

from services.notifications.push_device_selection import (
    android_has_expo_only,
    prepare_driver_push_targets,
    prioritize_android_fcm_devices,
)
from services.notifications.push_token_platform import (
    infer_fcm_platform,
    is_android_fcm_registration_token,
    looks_like_fcm_token,
)


def test_looks_like_fcm_token_modern_prefix_format() -> None:
    token = "FakeFcmInstanceId:APA91bTestRegistrationToken_9I-ZK9iUTWXRY"
    assert looks_like_fcm_token(token) is True
    assert is_android_fcm_registration_token(token) is True


def test_infer_fcm_platform_ios_to_android() -> None:
    token = "prefix:APA91bFakeToken"
    assert infer_fcm_platform(token, "ios") == "android"


def test_prioritize_android_fcm_over_expo() -> None:
    devices = [
        {
            "id": 1,
            "token": "ExponentPushToken[abc]",
            "device_id": "dev-1",
            "platform": "android",
            "provider": "expo",
        },
        {
            "id": 2,
            "token": "abc:APA91bNative",
            "device_id": "dev-1",
            "platform": "android",
            "provider": "fcm",
        },
    ]
    selected = prioritize_android_fcm_devices(devices, driver_id=7514)
    assert len(selected) == 1
    assert selected[0]["provider"] == "fcm"


def test_prioritize_android_expo_fallback_when_no_fcm() -> None:
    devices = [
        {
            "id": 1,
            "token": "ExponentPushToken[abc]",
            "device_id": "dev-1",
            "platform": "android",
            "provider": "expo",
        },
    ]
    selected = prioritize_android_fcm_devices(devices, driver_id=7514)
    assert len(selected) == 1
    assert selected[0]["provider"] == "expo"


def test_prepare_driver_push_targets_from_orm_rows() -> None:
    rows = [
        SimpleNamespace(
            id=55,
            token="ExponentPushToken[abc]",
            device_id="dev-1",
            platform="android",
            provider="expo",
            updated_at=None,
        ),
        SimpleNamespace(
            id=56,
            token="xyz:APA91bNative",
            device_id="dev-1",
            platform="android",
            provider="fcm",
            updated_at=None,
        ),
    ]
    targets = prepare_driver_push_targets(rows, driver_id=7514)
    assert len(targets) == 1
    assert targets[0]["provider"] == "fcm"


def test_android_has_expo_only() -> None:
    tokens = [
        SimpleNamespace(platform="android", provider="expo"),
    ]
    assert android_has_expo_only(tokens) is True

    tokens_with_fcm = [
        SimpleNamespace(platform="android", provider="expo"),
        SimpleNamespace(platform="android", provider="fcm"),
    ]
    assert android_has_expo_only(tokens_with_fcm) is False
