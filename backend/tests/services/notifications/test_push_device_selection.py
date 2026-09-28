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
    # :APA91 n'identifie pas Android : les tokens FCM iOS ont la même forme.
    assert is_android_fcm_registration_token(token) is False


def test_infer_fcm_platform_keeps_registered_ios() -> None:
    token = "prefix:APA91bFakeToken"
    assert infer_fcm_platform(token, "ios") == "ios"
    assert infer_fcm_platform(token, "android") == "android"


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


def test_prepare_driver_push_targets_skips_expo_when_fcm_on_other_device_id() -> None:
    """Même téléphone, device_id roté : un seul push FCM (pas Expo + FCM)."""
    rows = [
        SimpleNamespace(
            id=55,
            token="ExponentPushToken[abc]",
            device_id="dev-old",
            platform="android",
            provider="expo",
            updated_at=None,
        ),
        SimpleNamespace(
            id=56,
            token="xyz:APA91bNative",
            device_id="dev-new",
            platform="android",
            provider="fcm",
            updated_at=None,
        ),
    ]
    targets = prepare_driver_push_targets(rows, driver_id=7514)
    assert len(targets) == 1
    assert targets[0]["provider"] == "fcm"
    assert targets[0]["device_id"] == "dev-new"


def test_prepare_driver_push_targets_single_latest_android_fcm() -> None:
    from datetime import UTC, datetime

    rows = [
        SimpleNamespace(
            id=56,
            token="fcm-old",
            device_id="dev-old",
            platform="android",
            provider="fcm",
            updated_at=datetime(2026, 6, 1, tzinfo=UTC),
        ),
        SimpleNamespace(
            id=57,
            token="fcm-new",
            device_id="dev-new",
            platform="android",
            provider="fcm",
            updated_at=datetime(2026, 6, 21, tzinfo=UTC),
        ),
    ]
    targets = prepare_driver_push_targets(rows, driver_id=7514)
    assert len(targets) == 1
    assert targets[0]["id"] == 57


def _ios_row(row_id: int, device_id: str, provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id,
        token=f"token-{row_id}",
        device_id=device_id,
        platform="ios",
        provider=provider,
        updated_at=None,
    )


def test_device_provider_01_ios_fcm_preferred_over_expo() -> None:
    """DEVICE-PROVIDER-01 : même appareil, Expo + FCM iOS => un seul FCM."""
    rows = [
        _ios_row(824, "atmr-1787520916767-a4kexsnra", "expo"),
        _ios_row(825, "atmr-1787520916767-a4kexsnra", "fcm"),
    ]
    targets = prepare_driver_push_targets(rows, driver_id=20135)
    assert len(targets) == 1
    assert targets[0]["id"] == 825
    assert targets[0]["provider"] == "fcm"


def test_device_provider_02_expo_fallback_when_fcm_absent() -> None:
    """DEVICE-PROVIDER-02 : FCM absent de la sélection => Expo."""
    rows = [_ios_row(824, "atmr-1787520916767-a4kexsnra", "expo")]
    targets = prepare_driver_push_targets(rows, driver_id=20135)
    assert len(targets) == 1
    assert targets[0]["provider"] == "expo"


def test_device_provider_03_two_physical_devices() -> None:
    """DEVICE-PROVIDER-03 : deux device_id => une alerte chacun."""
    rows = [
        _ios_row(824, "phone-a", "expo"),
        _ios_row(825, "phone-a", "fcm"),
        _ios_row(900, "phone-b", "expo"),
        _ios_row(901, "phone-b", "fcm"),
    ]
    targets = prepare_driver_push_targets(rows, driver_id=20135)
    assert len(targets) == 2
    by_device = {row["device_id"]: row["provider"] for row in targets}
    assert by_device == {"phone-a": "fcm", "phone-b": "fcm"}


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
