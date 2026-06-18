"""Classification des tokens push chauffeur (HEALTHY, STALE, etc.)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from models import DeviceToken

HEALTHY_PUSH_MAX_AGE_DAYS = 7
FCM_TOKEN_PREFIX = "APA91"
EXPO_TOKEN_PREFIX = "ExponentPushToken["


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _looks_like_fcm_token(token: str) -> bool:
    return token.startswith(FCM_TOKEN_PREFIX) or (
        len(token) > 100 and not token.startswith(EXPO_TOKEN_PREFIX)
    )


def _looks_like_expo_token(token: str) -> bool:
    return token.startswith(EXPO_TOKEN_PREFIX)


def _provider_mismatch(token: DeviceToken) -> bool:
    provider = (token.provider or "expo").lower()
    value = token.token or ""
    if provider == "expo" and _looks_like_fcm_token(value):
        return True
    return provider == "fcm" and _looks_like_expo_token(value)


def classify_token(token: DeviceToken, *, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    if not token.is_active:
        return "INACTIVE"
    if _provider_mismatch(token):
        return "MISMATCH_PROVIDER"
    last_success = _as_utc(token.last_push_success_at)
    last_seen = _as_utc(token.last_seen_at)
    stale_threshold = now - timedelta(days=HEALTHY_PUSH_MAX_AGE_DAYS)
    if last_success and last_success >= stale_threshold:
        return "HEALTHY"
    if last_seen and last_seen >= stale_threshold:
        return "HEALTHY"
    if last_success is None and last_seen is None:
        return "STALE"
    return "STALE"
