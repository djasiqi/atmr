"""Détection format token FCM / Expo et inférence platform pour le routage push."""

from __future__ import annotations

FCM_TOKEN_PREFIX = "APA91"
EXPO_TOKEN_PREFIX = "ExponentPushToken["


def looks_like_expo_token(token: str) -> bool:
    return token.startswith(EXPO_TOKEN_PREFIX)


def looks_like_fcm_token(token: str) -> bool:
    """True si la valeur ressemble à un token FCM natif (legacy ou format prefix:APA91b…)."""
    if not token or looks_like_expo_token(token):
        return False
    if token.startswith((FCM_TOKEN_PREFIX, "APA91b")):
        return True
    if ":APA91" in token:
        return True
    return len(token) > 100


def is_android_fcm_registration_token(token: str) -> bool:
    """Token FCM émis par Firebase Android SDK (pas un token APNs iOS)."""
    if looks_like_expo_token(token):
        return False
    if token.startswith((FCM_TOKEN_PREFIX, "APA91b")):
        return True
    return ":APA91" in token


def infer_fcm_platform(token: str, platform: str | None) -> str | None:
    """Corrige platform=ios lorsqu'un token FCM Android est enregistré par erreur."""
    normalized = (platform or "").strip().lower() or None
    if normalized == "android":
        return "android"
    if normalized == "ios" and is_android_fcm_registration_token(token):
        return "android"
    return normalized
