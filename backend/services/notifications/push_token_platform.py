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
    """Ne déduit plus Android depuis la forme du token.

    ``:APA91`` est présent sur les tokens FCM iOS et Android. La plateforme
    enregistrée par l'application est la seule source de vérité.
    """
    del token
    return False


def infer_fcm_platform(token: str, platform: str | None) -> str | None:
    """Normalise la plateforme enregistrée, sans la réécrire depuis le token.

    ``token`` est ignoré : un token FCM iOS qui contient ``:APA91`` reste iOS.
    """
    del token
    normalized = (platform or "").strip().lower()
    if normalized in ("ios", "android"):
        return normalized
    return None
