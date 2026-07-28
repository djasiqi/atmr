"""Feature flags iOS FCM/Expo (Phases B/C — off par défaut)."""

from __future__ import annotations

import os


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def ios_native_fcm_preferred() -> bool:
    """Phase B : préférer FCM iOS à la sélection (défaut off jusqu'à preuve)."""
    return _flag("IOS_NATIVE_FCM_PREFERRED", "0")


def ios_expo_fallback_enabled() -> bool:
    """Phase B : autoriser fallback Expo uniquement sur failure_before_send."""
    return _flag("IOS_EXPO_FALLBACK_ENABLED", "1")


def ios_disable_expo_on_fcm_upsert() -> bool:
    """Phase C : désactiver Expo même installation à l'upsert FCM (défaut off)."""
    return _flag("IOS_DISABLE_EXPO_ON_FCM_UPSERT", "0")


# Issues FCM pour décision de fallback (Phase B)
FAILURE_BEFORE_SEND = "failure_before_send"
OUTCOME_UNKNOWN = "outcome_unknown"
PROVIDER_ACCEPTED = "provider_accepted"
PROVIDER_REJECTED = "provider_rejected"


def classify_fcm_issue_for_fallback(result: dict) -> str:
    """Classe le résultat FCM pour décider d'un fallback Expo.

    Fallback Expo permis uniquement pour ``failure_before_send``.
    ``outcome_unknown`` (timeout ambigu après émission) → interdit.
    """
    if result.get("ok"):
        return PROVIDER_ACCEPTED
    if result.get("configuration_error") or result.get("error") == "sender_id_mismatch":
        return "configuration_error"
    if result.get("token_invalid"):
        return "invalid_token"
    if result.get("circuit_breaker_open"):
        return FAILURE_BEFORE_SEND
    err = str(result.get("error") or "").lower()
    if "firebase not initialized" in err:
        return FAILURE_BEFORE_SEND
    # Timeout après tentative d'émission → outcome inconnu (pas de double push)
    if "timeout" in err or "timed out" in err:
        if result.get("emitted"):
            return OUTCOME_UNKNOWN
        return FAILURE_BEFORE_SEND
    if any(x in err for x in ("connection", "network", "unavailable")):
        return FAILURE_BEFORE_SEND
    if result.get("error") in ("provider_rejected",) or not result.get("retryable"):
        return PROVIDER_REJECTED
    return FAILURE_BEFORE_SEND


def allow_expo_fallback_for_fcm_issue(issue: str) -> bool:
    if not ios_expo_fallback_enabled():
        return False
    return issue == FAILURE_BEFORE_SEND
