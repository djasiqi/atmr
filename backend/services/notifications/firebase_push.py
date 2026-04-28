"""Firebase Cloud Messaging push service.

Platform-specific routing:
- Android: data-only (high priority) → background handler + Notifee display
- iOS: notification+data (alert) → system tray display natively
- Silent: data-only with correct APNs headers for background sync
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any

from ext import app_logger

_firebase_initialized = False

FCM_RETRY_MAX_ATTEMPTS = max(1, int(os.getenv("FCM_RETRY_MAX_ATTEMPTS", "3")))
FCM_RETRY_BASE_DELAY_MS = max(100, int(os.getenv("FCM_RETRY_BASE_DELAY_MS", "2000")))
FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD = max(
    1, int(os.getenv("FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
)
FCM_CIRCUIT_BREAKER_OPEN_SECONDS = max(
    1, int(os.getenv("FCM_CIRCUIT_BREAKER_OPEN_SECONDS", "60"))
)


class _InMemoryCircuitBreaker:
    """Circuit breaker local au process pour appels FCM."""

    def __init__(self, *, failure_threshold: int, open_seconds: int):
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._lock = threading.Lock()
        self._opened_until_ts = 0.0
        self._consecutive_retryable_failures = 0
        self._half_open_probe_in_flight = False

    def allow_request(self) -> bool:
        now = time.time()
        with self._lock:
            if self._opened_until_ts > now:
                return False
            if self._opened_until_ts > 0:
                # Passage en half-open (max 1 probe simultané)
                if self._half_open_probe_in_flight:
                    return False
                self._half_open_probe_in_flight = True
                return True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._opened_until_ts = 0.0
            self._consecutive_retryable_failures = 0
            self._half_open_probe_in_flight = False

    def record_retryable_failure(self) -> None:
        now = time.time()
        with self._lock:
            if self._half_open_probe_in_flight:
                self._half_open_probe_in_flight = False
            self._consecutive_retryable_failures += 1
            if self._consecutive_retryable_failures >= self._failure_threshold:
                self._opened_until_ts = now + self._open_seconds
                self._consecutive_retryable_failures = 0
                app_logger.warning(
                    "[fcm] Circuit breaker OPEN for %ss after retryable failures",
                    self._open_seconds,
                )

    def record_non_retryable_failure(self) -> None:
        with self._lock:
            self._half_open_probe_in_flight = False


_fcm_circuit_breaker = _InMemoryCircuitBreaker(
    failure_threshold=FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    open_seconds=FCM_CIRCUIT_BREAKER_OPEN_SECONDS,
)


def _fcm_generic_error_result(exc: BaseException) -> dict[str, Any]:
    """Erreur FCM non typée : enrichit pour lifecycle / observabilité (PR3/PR4)."""
    name = type(exc).__name__
    msg = str(exc)[:500]
    out: dict[str, Any] = {
        "ok": False,
        "error": "fcm_send_error",
        "error_class": name,
        "error_message": msg,
    }
    low = msg.lower()
    if any(
        x in low
        for x in (
            "registration",
            "not a valid fcm",
            "invalid-argument",
            "requested entity was not found",
        )
    ):
        out["token_invalid"] = True
    return out


def _is_retryable_fcm_exception(exc: BaseException) -> bool:
    """Détermine si l'erreur FCM est transitoire et éligible au retry."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    retryable_names = {
        "quotaexceedederror",
        "internalerror",
        "unavailableerror",
        "deadlineexceedederror",
    }
    if name in retryable_names:
        return True
    retryable_hints = (
        "timeout",
        "timed out",
        "temporarily unavailable",
        "unavailable",
        "deadline exceeded",
        "internal error",
        "connection reset",
        "connection aborted",
        "503",
        "500",
    )
    return any(hint in msg for hint in retryable_hints)


def _send_with_retry(msg: Any, *, platform: str) -> dict[str, Any]:
    from firebase_admin import messaging

    if not _fcm_circuit_breaker.allow_request():
        return {
            "ok": False,
            "error": "circuit_breaker_open",
            "circuit_breaker_open": True,
        }

    last_exception: BaseException | None = None
    for attempt in range(FCM_RETRY_MAX_ATTEMPTS):
        try:
            message_id = messaging.send(msg)
            _fcm_circuit_breaker.record_success()
            app_logger.info("[fcm] %s push sent: %s", platform, message_id)
            return {"ok": True, "message_id": message_id}
        except messaging.UnregisteredError:
            _fcm_circuit_breaker.record_non_retryable_failure()
            return {"ok": False, "error": "token_unregistered", "token_invalid": True}
        except messaging.SenderIdMismatchError:
            _fcm_circuit_breaker.record_non_retryable_failure()
            return {"ok": False, "error": "sender_id_mismatch", "token_invalid": True}
        except Exception as e:
            last_exception = e
            retryable = _is_retryable_fcm_exception(e)
            if not retryable:
                _fcm_circuit_breaker.record_non_retryable_failure()
                app_logger.exception("[fcm] %s push failed (non retryable)", platform)
                return _fcm_generic_error_result(e)

            is_last_attempt = attempt >= (FCM_RETRY_MAX_ATTEMPTS - 1)
            if is_last_attempt:
                _fcm_circuit_breaker.record_retryable_failure()
                app_logger.warning(
                    "[fcm] %s push failed after %s retry attempts: %s",
                    platform,
                    FCM_RETRY_MAX_ATTEMPTS,
                    type(e).__name__,
                )
                break

            delay_seconds = (FCM_RETRY_BASE_DELAY_MS / 1000.0) * (2**attempt)
            jitter = random.uniform(0.0, delay_seconds * 0.2)
            sleep_seconds = delay_seconds + jitter
            app_logger.warning(
                "[fcm] %s retry attempt %s/%s in %.2fs (%s)",
                platform,
                attempt + 2,
                FCM_RETRY_MAX_ATTEMPTS,
                sleep_seconds,
                type(e).__name__,
            )
            time.sleep(sleep_seconds)

    if last_exception is not None:
        return _fcm_generic_error_result(last_exception)
    return {"ok": False, "error": "fcm_send_error"}


def _init_firebase() -> bool:
    """Lazy-init Firebase Admin SDK.

    Supports two config methods (checked in order):
    1. FIREBASE_SERVICE_ACCOUNT_PATH — path to the JSON key file
    2. FIREBASE_SERVICE_ACCOUNT_JSON — raw JSON string (for Docker/CI)
    """
    global _firebase_initialized
    if _firebase_initialized:
        return True

    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        app_logger.error("[fcm] firebase-admin not installed")
        return False

    if firebase_admin._apps:
        _firebase_initialized = True
        return True

    cred_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH")
    cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

    try:
        if cred_path:
            if not os.path.isfile(cred_path):
                app_logger.error(
                    "[fcm] Firebase init failed: FIREBASE_SERVICE_ACCOUNT_PATH file not found — %s",
                    cred_path,
                )
                return False
            cred = credentials.Certificate(cred_path)
        elif cred_json:
            cred = credentials.Certificate(json.loads(cred_json))
        else:
            app_logger.warning(
                "[fcm] Neither FIREBASE_SERVICE_ACCOUNT_PATH nor "
                "FIREBASE_SERVICE_ACCOUNT_JSON set — FCM disabled"
            )
            return False

        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        app_logger.info("[fcm] Firebase Admin SDK initialized")
        return True
    except json.JSONDecodeError as e:
        app_logger.exception(
            "[fcm] Firebase init failed: FIREBASE_SERVICE_ACCOUNT_JSON invalid JSON — %s",
            str(e),
        )
        return False
    except Exception as e:
        app_logger.exception(
            "[fcm] Firebase init failed: %s — %s",
            type(e).__name__,
            str(e),
        )
        return False


def send_fcm_android(
    token: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
    channel_id: str = "missions_v2",
) -> dict[str, Any]:
    """Send data-only FCM message for Android (background handler + Notifee)."""
    if not _init_firebase():
        return {"ok": False, "error": "Firebase not initialized"}

    from firebase_admin import messaging

    str_data = {k: str(v) for k, v in (data or {}).items()}
    str_data.update({"title": title, "body": body, "channelId": channel_id})

    msg = messaging.Message(
        token=token,
        data=str_data,
        android=messaging.AndroidConfig(priority="high"),
    )
    result = _send_with_retry(msg, platform="Android")
    if result.get("error") == "token_unregistered":
        app_logger.warning("[fcm] Android token unregistered: %s...", token[:20])
    elif result.get("error") == "sender_id_mismatch":
        app_logger.warning("[fcm] Sender ID mismatch for token: %s...", token[:20])
    return result


def send_fcm_ios(
    token: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send notification+data FCM message for iOS (system tray display)."""
    if not _init_firebase():
        return {"ok": False, "error": "Firebase not initialized"}

    from firebase_admin import messaging

    str_data = {k: str(v) for k, v in (data or {}).items()}

    msg = messaging.Message(
        token=token,
        notification=messaging.Notification(title=title, body=body),
        data=str_data,
        apns=messaging.APNSConfig(
            headers={"apns-push-type": "alert", "apns-priority": "10"},
            payload=messaging.APNSPayload(
                aps=messaging.Aps(sound="default"),
            ),
        ),
    )
    result = _send_with_retry(msg, platform="iOS")
    if result.get("error") == "token_unregistered":
        app_logger.warning("[fcm] iOS token unregistered: %s...", token[:20])
    elif result.get("error") == "sender_id_mismatch":
        app_logger.warning("[fcm] Sender ID mismatch for iOS token: %s...", token[:20])
    return result


def send_fcm_silent(
    token: str,
    data: dict[str, Any] | None = None,
    platform: str = "android",
) -> dict[str, Any]:
    """Send silent/data-only FCM push for background sync.

    - Android: data-only, high priority (handler decides whether to display)
    - iOS: content-available with correct APNs headers (background, priority 5)
    """
    if not _init_firebase():
        return {"ok": False, "error": "Firebase not initialized"}

    from firebase_admin import messaging

    str_data = {"type": "silent_update", **{k: str(v) for k, v in (data or {}).items()}}

    apns_config = None
    android_config = None

    if platform == "ios":
        apns_config = messaging.APNSConfig(
            headers={"apns-push-type": "background", "apns-priority": "5"},
            payload=messaging.APNSPayload(
                aps=messaging.Aps(content_available=True),
            ),
        )
    else:
        android_config = messaging.AndroidConfig(priority="high")

    msg = messaging.Message(
        token=token,
        data=str_data,
        android=android_config,
        apns=apns_config,
    )
    result = _send_with_retry(msg, platform=f"Silent-{platform}")
    if result.get("error") == "token_unregistered":
        app_logger.warning("[fcm] Silent push token unregistered: %s...", token[:20])
    elif result.get("error") == "sender_id_mismatch":
        app_logger.warning("[fcm] Silent push sender ID mismatch: %s...", token[:20])
    return result
