"""Firebase Cloud Messaging push service.

Platform-specific routing:
- Android: data-only (high priority) → background handler + Notifee display
- iOS: notification+data (alert) → system tray display natively
- Silent: data-only with correct APNs headers for background sync
"""

from __future__ import annotations

import json
import os
from typing import Any

from ext import app_logger

_firebase_initialized = False


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
        if cred_path and os.path.isfile(cred_path):
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
    except Exception:
        app_logger.exception("[fcm] Firebase init failed")
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
    try:
        message_id = messaging.send(msg)
        app_logger.info("[fcm] Android push sent: %s", message_id)
        return {"ok": True, "message_id": message_id}
    except messaging.UnregisteredError:
        app_logger.warning("[fcm] Android token unregistered: %s...", token[:20])
        return {"ok": False, "error": "token_unregistered", "token_invalid": True}
    except messaging.SenderIdMismatchError:
        app_logger.warning("[fcm] Sender ID mismatch for token: %s...", token[:20])
        return {"ok": False, "error": "sender_id_mismatch", "token_invalid": True}
    except messaging.QuotaExceededError:
        app_logger.warning("[fcm] FCM quota exceeded")
        return {"ok": False, "error": "quota_exceeded"}
    except Exception:
        app_logger.exception("[fcm] Android push failed")
        return {"ok": False, "error": "fcm_send_error"}


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
    try:
        message_id = messaging.send(msg)
        app_logger.info("[fcm] iOS push sent: %s", message_id)
        return {"ok": True, "message_id": message_id}
    except messaging.UnregisteredError:
        app_logger.warning("[fcm] iOS token unregistered: %s...", token[:20])
        return {"ok": False, "error": "token_unregistered", "token_invalid": True}
    except messaging.SenderIdMismatchError:
        app_logger.warning("[fcm] Sender ID mismatch for iOS token: %s...", token[:20])
        return {"ok": False, "error": "sender_id_mismatch", "token_invalid": True}
    except messaging.QuotaExceededError:
        app_logger.warning("[fcm] FCM quota exceeded")
        return {"ok": False, "error": "quota_exceeded"}
    except Exception:
        app_logger.exception("[fcm] iOS push failed")
        return {"ok": False, "error": "fcm_send_error"}


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
    try:
        message_id = messaging.send(msg)
        app_logger.info("[fcm] Silent push sent (%s): %s", platform, message_id)
        return {"ok": True, "message_id": message_id}
    except messaging.UnregisteredError:
        app_logger.warning("[fcm] Silent push token unregistered: %s...", token[:20])
        return {"ok": False, "error": "token_unregistered", "token_invalid": True}
    except Exception:
        app_logger.exception("[fcm] Silent push failed (%s)", platform)
        return {"ok": False, "error": "fcm_send_error"}
