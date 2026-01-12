# backend/services/notifications/sms.py
"""Service d'envoi de SMS via Twilio (ou autre provider).

Utilisé comme fallback quand les notifications push échouent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Configuration Twilio (à définir dans .env)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # Numéro d'envoi
SMS_ENABLED = os.getenv("SMS_NOTIFICATIONS_ENABLED", "false").lower() == "true"


def send_sms_notification(
    phone: str,
    message: str,
    notification_type: str = "unknown",
) -> Dict[str, Any]:
    """Envoie un SMS via Twilio.

    Args:
        phone: Numéro de téléphone au format international (+33...)
        message: Contenu du SMS (max 160 caractères recommandé)
        notification_type: Type de notification pour logging

    Returns:
        Dict avec "ok" (bool) et "error" (str) ou "message_sid"
    """
    if not SMS_ENABLED:
        logger.debug(
            "[sms] SMS notifications disabled (SMS_NOTIFICATIONS_ENABLED=false)"
        )
        return {"ok": False, "error": "SMS notifications disabled"}

    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        logger.error("[sms] Twilio credentials not configured")
        return {"ok": False, "error": "Twilio not configured"}

    try:
        # Import Twilio (optionnel, n'échouera pas si non installé)
        try:
            from twilio.rest import Client  # type: ignore
        except ImportError:
            logger.error(
                "[sms] twilio package not installed. Install with: pip install twilio"
            )
            return {"ok": False, "error": "twilio package not installed"}

        # Créer client Twilio
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # Envoyer SMS
        logger.info(
            "[sms] Sending SMS to %s (type: %s)",
            phone[-4:],  # Masquer pour privacy
            notification_type,
        )

        tw_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone,
        )

        logger.info(
            "[sms] SMS sent successfully (SID: %s)",
            tw_message.sid,
        )

        return {
            "ok": True,
            "message_sid": tw_message.sid,
            "status": tw_message.status,
        }

    except Exception as e:
        logger.exception("[sms] SMS sending failed: %s", e)
        return {"ok": False, "error": str(e)}


def send_bulk_sms(
    recipients: list[tuple[str, str]],
    notification_type: str = "unknown",
) -> Dict[str, Any]:
    """Envoie des SMS en masse.

    Args:
        recipients: Liste de tuples (phone, message)
        notification_type: Type de notification pour logging

    Returns:
        Dict avec statistiques d'envoi
    """
    success_count = 0
    failed_count = 0
    errors = []

    for phone, message in recipients:
        result = send_sms_notification(phone, message, notification_type)
        if result.get("ok"):
            success_count += 1
        else:
            failed_count += 1
            errors.append(
                {
                    "phone": phone[-4:],  # Masquer
                    "error": result.get("error"),
                }
            )

    return {
        "ok": failed_count == 0,
        "total": len(recipients),
        "success": success_count,
        "failed": failed_count,
        "errors": errors if errors else None,
    }
