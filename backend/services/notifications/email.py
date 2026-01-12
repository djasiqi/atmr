# backend/services/notifications/email.py
"""Service d'envoi d'emails via SMTP ou Brevo (ex-Sendinblue).

Utilisé comme fallback ultime quand push et SMS échouent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Configuration Email
EMAIL_ENABLED = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true"
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "smtp")  # smtp, brevo, sendgrid

# Configuration SMTP
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "notifications@atmr.app")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "ATMR Notifications")

# Configuration Brevo
BREVO_API_KEY = os.getenv("BREVO_API_KEY")


def send_email_notification(
    email: str,
    subject: str,
    body: str,
    notification_type: str = "unknown",
    *,
    html: bool = False,
) -> Dict[str, Any]:
    """Envoie un email via SMTP ou Brevo.

    Args:
        email: Adresse email du destinataire
        subject: Sujet de l'email
        body: Corps de l'email (texte ou HTML)
        notification_type: Type de notification pour logging
        html: Si True, body est du HTML

    Returns:
        Dict avec "ok" (bool) et "error" (str) ou "message_id"
    """
    if not EMAIL_ENABLED:
        logger.debug(
            "[email] Email notifications disabled (EMAIL_NOTIFICATIONS_ENABLED=false)"
        )
        return {"ok": False, "error": "Email notifications disabled"}

    if EMAIL_PROVIDER == "brevo":
        return _send_via_brevo(email, subject, body, notification_type, html=html)

    return _send_via_smtp(email, subject, body, notification_type, html=html)


def _send_via_smtp(
    email: str,
    subject: str,
    body: str,
    notification_type: str,
    *,
    html: bool = False,
) -> Dict[str, Any]:
    """Envoie un email via SMTP.

    Args:
        email: Adresse email du destinataire
        subject: Sujet de l'email
        body: Corps de l'email
        notification_type: Type de notification
        html: Si True, body est du HTML

    Returns:
        Dict avec status de l'envoi
    """
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
        logger.error("[email] SMTP credentials not configured")
        return {"ok": False, "error": "SMTP not configured"}

    # Type narrowing pour basedpyright
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.error("[email] SMTP credentials are None")
        return {"ok": False, "error": "SMTP credentials are None"}

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        logger.info(
            "[email] Sending email to %s via SMTP (type: %s)",
            email.split("@")[0][:3] + "***",  # Masquer pour privacy
            notification_type,
        )

        # Créer le message
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        msg["To"] = email
        msg["Subject"] = subject

        # Ajouter le corps (texte ou HTML)
        if html:
            part = MIMEText(body, "html", "utf-8")
        else:
            part = MIMEText(body, "plain", "utf-8")
        msg.attach(part)

        # Envoyer via SMTP
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info("[email] Email sent successfully via SMTP")
        return {"ok": True, "provider": "smtp"}

    except Exception as e:
        logger.exception("[email] SMTP sending failed: %s", e)
        return {"ok": False, "error": str(e)}


def _send_via_brevo(
    email: str,
    subject: str,
    body: str,
    notification_type: str,
    *,
    html: bool = False,
) -> Dict[str, Any]:
    """Envoie un email via Brevo API.

    Args:
        email: Adresse email du destinataire
        subject: Sujet de l'email
        body: Corps de l'email
        notification_type: Type de notification
        html: Si True, body est du HTML

    Returns:
        Dict avec status de l'envoi
    """
    if not BREVO_API_KEY:
        logger.error("[email] Brevo API key not configured")
        return {"ok": False, "error": "Brevo not configured"}

    try:
        import requests

        logger.info(
            "[email] Sending email to %s via Brevo (type: %s)",
            email.split("@")[0][:3] + "***",  # Masquer pour privacy
            notification_type,
        )

        # Préparer le payload Brevo
        payload = {
            "sender": {
                "name": SMTP_FROM_NAME,
                "email": SMTP_FROM_EMAIL,
            },
            "to": [{"email": email}],
            "subject": subject,
        }

        if html:
            payload["htmlContent"] = body
        else:
            payload["textContent"] = body

        # Envoyer via API Brevo
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "accept": "application/json",
                "api-key": BREVO_API_KEY,
                "content-type": "application/json",
            },
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()

        logger.info(
            "[email] Email sent successfully via Brevo (message_id: %s)",
            data.get("messageId"),
        )

        return {
            "ok": True,
            "provider": "brevo",
            "message_id": data.get("messageId"),
        }

    except Exception as e:
        logger.exception("[email] Brevo sending failed: %s", e)
        return {"ok": False, "error": str(e)}


def send_bulk_emails(
    recipients: list[tuple[str, str, str]],  # (email, subject, body)
    notification_type: str = "unknown",
) -> Dict[str, Any]:
    """Envoie des emails en masse.

    Args:
        recipients: Liste de tuples (email, subject, body)
        notification_type: Type de notification pour logging

    Returns:
        Dict avec statistiques d'envoi
    """
    success_count = 0
    failed_count = 0
    errors = []

    for email, subject, body in recipients:
        result = send_email_notification(email, subject, body, notification_type)
        if result.get("ok"):
            success_count += 1
        else:
            failed_count += 1
            errors.append(
                {
                    "email": email.split("@")[0][:3] + "***",  # Masquer
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
