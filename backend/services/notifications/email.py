# backend/services/notifications/email.py
"""Service d'envoi d'emails via SMTP ou Brevo (ex-Sendinblue).

Utilisé comme fallback ultime quand push et SMS échouent.
"""

from __future__ import annotations

import html as html_lib
import logging
import os
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Configuration Email
EMAIL_ENABLED = os.getenv("EMAIL_NOTIFICATIONS_ENABLED", "false").lower() == "true"
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "smtp")  # smtp, brevo, sendgrid

# Configuration SMTP
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER") or os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "notifications@atmr.app")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "ATMR Notifications")

# Configuration Brevo
BREVO_API_KEY = os.getenv("BREVO_API_KEY")


def _html_to_text(html_content: str) -> str:
    """Convertit un HTML simple en texte brut pour améliorer la délivrabilité."""
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", html_content, flags=re.IGNORECASE)
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", no_style, flags=re.IGNORECASE)
    with_newlines = re.sub(
        r"</(p|div|br|li|h1|h2|h3|tr|table)>", "\n", no_script, flags=re.IGNORECASE
    )
    no_tags = re.sub(r"<[^>]+>", " ", with_newlines)
    normalized = re.sub(r"[ \t\r\f\v]+", " ", no_tags)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return html_lib.unescape(normalized).strip()


def send_email_notification(
    email: str,
    subject: str,
    body: str,
    notification_type: str = "unknown",
    *,
    html: bool = False,
    reply_to: str | None = None,
    from_email: str | None = None,
    from_name: str | None = None,
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
        return _send_via_brevo(
            email,
            subject,
            body,
            notification_type,
            html=html,
            reply_to=reply_to,
            from_email=from_email,
            from_name=from_name,
        )

    return _send_via_smtp(
        email,
        subject,
        body,
        notification_type,
        html=html,
        reply_to=reply_to,
        from_email=from_email,
        from_name=from_name,
    )


def _send_via_smtp(
    email: str,
    subject: str,
    body: str,
    notification_type: str,
    *,
    html: bool = False,
    reply_to: str | None = None,
    from_email: str | None = None,
    from_name: str | None = None,
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
        sender_email = (from_email or SMTP_FROM_EMAIL).strip()
        sender_name = (from_name or SMTP_FROM_NAME).strip()

        # Créer le message
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{sender_name} <{sender_email}>"
        msg["To"] = email
        msg["Subject"] = subject
        if reply_to:
            msg["Reply-To"] = reply_to

        # Ajouter le corps (texte + HTML quand disponible) pour une meilleure délivrabilité.
        if html:
            text_part = MIMEText(_html_to_text(body), "plain", "utf-8")
            html_part = MIMEText(body, "html", "utf-8")
            msg.attach(text_part)
            msg.attach(html_part)
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
    reply_to: str | None = None,
    from_email: str | None = None,
    from_name: str | None = None,
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
                "name": (from_name or SMTP_FROM_NAME).strip(),
                "email": (from_email or SMTP_FROM_EMAIL).strip(),
            },
            "to": [{"email": email}],
            "subject": subject,
        }
        if reply_to:
            payload["replyTo"] = {"email": reply_to}

        if html:
            payload["htmlContent"] = body
            payload["textContent"] = _html_to_text(body)
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
