# backend/services/notifications/email.py
"""Service d'envoi d'emails via SMTP ou Brevo (ex-Sendinblue).

Utilisé comme fallback ultime quand push et SMS échouent.
L'envoi Brevo délègue à BrevoEmailProvider (chemin unique).
"""

from __future__ import annotations

import html as html_lib
import logging
import os
import re
from typing import Any, Dict

from services.notifications.email_errors import (
    EmailPermanentError,
    EmailRetryableError,
)

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


def is_email_provider_configured() -> tuple[bool, str | None]:
    """Vérifie si le provider email est activé et configuré (pré-check enqueue)."""
    if not EMAIL_ENABLED:
        return False, "Email notifications disabled"
    if EMAIL_PROVIDER == "brevo":
        if not (BREVO_API_KEY or os.getenv("BREVO_API_KEY")):
            return False, "Brevo not configured"
        return True, None
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
        return False, "SMTP not configured"
    return True, None


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
    raise_on_error: bool = False,
    headers: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Envoie un email via SMTP ou Brevo.

    Args:
        raise_on_error: Si True, lève EmailRetryableError / EmailPermanentError
            au lieu de retourner {"ok": False}.
        headers: En-têtes SMTP Brevo (ex. X-Mailin-custom). Ignorés en SMTP.

    Returns:
        Dict avec "ok" (bool) et "error" (str) ou "message_id"
    """
    if not EMAIL_ENABLED:
        logger.debug(
            "[email] Email notifications disabled (EMAIL_NOTIFICATIONS_ENABLED=false)"
        )
        result = {"ok": False, "error": "Email notifications disabled", "retryable": False}
        if raise_on_error:
            raise EmailPermanentError("Email notifications disabled")
        return result

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
            raise_on_error=raise_on_error,
            headers=headers,
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
        raise_on_error=raise_on_error,
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
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Envoie un email via SMTP."""
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):
        logger.error("[email] SMTP credentials not configured")
        if raise_on_error:
            raise EmailPermanentError("SMTP not configured")
        return {"ok": False, "error": "SMTP not configured", "retryable": False}

    if not SMTP_USER or not SMTP_PASSWORD:
        logger.error("[email] SMTP credentials are None")
        if raise_on_error:
            raise EmailPermanentError("SMTP credentials are None")
        return {"ok": False, "error": "SMTP credentials are None", "retryable": False}

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        logger.info(
            "[email] Sending email to %s via SMTP (type: %s)",
            email.split("@")[0][:3] + "***",
            notification_type,
        )
        sender_email = (from_email or SMTP_FROM_EMAIL).strip()
        sender_name = (from_name or SMTP_FROM_NAME).strip()

        msg = MIMEMultipart("alternative")
        msg["From"] = f"{sender_name} <{sender_email}>"
        msg["To"] = email
        msg["Subject"] = subject
        if reply_to:
            msg["Reply-To"] = reply_to

        if html:
            text_part = MIMEText(_html_to_text(body), "plain", "utf-8")
            html_part = MIMEText(body, "html", "utf-8")
            msg.attach(text_part)
            msg.attach(html_part)
        else:
            part = MIMEText(body, "plain", "utf-8")
            msg.attach(part)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info("[email] Email sent successfully via SMTP")
        return {"ok": True, "provider": "smtp"}

    except (TimeoutError, ConnectionError, OSError) as e:
        logger.exception("[email] SMTP sending failed (retryable): %s", e)
        if raise_on_error:
            raise EmailRetryableError(str(e)) from e
        return {"ok": False, "error": str(e), "retryable": True}
    except Exception as e:
        logger.exception("[email] SMTP sending failed: %s", e)
        if raise_on_error:
            raise EmailPermanentError(str(e)) from e
        return {"ok": False, "error": str(e), "retryable": False}


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
    raise_on_error: bool = False,
    headers: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Envoie un email via BrevoEmailProvider (chemin unique)."""
    api_key = BREVO_API_KEY or os.getenv("BREVO_API_KEY")
    if not api_key:
        logger.error("[email] Brevo API key not configured")
        if raise_on_error:
            raise EmailPermanentError("Brevo not configured")
        return {"ok": False, "error": "Brevo not configured", "retryable": False}

    try:
        from services.email.brevo_provider import BrevoEmailProvider

        provider = BrevoEmailProvider(api_key=api_key)
        result = provider.send_transactional(
            to_email=email,
            subject=subject,
            html_content=body if html else None,
            text_content=None if html else body,
            from_email=(from_email or SMTP_FROM_EMAIL).strip(),
            from_name=(from_name or SMTP_FROM_NAME).strip(),
            reply_to=reply_to,
            notification_type=notification_type,
            headers=headers,
        )
    except ValueError as e:
        if raise_on_error:
            raise EmailPermanentError(str(e)) from e
        return {"ok": False, "error": str(e), "retryable": False}

    if result.success:
        return {
            "ok": True,
            "provider": "brevo",
            "message_id": result.message_id,
            "status_code": result.status_code,
        }

    error_msg = result.error or "Brevo provider error"
    if raise_on_error:
        if result.retryable:
            raise EmailRetryableError(error_msg, status_code=result.status_code)
        raise EmailPermanentError(error_msg, status_code=result.status_code)

    return {
        "ok": False,
        "error": error_msg,
        "retryable": result.retryable,
        "status_code": result.status_code,
    }


def send_bulk_emails(
    recipients: list[tuple[str, str, str]],  # (email, subject, body)
    notification_type: str = "unknown",
) -> Dict[str, Any]:
    """Envoie des emails en masse."""
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
                    "email": email.split("@")[0][:3] + "***",
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
