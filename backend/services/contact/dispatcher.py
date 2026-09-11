from __future__ import annotations

import os
from datetime import UTC, datetime
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

from services.notifications.email import send_email_notification

CONTACT_INTERNAL_TO_DEFAULT = "info@lirie.ch"
CONTACT_INTERNAL_FROM_DEFAULT = "noreply@lirie.ch"
CONTACT_TIMEZONE = ZoneInfo("Europe/Zurich")

CONTACT_CATEGORY_LABELS = {
    "support": "Support technique",
    "institution": "Institution / Intégration",
    "transport": "Entreprise de transport",
    "demo": "Démonstration",
    "billing": "Facturation",
    "family": "Famille / Proche aidant",
}

# Conservé pour compatibilité d'import : le routage n'utilise plus une boîte par catégorie.
CONTACT_CATEGORY_TO_ENV: dict[str, str] = {}


def category_label(category: str) -> str:
    return CONTACT_CATEGORY_LABELS.get(str(category or "").strip(), "Contact")


def get_destination_email(_category: str | None = None) -> str:
    """Boîte opérationnelle unique pour toutes les demandes de contact."""
    value = (os.getenv("CONTACT_EMAIL_DEFAULT") or CONTACT_INTERNAL_TO_DEFAULT).strip()
    return value or CONTACT_INTERNAL_TO_DEFAULT


def get_sender_email(_category: str | None = None) -> str:
    """Expéditeur technique unique (noreply), jamais une boîte métier."""
    value = (os.getenv("CONTACT_FROM_EMAIL") or CONTACT_INTERNAL_FROM_DEFAULT).strip()
    return value or CONTACT_INTERNAL_FROM_DEFAULT


def format_contact_datetime(value: datetime | None = None) -> str:
    moment = value or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(CONTACT_TIMEZONE).strftime("%d.%m.%Y %H:%M")


def build_internal_subject(category: str, trace_id: str) -> str:
    reference = str(trace_id or "").strip() or "—"
    return f"[LIRIE Contact][{category_label(category)}] Nouvelle demande — {reference}"


def _autoresponse_subject(category: str) -> str:
    return f"[LIRIE] Confirmation de réception - {category_label(category)}"


def _display(value: Any, fallback: str = "—") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _autoresponse_html(payload: dict[str, Any]) -> str:
    name = (payload.get("name") or "").strip()
    greeting = f"Bonjour {escape(name)}," if name else "Bonjour,"
    trace_id = escape(str(payload.get("trace_id") or "—"))
    logo_url = os.getenv(
        "CONTACT_AUTOREPLY_LOGO_URL",
        "https://www.lirie.ch/logo-lirie.png",
    ).strip()
    logo_block = ""
    if logo_url:
        logo_block = (
            f'<img src="{escape(logo_url)}" alt="LIRIE" '
            'style="height:32px; width:auto; display:block; margin-top:12px;" />'
        )

    return f"""
<div style="font-family: Arial, sans-serif; color: #1f2937; line-height: 1.55;">
  <p style="margin: 0 0 12px;">{greeting}</p>
  <p style="margin: 0 0 12px;">
    Nous avons bien reçu votre demande de contact.
    Notre équipe la traitera dans les plus brefs délais (généralement sous 24h ouvrées).
  </p>
  <p style="margin: 0 0 16px;"><strong>Référence :</strong> {trace_id}</p>
  <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 16px 0;" />
  <p style="margin: 0 0 6px; color: #4b5563;">
    Ceci est un message automatique, merci de ne pas y répondre.
  </p>
  <p style="margin: 0; color: #4b5563;">
    L'équipe LIRIE - <a href="mailto:info@lirie.ch" style="color:#00796B; text-decoration:none;">info@lirie.ch</a>
  </p>
  {logo_block}
</div>
""".strip()


def build_contact_email_body(payload: dict[str, Any]) -> str:
    payload_json = payload.get("payload_json") or {}
    created_at = payload.get("created_at")
    created_dt = created_at if isinstance(created_at, datetime) else None

    extra_lines = []
    for key, value in payload_json.items():
        if value in (None, "", []):
            continue
        extra_lines.append(f"{key} : {value}")

    parts = [
        "Nouvelle demande de contact LIRIE",
        "",
        f"Référence : {_display(payload.get('trace_id'))}",
        f"Type : {category_label(str(payload.get('category') or ''))}",
        "",
        f"Nom : {_display(payload.get('name'))}",
        f"Email : {_display(payload.get('email'))}",
        f"Téléphone : {_display(payload.get('phone'))}",
        f"Institution : {_display(payload.get('organization'))}",
        f"Message : {_display(payload.get('message'), '')}",
        "",
        f"Date : {format_contact_datetime(created_dt)}",
    ]
    if extra_lines:
        parts.extend(["", "Champs contextuels :", *extra_lines])
    return "\n".join(parts)


def payload_from_contact_request(row: Any) -> dict[str, Any]:
    return {
        "category": getattr(row, "category", None),
        "name": getattr(row, "name", None),
        "email": getattr(row, "email", None),
        "phone": getattr(row, "phone", None),
        "organization": getattr(row, "organization", None),
        "message": getattr(row, "message", None),
        "priority": getattr(row, "priority", None),
        "payload_json": getattr(row, "payload_json", None) or {},
        "trace_id": getattr(row, "trace_id", None),
        "user_id": getattr(row, "user_id", None),
        "user_role": getattr(row, "user_role", None),
        "created_at": getattr(row, "created_at", None),
    }


def send_internal_contact_notification(payload: dict[str, Any]) -> dict[str, Any]:
    category = str(payload.get("category") or "support")
    destination_email = get_destination_email()
    from_email = get_sender_email()
    reply_to = str(payload.get("email") or "").strip() or None
    result = send_email_notification(
        destination_email,
        build_internal_subject(category, str(payload.get("trace_id") or "")),
        build_contact_email_body(payload),
        notification_type="contact_request",
        html=False,
        reply_to=reply_to,
        from_email=from_email,
        from_name="LIRIE",
    )
    return {
        "ok": bool(result.get("ok")),
        "provider": result.get("provider"),
        "error": result.get("error"),
        "destination": destination_email,
        "from_email": from_email,
        "reply_to": reply_to,
    }


def send_contact_autoreply(payload: dict[str, Any]) -> dict[str, Any]:
    auto_reply_enabled = (
        os.getenv("CONTACT_AUTOREPLY_ENABLED", "true").lower() == "true"
    )
    client_email = str(payload.get("email") or "").strip()
    if not auto_reply_enabled or not client_email:
        return {"ok": False, "skipped": True, "error": None}

    category = str(payload.get("category") or "support")
    result = send_email_notification(
        client_email,
        _autoresponse_subject(category),
        _autoresponse_html(payload),
        notification_type="contact_autoreply",
        html=True,
        from_email=os.getenv("CONTACT_AUTOREPLY_FROM_EMAIL", "noreply@lirie.ch"),
        from_name=os.getenv("CONTACT_AUTOREPLY_FROM_NAME", "LIRIE"),
        reply_to="info@lirie.ch",
    )
    return {
        "ok": bool(result.get("ok")),
        "skipped": False,
        "error": result.get("error"),
        "provider": result.get("provider"),
    }


def send_contact_notification(payload: dict[str, Any]) -> dict[str, Any]:
    """Envoie confirmation client et notification interne indépendamment.

    La confirmation client n'est jamais la preuve que LIRIE a reçu
    la notification interne.
    """
    autoresponse_result = send_contact_autoreply(payload)
    internal_result = send_internal_contact_notification(payload)
    return {
        "ok": bool(internal_result.get("ok")),
        "provider": internal_result.get("provider"),
        "internal_ok": bool(internal_result.get("ok")),
        "internal_error": internal_result.get("error"),
        "auto_reply_ok": bool(autoresponse_result.get("ok")),
        "auto_reply_skipped": bool(autoresponse_result.get("skipped")),
        "auto_reply_error": autoresponse_result.get("error"),
        "destination": internal_result.get("destination"),
        "from_email": internal_result.get("from_email"),
        "reply_to": internal_result.get("reply_to"),
    }
