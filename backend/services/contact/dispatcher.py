from __future__ import annotations

import os
from html import escape
from typing import Any

from services.notifications.email import send_email_notification

CONTACT_CATEGORY_TO_ENV = {
    "support": "CONTACT_EMAIL_SUPPORT",
    "institution": "CONTACT_EMAIL_INSTITUTION",
    "transport": "CONTACT_EMAIL_TRANSPORT",
    "demo": "CONTACT_EMAIL_DEMO",
    "billing": "CONTACT_EMAIL_BILLING",
    "family": "CONTACT_EMAIL_FAMILY",
}


def get_destination_email(category: str) -> str:
    defaults = {
        "support": "support@lirie.ch",
        "institution": "institution@lirie.ch",
        "transport": "entreprise@lirie.ch",
        "demo": "info@lirie.ch",
        "billing": "facturation@lirie.ch",
        "family": "famille@lirie.ch",
    }
    env_key = CONTACT_CATEGORY_TO_ENV.get(category)
    if env_key:
        return os.getenv(env_key, defaults.get(category, "info@lirie.ch"))
    return "info@lirie.ch"


CONTACT_CATEGORY_FROM_ENV = {
    "support": "CONTACT_FROM_EMAIL_SUPPORT",
    "institution": "CONTACT_FROM_EMAIL_INSTITUTION",
    "transport": "CONTACT_FROM_EMAIL_TRANSPORT",
    "demo": "CONTACT_FROM_EMAIL_DEMO",
    "billing": "CONTACT_FROM_EMAIL_BILLING",
    "family": "CONTACT_FROM_EMAIL_FAMILY",
}


def get_sender_email(category: str) -> str:
    defaults = {
        "support": "support@lirie.ch",
        "institution": "institution@lirie.ch",
        "transport": "entreprise@lirie.ch",
        "demo": "demo@lirie.ch",
        "billing": "facturation@lirie.ch",
        "family": "famille@lirie.ch",
    }
    env_key = CONTACT_CATEGORY_FROM_ENV.get(category)
    if env_key:
        return os.getenv(env_key, defaults.get(category, "support@lirie.ch"))
    return "support@lirie.ch"


def _autoresponse_subject(category: str) -> str:
    labels = {
        "support": "Support technique",
        "institution": "Institution / Integration",
        "transport": "Entreprise de transport",
        "demo": "Demonstration",
        "billing": "Facturation",
        "family": "Famille / Proche aidant",
    }
    return f"[LIRIE] Confirmation de reception - {labels.get(category, 'Contact')}"


def _autoresponse_html(payload: dict[str, Any]) -> str:
    name = (payload.get("name") or "").strip()
    greeting = f"Bonjour {escape(name)}," if name else "Bonjour,"
    trace_id = escape(str(payload.get("trace_id") or "-"))
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
    Nous avons bien recu votre demande de contact.
    Notre equipe la traitera dans les plus brefs delais (generalement sous 24h ouvrees).
  </p>
  <p style="margin: 0 0 16px;"><strong>Reference :</strong> {trace_id}</p>
  <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 16px 0;" />
  <p style="margin: 0 0 6px; color: #4b5563;">
    Ceci est un message automatique, merci de ne pas y repondre.
  </p>
  <p style="margin: 0; color: #4b5563;">
    L'equipe LIRIE - <a href="mailto:info@lirie.ch" style="color:#00796B; text-decoration:none;">info@lirie.ch</a>
  </p>
  {logo_block}
</div>
""".strip()


def build_contact_email_body(payload: dict[str, Any]) -> str:
    payload_json = payload.get("payload_json") or {}
    lines = []
    for key, value in payload_json.items():
        if value in (None, "", []):
            continue
        lines.append(f"- {key}: {value}")

    parts = [
        "Nouvelle demande de contact",
        "",
        f"Categorie: {payload.get('category')}",
        f"Priorite: {payload.get('priority')}",
        f"Nom: {payload.get('name')}",
        f"Email: {payload.get('email')}",
        f"Organisation: {payload.get('organization') or '-'}",
        f"Telephone: {payload.get('phone') or '-'}",
        f"Trace ID: {payload.get('trace_id') or '-'}",
        f"User ID: {payload.get('user_id') or '-'}",
        f"Role: {payload.get('user_role') or '-'}",
        "",
        "Message:",
        payload.get("message") or "",
    ]
    if lines:
        parts.extend(["", "Champs contextuels:", *lines])
    return "\n".join(parts)


def send_contact_notification(payload: dict[str, Any]) -> dict[str, Any]:
    category = str(payload.get("category") or "support")
    destination_email = get_destination_email(category)
    email_subject = f"[LIRIE Contact] {category}"
    body = build_contact_email_body(payload)
    reply_to = str(payload.get("email") or "").strip() or None
    internal_result = send_email_notification(
        destination_email,
        email_subject,
        body,
        notification_type="contact_request",
        html=False,
        reply_to=reply_to,
        from_email=get_sender_email(category),
        from_name="LIRIE",
    )
    if not internal_result.get("ok"):
        return internal_result

    auto_reply_enabled = (
        os.getenv("CONTACT_AUTOREPLY_ENABLED", "true").lower() == "true"
    )
    client_email = str(payload.get("email") or "").strip()
    if not auto_reply_enabled or not client_email:
        return internal_result

    autoresponse_result = send_email_notification(
        client_email,
        _autoresponse_subject(category),
        _autoresponse_html(payload),
        notification_type="contact_autoreply",
        html=True,
        from_email=os.getenv("CONTACT_AUTOREPLY_FROM_EMAIL", "noreply@lirie.ch"),
        from_name=os.getenv("CONTACT_AUTOREPLY_FROM_NAME", "LIRIE"),
        reply_to="info@lirie.ch",
    )
    if not autoresponse_result.get("ok"):
        return {
            "ok": True,
            "provider": internal_result.get("provider"),
            "auto_reply_ok": False,
            "auto_reply_error": autoresponse_result.get("error"),
        }
    return {
        "ok": True,
        "provider": internal_result.get("provider"),
        "auto_reply_ok": True,
    }
