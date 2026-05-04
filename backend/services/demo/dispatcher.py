from __future__ import annotations

import os
from datetime import datetime
from html import escape
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from services.notifications.email import send_email_notification


def get_demo_destination_email() -> str:
    return os.getenv(
        "DEMO_EMAIL_SALES", os.getenv("CONTACT_EMAIL_DEFAULT", "info@lirie.ch")
    )


def build_demo_email_body(payload: dict[str, Any]) -> str:
    parts = [
        "Nouvelle demande de demonstration",
        "",
        f"Nom: {payload.get('name')}",
        f"Email: {payload.get('email')}",
        f"Telephone: {payload.get('phone') or '-'}",
        f"Organisation: {payload.get('organization')}",
        f"Type organisation: {payload.get('organization_type')}",
        f"Cas d'usage: {payload.get('use_case')}",
        f"Volume: {payload.get('volume_range') or '-'}",
        f"Integration requise: {payload.get('integration_required')}",
        f"Systeme integration: {payload.get('integration_system') or '-'}",
        f"Timing: {payload.get('timing')}",
        f"Creneau: {payload.get('preferred_slot')}",
        f"Plage: {payload.get('preferred_period')}",
        f"Score: {payload.get('score')}",
        f"Priorite: {payload.get('priority')}",
        f"Trace ID: {payload.get('trace_id')}",
        "",
        "Commentaire:",
        payload.get("comment") or "-",
    ]
    return "\n".join(parts)


def send_demo_notification(payload: dict[str, Any]) -> dict[str, Any]:
    destination = get_demo_destination_email()
    subject = (
        f"[LIRIE Demo] {payload.get('organization')} "
        f"({payload.get('priority', 'standard').upper()})"
    )
    body = build_demo_email_body(payload)
    requester_email = str(payload.get("email") or "").strip() or None
    return send_email_notification(
        destination,
        subject,
        body,
        notification_type="demo_request",
        html=False,
        reply_to=requester_email,
        from_email=os.getenv(
            "DEMO_EMAIL_NOREPLY", os.getenv("SMTP_FROM_EMAIL", "noreply@lirie.ch")
        ),
        from_name=os.getenv("DEMO_EMAIL_NOREPLY_NAME", "LIRIE"),
    )


def send_demo_acknowledgement(payload: dict[str, Any]) -> dict[str, Any]:
    client_email = payload.get("email")
    if not client_email:
        return {"ok": False, "error": "missing_email"}

    subject = "Nous avons bien reçu votre demande de démonstration"
    body = _build_email_shell(
        title="Demande de démonstration reçue",
        intro="Bonjour,",
        lines=[
            "Merci pour votre demande de démonstration LIRIE.",
            "Notre équipe reviendra vers vous sous 24h ouvrées.",
        ],
    )
    return send_email_notification(
        client_email,
        subject,
        body,
        notification_type="demo_request_ack",
        html=True,
        from_email=os.getenv(
            "DEMO_EMAIL_NOREPLY", os.getenv("SMTP_FROM_EMAIL", "noreply@lirie.ch")
        ),
        from_name=os.getenv("DEMO_EMAIL_NOREPLY_NAME", "LIRIE"),
    )


def _build_demo_magic_link(magic_token: str) -> str:
    configured_base = (os.getenv("DEMO_MAGIC_LINK_BASE_URL") or "").strip()
    if configured_base:
        base_url = configured_base
    else:
        app_env = str(os.getenv("APP_ENV", "")).strip().lower()
        flask_config = str(os.getenv("FLASK_CONFIG", "")).strip().lower()
        is_prod = app_env == "production" or flask_config == "production"
        base_url = (
            "https://www.lirie.ch/demo-access/consume"
            if is_prod
            else "http://localhost:3000/demo-access/consume"
        )
    query = urlencode({"token": magic_token})
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{query}"


def _format_local_datetime(value: Any) -> str:
    if not isinstance(value, datetime):
        return "-"
    timezone_name = (
        os.getenv("APP_TIMEZONE") or os.getenv("TZ") or "Europe/Zurich"
    ).strip()
    try:
        local_dt = value.astimezone(ZoneInfo(timezone_name))
    except Exception:
        local_dt = value
    return local_dt.strftime("%H:%M %d.%m.%Y")


def _build_email_shell(*, title: str, intro: str, lines: list[str]) -> str:
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
    content = "".join(
        f'<p style="margin:0 0 10px;">{escape(line)}</p>'
        for line in lines
        if str(line or "").strip()
    )
    return f"""
<div style="font-family: Arial, sans-serif; color: #1f2937; line-height: 1.55;">
  <h2 style="margin: 0 0 12px; font-size: 18px; color: #143935;">{escape(title)}</h2>
  <p style="margin: 0 0 12px;">{escape(intro)}</p>
  {content}
  <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 16px 0;" />
  <p style="margin: 0 0 6px; color: #4b5563;">
    Ceci est un message automatique, merci de ne pas y repondre.
  </p>
  <p style="margin: 0; color: #4b5563;">L'equipe LIRIE - info@lirie.ch</p>
  <p style="margin: 0; color: #4b5563;">LIRIE</p>
  {logo_block}
</div>
""".strip()


def send_demo_access_ready_email(
    *,
    demo_request: Any,
    demo_access: Any,
    magic_token: str,
) -> dict[str, Any]:
    recipient = getattr(demo_request, "email", None)
    if not recipient:
        return {"ok": False, "error": "missing_email"}

    subject = "Vos acces demo LIRIE sont prets"
    expires = getattr(demo_access, "demo_expires_at", None)
    expires_label = _format_local_datetime(expires)
    magic_link = _build_demo_magic_link(magic_token)
    org_type = str(getattr(demo_request, "organization_type", "") or "").strip().lower()
    if org_type in {"transport_company", "transport"}:
        parcours_hint = "Parcours recommande: Transporteur."
    elif org_type in {"institution", "ems", "clinic", "hospital"}:
        parcours_hint = "Parcours recommande: Institution."
    else:
        parcours_hint = "Parcours recommande: selon votre besoin metier."

    include_local_credentials = str(
        os.getenv("DEMO_EMAIL_INCLUDE_LOCAL_CREDENTIALS", "false")
    ).strip().lower() in {"1", "true", "yes", "on"}
    local_login_email = (os.getenv("DEMO_LOCAL_LOGIN_EMAIL") or "").strip()
    local_login_password = (os.getenv("DEMO_LOCAL_LOGIN_PASSWORD") or "").strip()
    local_credentials_block: list[str] = []
    if include_local_credentials and local_login_email and local_login_password:
        local_credentials_block = [
            "",
            "Acces local (developpement):",
            f"Email: {local_login_email}",
            f"Mot de passe temporaire: {local_login_password}",
        ]

    body_lines = [
        "Nous avons le plaisir de vous transmettre vos acces a la version demo LIRIE.",
        f"Acces direct a votre espace demo : {magic_link}",
        "Ce lien de connexion securise expire dans 30 minutes.",
        f"Votre acces demo est actif pendant 24h, soit jusqu'au {expires_label} (heure locale).",
        parcours_hint,
        *local_credentials_block,
        "Besoin d'aide ? Contactez-nous a info@lirie.ch.",
    ]
    body = _build_email_shell(
        title="Acces demo LIRIE",
        intro="Bonjour,",
        lines=body_lines,
    )
    return send_email_notification(
        recipient,
        subject,
        body,
        notification_type="demo_access_ready",
        html=True,
        from_email=os.getenv("DEMO_EMAIL_FROM")
        or os.getenv("SMTP_FROM_EMAIL", "noreply@lirie.ch"),
        from_name=os.getenv("DEMO_EMAIL_FROM_NAME")
        or os.getenv("SMTP_FROM_NAME", "LIRIE"),
        reply_to=get_demo_destination_email(),
    )


def send_demo_rejection_email(*, demo_request: Any) -> dict[str, Any]:
    recipient = getattr(demo_request, "email", None)
    if not recipient:
        return {"ok": False, "error": "missing_email"}

    organization = str(getattr(demo_request, "organization", "") or "").strip()
    subject = "Mise a jour de votre demande de demonstration LIRIE"
    body = _build_email_shell(
        title="Mise a jour de votre demande de demonstration",
        intro="Bonjour,",
        lines=[
            "Merci pour votre interet pour LIRIE.",
            (
                f"Apres analyse de votre demande pour {organization}, "
                + "nous ne pouvons pas valider l'acces demo pour le moment."
                if organization
                else "Apres analyse de votre demande, nous ne pouvons pas valider l'acces demo pour le moment."
            ),
            "N'hesitez pas a nous recontacter avec davantage de contexte metier.",
        ],
    )
    return send_email_notification(
        recipient,
        subject,
        body,
        notification_type="demo_request_rejected",
        html=True,
        from_email=os.getenv("DEMO_EMAIL_FROM", "demo@lirie.ch"),
        from_name=os.getenv("DEMO_EMAIL_FROM_NAME", "LIRIE Demo"),
        reply_to=get_demo_destination_email(),
    )
