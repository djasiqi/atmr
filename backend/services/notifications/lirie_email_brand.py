"""Assets / helpers de marque pour les emails transactionnels LIRIE."""

from __future__ import annotations

import os
from pathlib import Path

_LOGO_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "assets" / "lirie" / "logo-lirie.png",
    Path(__file__).resolve().parents[3]
    / "frontend"
    / "public"
    / "logo-lirie.png",
)


def resolve_lirie_logo_bytes() -> bytes | None:
    """Charge le PNG logo LIRIE depuis le disque (backend/assets prioritaire)."""
    for path in _LOGO_CANDIDATES:
        try:
            if path.is_file():
                data = path.read_bytes()
                if data:
                    return data
        except OSError:
            continue
    return None


def resolve_lirie_logo_public_url() -> str | None:
    """URL absolue de secours si le CID inline n'est pas disponible."""
    explicit = (os.getenv("LIRIE_EMAIL_LOGO_URL") or "").strip()
    if explicit.startswith(("http://", "https://")):
        return explicit.rstrip("/")
    frontend = (
        os.getenv("PUBLIC_FRONTEND_URL")
        or os.getenv("FRONTEND_URL")
        or os.getenv("PUBLIC_APP_URL")
        or ""
    ).strip().rstrip("/")
    if frontend.startswith(("http://", "https://")) and "localhost" not in frontend:
        return f"{frontend}/logo-lirie.png"
    # Prod publique connue (emails consultés hors machine locale)
    return "https://www.lirie.ch/logo-lirie.png"


def build_lirie_logo_email_assets() -> tuple[str | None, list[dict]]:
    """Retourne ``(logo_src, attachments)`` pour le template d'activation.

    Préfère ``cid:company_logo`` (image embarquée) avec fallback URL publique.
    """
    logo_bytes = resolve_lirie_logo_bytes()
    if logo_bytes:
        return "cid:company_logo", [
            {
                "filename": "logo-lirie.png",
                "content": logo_bytes,
                "cid": "company_logo",
                "mime_type": "image/png",
            }
        ]
    return resolve_lirie_logo_public_url(), []
