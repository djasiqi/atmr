"""Service d'invitation d'utilisateurs institution par email.

Gère:
- Génération de tokens d'invitation (secrets.token_urlsafe + sha256)
- Envoi d'email via Brevo provider existant
- Validation et activation de comptes
"""

import hashlib
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from flask import current_app, render_template

from services.email.brevo_provider import BrevoEmailProvider

logger = logging.getLogger(__name__)

INVITE_TOKEN_BYTES = 32
INVITE_EXPIRY_HOURS = 48

ROLE_LABELS = {
    "institution_admin": "Administrateur",
    "institution_requester": "Demandeur",
    "institution_reader": "Lecteur",
    "institution_billing": "Facturation",
}


@dataclass
class InviteResult:
    """Résultat d'envoi d'invitation."""

    success: bool
    token: str | None = None  # Raw token (à retourner une seule fois)
    error: str | None = None  # Message safe pour le frontend (jamais de secrets)


_EMAIL_ERROR_KEYWORDS: dict[str, str] = {
    "timeout": "Délai d'attente dépassé (timeout)",
    "timed out": "Délai d'attente dépassé (timeout)",
    "connection refused": "Impossible de contacter le service d'envoi",
    "connect": "Impossible de contacter le service d'envoi",
    "401": "Erreur d'authentification du service email",
    "403": "Erreur d'authentification du service email",
    "unauthorized": "Erreur d'authentification du service email",
    "forbidden": "Erreur d'authentification du service email",
    "smtp": "Erreur du serveur d'envoi",
    "relay": "Erreur du serveur d'envoi",
}


def _sanitize_email_error(raw_error: str | None) -> str:
    """Transforme une erreur brute en message safe pour le frontend.

    Ne jamais exposer : API keys, tokens, credentials SMTP, stack traces.
    Le détail complet est loggé côté serveur uniquement.
    """
    if not raw_error:
        return "Erreur inconnue lors de l'envoi"

    lower = raw_error.lower()

    # Correspondance par mot-clé simple
    for keyword, safe_msg in _EMAIL_ERROR_KEYWORDS.items():
        if keyword in lower:
            return safe_msg

    # Conditions combinées (2 mots-clés requis)
    if "rate" in lower and "limit" in lower:
        return "Limite d'envoi atteinte, réessayez plus tard"
    if "invalid" in lower and ("email" in lower or "address" in lower):
        return "Adresse email invalide ou rejetée"

    # Fallback générique (ne jamais passer le message brut)
    return "Erreur lors de l'envoi de l'email"


def generate_invite_token() -> tuple[str, str]:
    """Génère un token d'invitation et son hash.

    Returns:
        (raw_token, token_hash) - raw_token à envoyer par email, hash à stocker en DB
    """
    raw_token = secrets.token_urlsafe(INVITE_TOKEN_BYTES)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, token_hash


def hash_token(raw_token: str) -> str:
    """Hash un token brut pour comparaison avec la DB."""
    return hashlib.sha256(raw_token.encode()).hexdigest()


def get_invite_expiry() -> datetime:
    """Retourne la date d'expiration d'une invitation (now + 48h)."""
    return datetime.now(UTC) + timedelta(hours=INVITE_EXPIRY_HOURS)


def build_invite_url(raw_token: str) -> str:
    """Construit l'URL d'invitation pour le frontend."""
    # En dev: http://localhost:3000, en prod: domaine public
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    return f"{frontend_url}/invite/{raw_token}"


def get_role_label(role: str) -> str:
    """Retourne le libellé français d'un rôle institution."""
    return ROLE_LABELS.get(role, role or "Inconnu")


def send_invitation_email(
    to_email: str,
    first_name: str | None,
    institution_name: str,
    inviter_name: str,
    role: str,
    raw_token: str,
) -> InviteResult:
    """Envoie l'email d'invitation via Brevo.

    Args:
        to_email: Email du destinataire
        first_name: Prénom (optionnel, pour personnaliser)
        institution_name: Nom de l'institution
        inviter_name: Nom de l'admin qui invite
        role: Rôle institution attribué
        raw_token: Token brut (pour construire l'URL)

    Returns:
        InviteResult avec succès/erreur
    """
    try:
        invite_url = build_invite_url(raw_token)
        role_label = get_role_label(role)

        # Rendre le template HTML
        with current_app.app_context():
            html_content = render_template(
                "emails/invitation_email.html",
                institution_name=institution_name,
                first_name=first_name,
                inviter_name=inviter_name,
                role_label=role_label,
                invite_url=invite_url,
                current_year=datetime.now(UTC).year,
            )

        # Envoyer via Brevo
        provider = BrevoEmailProvider()
        from_email = os.getenv("INVITE_FROM_EMAIL", "noreply@lirie.ch")
        from_name = os.getenv("INVITE_FROM_NAME", "Lirie - Portail Institution")

        result = provider.send_invoice_email(
            from_email=from_email,
            from_name=from_name,
            to_email=to_email,
            to_name=first_name or to_email.split("@")[0],
            subject=f"Invitation - {institution_name}",
            html_content=html_content,
        )

        if result.success:
            logger.info(
                "[Invitation] Email envoyé à %s pour institution '%s'",
                to_email,
                institution_name,
            )
            return InviteResult(success=True, token=raw_token)

        logger.error(
            "[Invitation] Échec envoi email à %s: %s",
            to_email,
            result.error,
        )
        return InviteResult(
            success=False,
            error=_sanitize_email_error(result.error),
        )

    except Exception as e:
        # Log complet côté serveur, message safe côté client
        logger.exception("[Invitation] Erreur envoi email à %s: %s", to_email, e)
        return InviteResult(
            success=False,
            error=_sanitize_email_error(str(e)),
        )
