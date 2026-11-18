"""Verrou d'activation des invitations chauffeur.

Distinct de ``ORGANIZATION_APPROVAL_ENABLED`` et de
``PORTAL_GRANT_ENFORCEMENT``. Éteint par défaut : accepter un lien ne crée
ni chauffeur actif, ni habilitation.
"""

from __future__ import annotations


def driver_invitation_activation_enabled() -> bool:
    """Vrai seulement si ce processus a ouvert l'activation des invitations."""
    from flask import current_app

    return bool(current_app.config.get("DRIVER_INVITATION_ACTIVATION"))
