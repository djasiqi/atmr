"""Verrou d'approbation des préinscriptions.

Distinct de ``PORTAL_GRANT_ENFORCEMENT``. Éteint par défaut : aucune
organisation ni habilitation n'est créée.
"""

from __future__ import annotations


def approval_enabled() -> bool:
    """Vrai seulement si l'approbation a été ouverte pour ce processus."""
    from flask import current_app

    return bool(current_app.config.get("ORGANIZATION_APPROVAL_ENABLED"))
