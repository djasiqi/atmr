"""Hooks legacy → control plane (même transaction que l'écriture appelante)."""

from __future__ import annotations

import logging

from models.company import Company
from models.user import User

logger = logging.getLogger(__name__)


def project_company_tenant(company: Company) -> None:
    """Projette une Company si elle est classée TRANSPORT_TENANT (sinon no-op)."""
    from services.control_plane.projector import get_projector

    get_projector().ensure_company_organization(company)


def project_institution_user(user: User) -> None:
    """Re-synchronise membership + rôle template depuis l'utilisateur institution."""
    from services.control_plane.projector import get_projector

    get_projector().sync_institution_user(user)


def project_user_account_state(user: User) -> None:
    """Propage disable/archive vers les memberships projetées."""
    from services.control_plane.projector import get_projector

    get_projector().sync_user_account_state(user)
