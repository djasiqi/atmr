"""Règles d'accès bootstrap — alignées sur `_check_user_profile_active` (auth.py)."""

from __future__ import annotations

import logging
from application.auth_bootstrap import access_denied_codes as codes
from application.auth_bootstrap.session_bootstrap_snapshot import (
    SessionBootstrapSnapshot,
)
from models import User
from models.enums import UserRole
from services.demo.access_service import enforce_demo_user_access_validity

logger = logging.getLogger(__name__)


def evaluate_access_denial(
    snapshot: SessionBootstrapSnapshot,
    user_orm: User,
) -> tuple[str, str] | None:
    """Retourne `(access_denied_code, message)` si accès refusé, sinon `None`."""
    if snapshot.account_status == "pending_activation":
        return (
            codes.PENDING_ACTIVATION,
            "Compte en attente de validation email/SMS.",
        )

    if snapshot.role == UserRole.DRIVER and snapshot.driver_id is not None:
        if snapshot.driver_is_active is False:
            return (codes.DRIVER_PROFILE_INACTIVE, "Compte désactivé")

    if snapshot.role == UserRole.CLIENT and snapshot.client_active_flags:
        if not any(snapshot.client_active_flags):
            return (codes.NO_ACTIVE_CLIENT_PROFILE, "Compte désactivé")

    if snapshot.role == UserRole.INSTITUTION:
        if snapshot.account_status == "disabled":
            return (codes.INSTITUTION_DISABLED, "Compte désactivé")
        if snapshot.account_status == "invited":
            return (
                codes.INSTITUTION_INVITED,
                "Compte non encore activé. Vérifiez votre email d'invitation.",
            )

    demo_ok, demo_msg = enforce_demo_user_access_validity(user_orm)
    if not demo_ok:
        return (codes.DEMO_EXPIRED, demo_msg or "Accès démo expiré.")

    return None


def observe_driver_without_profile(snapshot: SessionBootstrapSnapshot) -> None:
    """Métrique / log si rôle driver sans ligne Driver (héritage ORM : accès autorisé)."""
    if snapshot.role != UserRole.DRIVER:
        return
    if snapshot.driver_id is not None:
        return
    try:
        from services.monitoring import auth_bootstrap_metrics as abm

        abm.inc_driver_role_without_driver_row()
    except Exception:  # noqa: BLE001 — métriques ne doivent pas casser le flux
        pass
    logger.info(
        "bootstrap_session driver role without driver row (allowed by legacy rules)",
        extra={
            "event": "driver_role_without_driver_row",
            "user_id": snapshot.user_id,
            "role": snapshot.role.value,
        },
    )
