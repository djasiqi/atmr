"""ACL INSTITUTION-07 — contrôle facturation (Admin + Billing uniquement)."""

from __future__ import annotations

from flask import abort
from flask_jwt_extended import get_jwt, get_jwt_identity

from models import User
from models.enums import InstitutionRole

BILLING_CONTROL_ALLOWED_ROLES = frozenset(
    {
        InstitutionRole.ADMIN.value,
        InstitutionRole.BILLING.value,
    }
)

ROLE_REQUIRED_MSG = "Rôle requis: %s. Votre rôle: %s"


def institution_billing_control_role_allowed(role: str | None) -> bool:
    return role in BILLING_CONTROL_ALLOWED_ROLES


def require_institution_billing_control_context() -> tuple[int, int | None, str | None]:
    """Contexte JWT institution avec ACL contrôle facturation.

    Returns:
        (institution_id, user_id, institution_role)

    Raises:
        werkzeug.exceptions.HTTPException 403 si rôle non autorisé.
    """
    claims = get_jwt()
    institution_id = claims.get("institution_id")
    institution_role = claims.get("institution_role")

    if not institution_id:
        abort(403, description="Accès réservé aux utilisateurs institution")

    if not institution_billing_control_role_allowed(institution_role):
        msg = ROLE_REQUIRED_MSG % (
            ", ".join(sorted(BILLING_CONTROL_ALLOWED_ROLES)),
            institution_role,
        )
        abort(403, description=msg)

    user_id: int | None = None
    raw_identity = get_jwt_identity()
    if raw_identity is not None:
        raw = str(raw_identity).strip()
        if raw:
            if raw.isdigit():
                user_id = int(raw)
            else:
                u = User.query.filter_by(public_id=raw).first()
                if u:
                    user_id = int(u.id)
    return int(institution_id), user_id, institution_role
