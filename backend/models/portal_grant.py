"""Habilitation d'un utilisateur vers un portail et une organisation.

Le rôle historique ``User.role`` n'est pas réécrit. Une ligne active n'ouvre
aucun accès tant que ``PORTAL_GRANT_ENFORCEMENT`` est éteint. Une ligne
révoquée refuse l'accès même si le drapeau est éteint, et ne peut pas être
supprimée par l'application.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    event,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ext import db

from .enums import PortalGrantStatus, PortalKind


class RevokedPortalGrantProtected(RuntimeError):
    """Une révocation ne se supprime pas : l'effacement rouvrirait le rôle."""


class PortalGrant(db.Model):
    """Une ligne = un droit explicite, jamais déduit d'un rattachement ambigu."""

    __tablename__ = "portal_grant"
    __table_args__ = (
        CheckConstraint(
            "portal IN ('client', 'company', 'driver', 'institution')",
            name="ck_portal_grant_portal",
        ),
        CheckConstraint(
            "status IN ('pending_activation', 'active', 'revoked')",
            name="ck_portal_grant_status",
        ),
        CheckConstraint(
            "("
            "(portal = 'client' AND company_id IS NULL "
            "AND institution_id IS NULL) OR "
            "(portal IN ('company', 'driver') AND company_id IS NOT NULL "
            "AND institution_id IS NULL) OR "
            "(portal = 'institution' AND institution_id IS NOT NULL "
            "AND company_id IS NULL)"
            ")",
            name="ck_portal_grant_organization",
        ),
        CheckConstraint(
            "(status <> 'revoked') OR (revoked_at IS NOT NULL)",
            name="ck_portal_grant_revoked_at",
        ),
        Index("ix_portal_grant_user_status", "user_id", "status"),
        Index(
            "uq_portal_grant_user_company_portal",
            "user_id",
            "portal",
            "company_id",
            unique=True,
            postgresql_where=text("company_id IS NOT NULL"),
        ),
        Index(
            "uq_portal_grant_user_institution_portal",
            "user_id",
            "portal",
            "institution_id",
            unique=True,
            postgresql_where=text("institution_id IS NOT NULL"),
        ),
        Index(
            "uq_portal_grant_user_client",
            "user_id",
            unique=True,
            postgresql_where=text("portal = 'client'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )
    portal: Mapped[str] = mapped_column(String(32), nullable=False)
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=True,
    )
    institution_id: Mapped[int | None] = mapped_column(
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=True,
    )
    internal_role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=PortalGrantStatus.ACTIVE.value,
        server_default=PortalGrantStatus.ACTIVE.value,
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        onupdate=func.now(),
    )

    user = relationship("User", foreign_keys=[user_id])
    company = relationship("Company", foreign_keys=[company_id])
    institution = relationship("Institution", foreign_keys=[institution_id])

    @property
    def organization_id(self) -> int | None:
        """Organisation visée. Nulle pour le portail client."""
        if self.portal == PortalKind.INSTITUTION.value:
            return self.institution_id
        if self.portal == PortalKind.CLIENT.value:
            return None
        return self.company_id


@event.listens_for(PortalGrant, "before_delete")
def _keep_revoked_portal_grant(_mapper, _connection, target) -> None:
    """Empêche l'effacement d'une révocation, qui rendrait le repli historique."""
    if getattr(target, "status", None) == PortalGrantStatus.REVOKED.value:
        raise RevokedPortalGrantProtected("revoked_portal_grant_undeletable")


REVOKED_DELETE_GUARD_FUNCTION = "portal_grant_forbid_revoked_delete"
REVOKED_DELETE_GUARD_TRIGGER = "portal_grant_forbid_revoked_delete"

REVOKED_DELETE_GUARD_SQL = (
    """
CREATE OR REPLACE FUNCTION portal_grant_forbid_revoked_delete()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.status = 'revoked' THEN
        RAISE EXCEPTION 'revoked_portal_grant_undeletable';
    END IF;
    RETURN OLD;
END;
$$
""",
    "DROP TRIGGER IF EXISTS portal_grant_forbid_revoked_delete ON portal_grant",
    """
CREATE TRIGGER portal_grant_forbid_revoked_delete
BEFORE DELETE ON portal_grant
FOR EACH ROW
EXECUTE PROCEDURE portal_grant_forbid_revoked_delete()
""",
)

REVOKED_DELETE_GUARD_DROP_SQL = (
    "DROP TRIGGER IF EXISTS portal_grant_forbid_revoked_delete ON portal_grant",
    "DROP FUNCTION IF EXISTS portal_grant_forbid_revoked_delete()",
)
