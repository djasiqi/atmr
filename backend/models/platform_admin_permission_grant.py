"""Grants RBAC plateforme — permissions nommées par utilisateur admin."""

from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class PlatformAdminPermissionGrant(db.Model):
    """Une ligne = une permission plateforme accordée à un compte `User` (rôle admin)."""

    __tablename__ = "platform_admin_permission_grant"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "permission",
            name="uq_platform_admin_perm_user_perm",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    permission: Mapped[str] = mapped_column(String(128), nullable=False)

    user = relationship("User", foreign_keys=[user_id])
