# models/web_session.py
"""Session web durable — autorité de révocation partagée (claim JWT sid).

Une WebSession survit aux rotations de refresh_token.
Utilisée pour le contrôle d'inactivité humaine des comptes institution.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso

if TYPE_CHECKING:
    pass


class WebSession(db.Model):
    __tablename__ = "web_session"
    __table_args__ = (
        Index("ix_web_session_user_id", "user_id"),
        Index("ix_web_session_institution_id", "institution_id"),
        Index("ix_web_session_revoked_at", "revoked_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
    )
    institution_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    last_interactive_activity_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)

    user = relationship("User", backref="web_sessions")

    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    def is_active(self) -> bool:
        if self.is_revoked():
            return False
        now = datetime.now(UTC)
        return now < self.expires_at

    def serialize(self) -> dict[str, object]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "institution_id": self.institution_id,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "last_interactive_activity_at": (
                _iso(self.last_interactive_activity_at)
                if self.last_interactive_activity_at
                else None
            ),
            "revoked_at": _iso(self.revoked_at) if self.revoked_at else None,
        }

    @override
    def __repr__(self) -> str:
        return f"<WebSession {self.id} user_id={self.user_id} revoked={self.is_revoked()}>"
