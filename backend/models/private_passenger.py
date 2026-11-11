"""Carnet privé de proches d'un titulaire de compte client.

Distinct des patients institutionnels. Aucune donnée médicale.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class PrivatePassenger(db.Model):
    __tablename__ = "private_passenger"
    __table_args__ = (Index("ix_private_passenger_owner", "owner_user_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=lambda: str(uuid.uuid4())
    )
    owner_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(30), nullable=True)
    access_notes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    wheelchair_need: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    assistance_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    assistance_note: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def serialize(self) -> dict[str, object]:
        return {
            "public_id": self.public_id,
            "display_name": self.display_name,
            "phone": self.phone,
            "access_notes": self.access_notes,
            "wheelchair_need": bool(self.wheelchair_need),
            "assistance_required": bool(self.assistance_required),
            "assistance_note": self.assistance_note
            if self.assistance_required
            else None,
        }
