"""Journal d'audit facturation (V1).

Objectif:
- tracer toute modification de décision de facturation (payeur, verrouillage, etc.)
- fournir auteur + motif + timestamp, conformément aux règles "pro" (avant/après émission).
"""

from __future__ import annotations

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class BillingAuditLog(db.Model):
    __tablename__ = "billing_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )
    booking_id: Mapped[int] = mapped_column(
        ForeignKey("booking.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Auteur (utilisateur backoffice/admin). Nullable pour imports ou jobs.
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Type d'action (string pour V1: flexible)
    action: Mapped[str] = mapped_column(Text, nullable=False)

    # Motif obligatoire côté API quand on modifie le payeur / lock / unlock.
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Snapshot avant/après (V1: JSONB libre, facile à étendre)
    before: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    after: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relations (optionnelles)
    company = relationship("Company", passive_deletes=True)
    booking = relationship("Booking", passive_deletes=True)
    actor_user = relationship("User", passive_deletes=True)
