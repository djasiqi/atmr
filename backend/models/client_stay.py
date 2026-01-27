"""Modèle ClientStay - Séjours (hospitalisation / établissement) d'un client.

Objectif: stocker une source structurée "client hospitalisé où et quand" afin
de fiabiliser la décision de facturation (clinique vs patient/tiers).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class ClientStay(db.Model):
    """Séjour d'un client dans un établissement (clinique/EMS).

    Notes:
        - `end_date` NULL signifie "séjour en cours".
        - `status` et `source` sont des champs de pilotage/traçabilité simples (V1).
    """

    __tablename__ = "client_stays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    client_id: Mapped[int] = mapped_column(
        ForeignKey("client.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )

    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Champ "léger" pour état du séjour (V1). Le statut est optionnel si `end_date`
    # suffit, mais utile pour annuler un séjour importé / erroné.
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="active"
    )

    # Source du séjour (saisie manuelle, import, détection...) pour audit/tri.
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relations
    client = relationship("Client", back_populates="stays", passive_deletes=True)
    company = relationship("Company", back_populates="client_stays", passive_deletes=True)
    created_by_user = relationship("User", passive_deletes=True)

    __table_args__ = (
        Index("ix_client_stays_client_start_date", "client_id", "start_date"),
        Index("ix_client_stays_company_start_date", "company_id", "start_date"),
    )

