# models/institution.py
"""Model Institution - Gestion des institutions (cliniques, EMS, IMAD, hôpitaux).

Nouveau tenant distinct de Company pour le portail institutionnel.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso


class Institution(db.Model):
    """Modèle Institution - Tenant institutionnel (clinique/EMS/IMAD/hôpital).

    Une institution peut avoir plusieurs utilisateurs avec différents rôles
    (admin, requester, reader, billing).
    """

    __tablename__ = "institutions"

    # Identifiant
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id = Column(
        String(36),
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )

    # Informations de base
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    institution_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # clinic, ems, imad, hospital, curatelle, other
    address: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Contact
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Facturation
    billing_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    billing_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    vat_number: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Métadonnées additionnelles
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relations
    # Un institution peut avoir plusieurs utilisateurs
    users = relationship(
        "User",
        back_populates="institution",
        foreign_keys="User.institution_id",
    )

    @override
    def __repr__(self) -> str:
        return f"<Institution {self.id}: {self.name} ({self.institution_type})>"

    @property
    def serialize(self) -> dict[str, Any]:
        """Sérialise l'institution pour l'API."""
        return {
            "id": self.id,
            "public_id": self.public_id,
            "name": self.name,
            "institution_type": self.institution_type,
            "address": self.address,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "billing_email": self.billing_email,
            "billing_address": self.billing_address,
            "vat_number": self.vat_number,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias pour serialize."""
        return self.serialize
