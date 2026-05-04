# models/institution_transport_preference.py
# pyright: reportCallIssue=false
"""Model InstitutionTransportPreference - Préférences de transporteurs par institution.

Permet à une institution de définir un ordre de préférence pour les entreprises
de transport. Utilisé lors de l'envoi séquentiel des offres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso

if TYPE_CHECKING:
    from .company import Company
    from .institution import Institution


class InstitutionTransportPreference(db.Model):
    """Préférence de transporteur pour une institution.

    Définit l'ordre de priorité des entreprises de transport pour une institution.
    Order 1 = première préférence, 2 = seconde, etc.
    """

    __tablename__ = "institution_transport_preferences"
    __table_args__ = (
        # Une seule préférence par (institution, company)
        UniqueConstraint(
            "institution_id", "company_id", name="uq_institution_transport_preference"
        ),
        # Index pour requêtes par institution
        Index("ix_institution_transport_pref_institution", "institution_id"),
    )

    # Identifiant
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Institution
    institution_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Entreprise de transport
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Ordre de préférence (1 = premier choix)
    order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now(),
    )

    # Relations
    institution: Mapped[Institution] = relationship(
        "Institution",
        backref="transport_preferences",
    )
    company: Mapped[Company] = relationship(
        "Company",
        backref="institution_preferences",
    )

    @override
    def __repr__(self) -> str:
        return f"<InstitutionTransportPreference institution={self.institution_id} company={self.company_id} order={self.order}>"

    @property
    def serialize(self) -> dict[str, Any]:
        """Sérialise la préférence pour l'API."""
        return {
            "id": self.id,
            "institution_id": self.institution_id,
            "company_id": self.company_id,
            "company_name": self.company.name if self.company else None,
            "order": self.order,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias pour serialize."""
        return self.serialize

    @classmethod
    def get_ordered_preferences(
        cls, institution_id: int
    ) -> list[InstitutionTransportPreference]:
        """Récupère les préférences d'une institution, ordonnées."""
        return (
            cls.query.filter_by(institution_id=institution_id)
            .order_by(cls.order.asc())
            .all()
        )

    @classmethod
    def get_company_ids_ordered(cls, institution_id: int) -> list[int]:
        """Récupère les IDs des entreprises préférées, dans l'ordre."""
        prefs = cls.get_ordered_preferences(institution_id)
        return [p.company_id for p in prefs]

    @classmethod
    def has_preferences(cls, institution_id: int) -> bool:
        """Vérifie si une institution a des préférences définies."""
        return cls.query.filter_by(institution_id=institution_id).first() is not None

    @classmethod
    def get_next_preference_after(
        cls, institution_id: int, current_order: int
    ) -> InstitutionTransportPreference | None:
        """Récupère la préférence suivante après un ordre donné."""
        return (
            cls.query.filter(
                cls.institution_id == institution_id,
                cls.order > current_order,
            )
            .order_by(cls.order.asc())
            .first()
        )

    @classmethod
    def set_preferences(
        cls, institution_id: int, company_ids: list[int]
    ) -> list[InstitutionTransportPreference]:
        """Définit les préférences d'une institution (remplace les existantes).

        Args:
            institution_id: ID de l'institution
            company_ids: Liste ordonnée des IDs d'entreprises

        Returns:
            Liste des préférences créées
        """
        # Supprimer les préférences existantes
        cls.query.filter_by(institution_id=institution_id).delete()

        # Créer les nouvelles préférences
        preferences = []
        for order, company_id in enumerate(company_ids, start=1):
            pref = cls(
                institution_id=institution_id,
                company_id=company_id,
                order=order,
            )
            db.session.add(pref)
            preferences.append(pref)

        return preferences
