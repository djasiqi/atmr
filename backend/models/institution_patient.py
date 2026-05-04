# models/institution_patient.py
"""Model InstitutionPatient - Patients gérés par les institutions.

Distinct de Client qui est lié à Company. Les institutions ont leur propre
base de patients avec external_reference pour mapping DPI.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _iso
from .enums import GenderEnum

if TYPE_CHECKING:
    from .institution import Institution


class InstitutionPatient(db.Model):
    """Patient appartenant à une institution.

    Chaque institution maintient sa propre base de patients, distincte
    des clients Company. Les patients peuvent avoir une external_reference
    unique par institution pour mapping avec le DPI.
    """

    __tablename__ = "institution_patients"
    __table_args__ = (
        # Index sur institution_id pour requêtes fréquentes
        Index("ix_institution_patients_institution_id", "institution_id"),
        # Unique externe_reference par institution (si présente)
        Index(
            "uq_institution_patient_external_ref",
            "institution_id",
            "external_reference",
            unique=True,
            postgresql_where="external_reference IS NOT NULL",
        ),
        # Index pour recherche par nom
        Index(
            "ix_institution_patients_name",
            "institution_id",
            "last_name",
            "first_name",
        ),
    )

    # Identifiant
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id = Column(
        String(36),
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )

    # Institution propriétaire
    institution_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Référence externe DPI (unique par institution si présente)
    external_reference: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )

    # Informations patient
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Coordonnées
    address: Mapped[str | None] = mapped_column(String(255), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    postal_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Accès & logistique (infos critiques pour le chauffeur)
    door_code: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Code porte / digicode"
    )
    floor: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Étage (ex: 3, RDC, 2B)"
    )
    access_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Notes d'accès (ascenseur, rampe, concierge...)"
    )
    residence_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Établissement de résidence (EMS, foyer, etc.)"
    )

    # Informations administratives (facturation, identification, curatelle)
    avs_number: Mapped[str | None] = mapped_column(
        String(16), nullable=True, comment="Numéro AVS (756.XXXX.XXXX.XX)"
    )
    insurance_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Nom de la caisse maladie"
    )
    insurance_number: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Numéro d'assuré"
    )
    has_guardianship: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false",
        comment="Patient sous curatelle",
    )
    guardianship_type: Mapped[str | None] = mapped_column(
        String(30), nullable=True,
        comment="Type de curatelle: curatorship, opad, lawyer, family, other",
    )
    guardian_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Nom du curateur / représentant légal"
    )
    guardian_organization: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
        comment="Organisation du curateur (OPAD Genève, Étude Me. Dupont, etc.)",
    )
    guardian_phone: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Téléphone du curateur"
    )
    guardian_email: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Email du curateur"
    )
    guardian_address: Mapped[str | None] = mapped_column(
        String(500), nullable=True,
        comment="Adresse complète du curateur (utilisée pour facturation)",
    )

    # Équipe de curateurs assignée (curatelle uniquement)
    curator_team_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("curator_teams.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Équipe de curateurs en charge de ce patient",
    )

    # Informations additionnelles
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Traçabilité sync curatelle : quels champs proviennent d'une synchronisation
    data_source_flags: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
        comment='Ex: {"address": "sync_curatelle", "phone": "local"}',
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relations
    institution: Mapped[Institution] = relationship(
        "Institution",
        backref="patients",
    )

    @override
    def __repr__(self) -> str:
        return f"<InstitutionPatient {self.id}: {self.last_name} {self.first_name}>"

    @validates("gender")
    def validate_gender(self, _key: str, value: str | None) -> str | None:
        """Valide le genre."""
        if value is None:
            return None
        valid_genders = [g.value for g in GenderEnum] + [g.name for g in GenderEnum]
        if value.upper() not in [v.upper() for v in valid_genders]:
            raise ValueError(f"Gender must be one of: {', '.join(GenderEnum._member_names_)}")
        return value.upper()

    @property
    def full_name(self) -> str:
        """Retourne le nom complet."""
        return f"{self.first_name} {self.last_name}"

    @property
    def serialize(self) -> dict[str, Any]:
        """Sérialise le patient pour l'API."""
        return {
            "id": self.id,
            "public_id": self.public_id,
            "external_reference": self.external_reference,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "dob": self.dob.isoformat() if self.dob else None,
            "gender": self.gender,
            "address": self.address,
            "city": self.city,
            "postal_code": self.postal_code,
            "phone": self.phone,
            "door_code": self.door_code,
            "floor": self.floor,
            "access_notes": self.access_notes,
            "residence_name": self.residence_name,
            "avs_number": self.avs_number,
            "insurance_name": self.insurance_name,
            "insurance_number": self.insurance_number,
            "has_guardianship": self.has_guardianship,
            "guardianship_type": self.guardianship_type,
            "guardian_name": self.guardian_name,
            "guardian_organization": self.guardian_organization,
            "guardian_phone": self.guardian_phone,
            "guardian_email": self.guardian_email,
            "guardian_address": self.guardian_address,
            "notes": self.notes,
            "curator_team_id": self.curator_team_id,
            "data_source_flags": self.data_source_flags,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias pour serialize."""
        return self.serialize

    @classmethod
    def find_by_external_reference(
        cls, institution_id: int, external_reference: str
    ) -> InstitutionPatient | None:
        """Trouve un patient par référence externe.

        Args:
            institution_id: ID de l'institution
            external_reference: Référence externe DPI

        Returns:
            InstitutionPatient ou None
        """
        return cls.query.filter_by(
            institution_id=institution_id,
            external_reference=external_reference,
        ).first()
