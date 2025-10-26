# models/vehicle.py
# Constantes pour éviter les valeurs magiques
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _as_bool, _as_dt

SEATS_ZERO = 0
VALUE_ZERO = 0
VALUE_THRESHOLD = 2100
YEAR_MAX_VALUE = 2100
MIN_PLATE_LENGTH = 3

"""Model Vehicle - Gestion des véhicules des entreprises.
Extrait depuis models.py (lignes ~607-688).
"""


class Vehicle(db.Model):
    __tablename__ = "vehicle"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "license_plate",
            name="uq_company_plate"),
        CheckConstraint(
            "year IS NULL OR year BETWEEN 1950 AND 2100",
            name="chk_vehicle_year"),
        CheckConstraint(
            "seats IS NULL OR seats >= SEATS_ZERO",
            name="chk_vehicle_seats"),
        Index("ix_vehicle_company_active", "company_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id = Column(
        Integer,
        ForeignKey(
            "company.id",
            ondelete="CASCADE"),
        nullable=False,
        index=True)

    # Infos véhicule
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    license_plate: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vin: Mapped[str] = mapped_column(String(32), nullable=True)

    # Capacités
    seats: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    wheelchair_accessible = Column(
        Boolean, nullable=False, server_default="false")

    # Suivi administratif
    insurance_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    inspection_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(
        DateTime(
            timezone=True),
        nullable=False,
        server_default=func.now())

    # Relations
    company = relationship(
        "Company",
        back_populates="vehicles",
        passive_deletes=True)

    # Validations
    @validates("model")
    def _v_model(self,_key, value):
        if not value or not value.strip():
            msg = "Le modèle ne peut pas être vide."
            raise ValueError(msg)
        return value.strip()

    @validates("license_plate")
    def _v_license_plate(self,_key, plate):
        if not plate or len(plate.strip()) < MIN_PLATE_LENGTH:
            msg = "Numéro de plaque invalide."
            raise ValueError(msg)
        return plate.strip().upper()

    @validates("seats")
    def _v_seats(self, _key, value):
        if value is not None and value < VALUE_ZERO:
            msg = "Le nombre de places doit être ≥ 0."
            raise ValueError(msg)
        return value

    @validates("year")
    def _v_year(self, _key, value):
        if value is not None and (value < VALUE_THRESHOLD or value > YEAR_MAX_VALUE):
            msg = "Année du véhicule invalide."
            raise ValueError(msg)
        return value

    # Sérialisation
    @property
    def serialize(self):
        ins_dt = _as_dt(self.insurance_expires_at)
        insp_dt = _as_dt(self.inspection_expires_at)
        created_dt = _as_dt(self.created_at)
        return {
            "id": self.id,
            "company_id": self.company_id,
            "model": self.model,
            "license_plate": self.license_plate,
            "year": self.year,
            "vin": self.vin,
            "seats": self.seats,
            "wheelchair_accessible": _as_bool(self.wheelchair_accessible),
            "insurance_expires_at": ins_dt.isoformat() if ins_dt else None,
            "inspection_expires_at": insp_dt.isoformat() if insp_dt else None,
            "is_active": _as_bool(self.is_active),
            "created_at": created_dt.isoformat() if created_dt else None,
        }

    @override
    def __repr__(self):
        return f"<Vehicle id={self.id} plate={self.license_plate} model={self.model} company_id={self.company_id}>"
