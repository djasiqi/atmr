# models/vehicle.py
"""
Model Vehicle - Gestion des véhicules des entreprises.
Extrait depuis models.py (lignes ~607-688).
"""
from __future__ import annotations

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
from sqlalchemy.orm import relationship, validates

from ext import db

from .base import _as_bool, _as_dt


class Vehicle(db.Model):
    __tablename__ = "vehicle"
    __table_args__ = (
        UniqueConstraint("company_id", "license_plate", name="uq_company_plate"),
        CheckConstraint("year IS NULL OR year BETWEEN 1950 AND 2100", name="chk_vehicle_year"),
        CheckConstraint("seats IS NULL OR seats >= 0", name="chk_vehicle_seats"),
        Index("ix_vehicle_company_active", "company_id", "is_active"),
    )

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)

    # Infos véhicule
    model = Column(String(120), nullable=False)
    license_plate = Column(String(20), nullable=False, index=True)
    year = Column(Integer, nullable=True)
    vin = Column(String(32), nullable=True)

    # Capacités
    seats = Column(Integer, nullable=True)
    wheelchair_accessible = Column(Boolean, nullable=False, server_default="false")

    # Suivi administratif
    insurance_expires_at = Column(DateTime(timezone=True), nullable=True)
    inspection_expires_at = Column(DateTime(timezone=True), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relations
    company = relationship("Company", back_populates="vehicles", passive_deletes=True)

    # Validations
    @validates("model")
    def _v_model(self, _key, value):
        if not value or not value.strip():
            raise ValueError("Le modèle ne peut pas être vide.")
        return value.strip()

    @validates("license_plate")
    def _v_license_plate(self, _key, plate):
        if not plate or len(plate.strip()) < 3:
            raise ValueError("Numéro de plaque invalide.")
        return plate.strip().upper()

    @validates("seats")
    def _v_seats(self, _key, value):
        if value is not None and value < 0:
            raise ValueError("Le nombre de places doit être ≥ 0.")
        return value

    @validates("year")
    def _v_year(self, _key, value):
        if value is not None and (value < 1950 or value > 2100):
            raise ValueError("Année du véhicule invalide.")
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

    def __repr__(self):
        return f"<Vehicle id={self.id} plate={self.license_plate} model={self.model} company_id={self.company_id}>"

