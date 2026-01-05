from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ext import db
from models import Vehicle


@dataclass(frozen=True, slots=True)
class SqlAlchemyVehicleWriter:
    """Adaptateur Infrastructure: création de Vehicle via SQLAlchemy."""

    def create_vehicle(self, *, company_id: int, attrs: dict[str, Any]) -> Vehicle:
        v = Vehicle()
        v.company_id = company_id

        # Champs principaux
        if "model" in attrs:
            v.model = attrs["model"]
        if "license_plate" in attrs:
            v.license_plate = attrs["license_plate"]
        if "year" in attrs:
            v.year = attrs["year"]
        if "vin" in attrs:
            v.vin = attrs["vin"]
        if "seats" in attrs:
            v.seats = attrs["seats"]

        # Champs accessibilité (compat)
        if "wheelchair_accessible" in attrs:
            v.wheelchair_accessible = bool(attrs["wheelchair_accessible"])

        if "is_active" in attrs:
            v.is_active = bool(attrs["is_active"])

        # Dates (optionnelles)
        if "insurance_expires_at" in attrs:
            v.insurance_expires_at = attrs["insurance_expires_at"]
        if "inspection_expires_at" in attrs:
            v.inspection_expires_at = attrs["inspection_expires_at"]

        db.session.add(v)
        return v
