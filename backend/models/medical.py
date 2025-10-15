# models/medical.py
"""
Models liés aux établissements médicaux et lieux favoris.
Extrait depuis models.py (lignes ~2016-2393).
"""
from __future__ import annotations

import re

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship, validates

from ext import db

from .base import _iso


class FavoritePlace(db.Model):
    __tablename__ = "favorite_place"
    __table_args__ = (
        UniqueConstraint("company_id", "address", name="uq_fav_company_address"),
        Index("ix_fav_company_label", "company_id", "label"),
        Index("ix_fav_company_coords", "company_id", "lat", "lon"),
        CheckConstraint("lat BETWEEN -90 AND 90", name="chk_fav_lat"),
        CheckConstraint("lon BETWEEN -180 AND 180", name="chk_fav_lon"),
    )

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)

    label = Column(String(200), nullable=False)
    address = Column(String(255), nullable=False)

    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    tags = Column(String(200))

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Normalisation & validations
    @staticmethod
    def _norm_text(s: str) -> str:
        return (s or "").strip()

    @staticmethod
    def _norm_address(s: str) -> str:
        s = (s or "").strip()
        s = " ".join(s.split())
        return s

    @validates("label")
    def _v_label(self, _k, value):
        value = self._norm_text(value)
        if not value:
            raise ValueError("Le champ 'label' ne peut pas être vide.")
        return value

    @validates("address")
    def _v_address(self, _k, value):
        value = self._norm_address(value)
        if not value:
            raise ValueError("Le champ 'address' ne peut pas être vide.")
        if len(value) > 255:
            raise ValueError("Le champ 'address' dépasse 255 caractères.")
        return value

    @validates("lat")
    def _v_lat(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Latitude invalide.")
        if not (-90.0 <= v <= 90.0):
            raise ValueError("Latitude hors bornes [-90; 90].")
        return v

    @validates("lon")
    def _v_lon(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Longitude invalide.")
        if not (-180.0 <= v <= 180.0):
            raise ValueError("Longitude hors bornes [-180; 180].")
        return v

    @validates("tags")
    def _v_tags(self, _k, value):
        return self._norm_text(value)

    def to_dict(self) -> dict:
        """Sérialisation standard (API)."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "label": self.label,
            "address": self.address,
            "lat": self.lat,
            "lon": self.lon,
            "tags": self.tags,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }

    @property
    def serialize(self):
        return {
            "id": self.id,
            "label": self.label,
            "address": self.address,
            "lat": self.lat,
            "lon": self.lon,
            "tags": self.tags,
        }

    def __repr__(self) -> str:
        return f"<FavoritePlace {self.label!r} @ {self.address!r} (company={self.company_id})>"


class MedicalEstablishment(db.Model):
    __tablename__ = "medical_establishment"
    __table_args__ = (
        UniqueConstraint("name", name="uq_med_estab_name"),
        UniqueConstraint("address", name="uq_med_estab_address"),
        CheckConstraint("lat BETWEEN -90 AND 90", name="chk_med_estab_lat"),
        CheckConstraint("lon BETWEEN -180 AND 180", name="chk_med_estab_lon"),
    )

    id = Column(Integer, primary_key=True)

    type = Column(String(50), nullable=False, default="hospital")
    name = Column(String(200), nullable=False)
    display_name = Column(String(255), nullable=False)
    address = Column(String(255), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    aliases = Column(String(500), nullable=True)
    active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    services = relationship("MedicalService", backref="establishment", cascade="all, delete-orphan", passive_deletes=True)

    def alias_list(self):
        return [a.strip().lower() for a in (self.aliases or "").split(";") if a.strip()]

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "display_name": self.display_name,
            "address": self.address,
            "lat": self.lat,
            "lon": self.lon,
            "aliases": self.alias_list(),
            "active": self.active,
        }

    @property
    def serialize(self):
        return self.to_dict()

    def __repr__(self):
        return f"<MedicalEstablishment {self.name!r} @ {self.address!r}>"

    # Validations
    @validates("name", "display_name", "address")
    def _v_text_not_empty(self, key, value):
        v = (value or "").strip()
        if not v:
            raise ValueError(f"'{key}' ne peut pas être vide.")
        if (key == "name" and len(v) > 200) or (key in {"display_name", "address"} and len(v) > 255):
            raise ValueError(f"'{key}' dépasse la longueur maximale autorisée.")
        return v

    @validates("type")
    def _v_type(self, _k, value):
        v = (value or "hospital").strip().lower()
        if len(v) > 50:
            raise ValueError("Le 'type' dépasse 50 caractères.")
        return v

    @validates("lat")
    def _v_lat(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Latitude invalide.")
        if not (-90.0 <= v <= 90.0):
            raise ValueError("Latitude hors bornes [-90; 90].")
        return v

    @validates("lon")
    def _v_lon(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Longitude invalide.")
        if not (-180.0 <= v <= 180.0):
            raise ValueError("Longitude hors bornes [-180; 180].")
        return v


class MedicalService(db.Model):
    __tablename__ = "medical_service"
    __table_args__ = (
        UniqueConstraint("establishment_id", "name", name="uq_med_service_per_estab"),
        CheckConstraint("lat IS NULL OR (lat BETWEEN -90 AND 90)", name="chk_med_service_lat"),
        CheckConstraint("lon IS NULL OR (lon BETWEEN -180 AND 180)", name="chk_med_service_lon"),
    )

    id = Column(Integer, primary_key=True)
    establishment_id = Column(Integer, ForeignKey("medical_establishment.id", ondelete="CASCADE"), nullable=False, index=True)

    category = Column(String(50), nullable=False, default="Service")
    name = Column(String(200), nullable=False)
    slug = Column(String(200), nullable=True)

    # Localisation / contact
    address_line = Column(String(255), nullable=True)
    postcode = Column(String(16), nullable=True)
    city = Column(String(100), nullable=True)
    building = Column(String(120), nullable=True)
    floor = Column(String(60), nullable=True)
    site_note = Column(String(255), nullable=True)
    phone = Column(String(40), nullable=True)
    email = Column(String(120), nullable=True)

    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)

    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "establishment_id": self.establishment_id,
            "category": self.category,
            "name": self.name,
            "slug": self.slug,
            "address_line": self.address_line,
            "postcode": self.postcode,
            "city": self.city,
            "building": self.building,
            "floor": self.floor,
            "site_note": self.site_note,
            "phone": self.phone,
            "email": self.email,
            "lat": self.lat,
            "lon": self.lon,
            "active": self.active,
        }

    @property
    def serialize(self):
        return self.to_dict()

    def __repr__(self):
        return f"<MedicalService {self.name!r} ({self.category}) estab={self.establishment_id}>"

    # Validations
    @validates("name")
    def _v_name(self, _k, value):
        v = (value or "").strip()
        if not v:
            raise ValueError("'name' ne peut pas être vide.")
        if len(v) > 200:
            raise ValueError("'name' dépasse 200 caractères.")
        return v

    @validates("category")
    def _v_category(self, _k, value):
        v = (value or "Service").strip()
        if len(v) > 50:
            raise ValueError("'category' dépasse 50 caractères.")
        return v

    @validates("slug")
    def _v_slug(self, _k, value):
        if value is None:
            return value
        v = value.strip()
        if len(v) > 200:
            raise ValueError("'slug' dépasse 200 caractères.")
        return v

    @validates("email")
    def _v_email(self, _k, value):
        if not value:
            return value
        v = value.strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Email invalide.")
        return v

    @validates("phone")
    def _v_phone(self, _k, value):
        if not value:
            return value
        v = value.strip()
        digits = re.sub(r"\D", "", v)
        if len(digits) < 6:
            raise ValueError("Numéro de téléphone invalide.")
        return v

    @validates("lat")
    def _v_lat(self, _k, value):
        if value is None:
            return value
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Latitude invalide.")
        if not (-90.0 <= v <= 90.0):
            raise ValueError("Latitude hors bornes [-90; 90].")
        return v

    @validates("lon")
    def _v_lon(self, _k, value):
        if value is None:
            return value
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Longitude invalide.")
        if not (-180.0 <= v <= 180.0):
            raise ValueError("Longitude hors bornes [-180; 180].")
        return v

