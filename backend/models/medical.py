# models/medical.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import re
from typing import Any

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
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _iso

V_THRESHOLD = 90
LATITUDE_MAX = 90
LONGITUDE_MAX = 180
ADDRESS_MAX_LENGTH = 255
NAME_MAX_LENGTH = 200
DISPLAY_NAME_MAX_LENGTH = 255
TYPE_MAX_LENGTH = 50
MIN_PHONE_DIGITS = 6

"""Models liés aux établissements médicaux et lieux favoris.
Extrait depuis models.py (lignes ~2016-2393).
"""


class FavoritePlace(db.Model):
    __tablename__ = "favorite_place"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "address",
            name="uq_fav_company_address"),
        Index("ix_fav_company_label", "company_id", "label"),
        Index("ix_fav_company_coords", "company_id", "lat", "lon"),
        CheckConstraint("lat BETWEEN -90 AND 90", name="chk_fav_lat"),
        CheckConstraint("lon BETWEEN -180 AND 180", name="chk_fav_lon"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id = Column(
        Integer,
        ForeignKey(
            "company.id",
            ondelete="CASCADE"),
        nullable=False,
        index=True)

    label: Mapped[str] = mapped_column(String(200), nullable=False)
    address: Mapped[str] = mapped_column(String(255), nullable=False)

    lat: Mapped[float] = mapped_column(Float, nullable=True)
    lon: Mapped[float] = mapped_column(Float, nullable=True)

    tags = Column(String(200))

    created_at = Column(
        DateTime(
            timezone=True),
        server_default=func.now(),
        nullable=False)
    updated_at = Column(
        DateTime(
            timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False)

    # Normalisation & validations
    @staticmethod
    def _norm_text(s: str) -> str:
        return (s or "").strip()

    @staticmethod
    def _norm_address(s: str) -> str:
        s = (s or "").strip()
        return " ".join(s.split())

    @validates("label")
    def _v_label(self, _k, value):
        value = self._norm_text(value)
        if not value:
            msg = "Le champ 'label' ne peut pas être vide."
            raise ValueError(msg)
        return value

    @validates("address")
    def _v_address(self, _k, value):
        value = self._norm_address(value)
        if not value:
            msg = "Le champ 'address' ne peut pas être vide."
            raise ValueError(msg)
        if len(value) > ADDRESS_MAX_LENGTH:
            msg = f"Le champ 'address' dépasse {ADDRESS_MAX_LENGTH} caractères."
            raise ValueError(msg)
        return value

    @validates("lat")
    def _v_lat(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            msg = "Latitude invalide."
            raise ValueError(msg) from None
        if not (-LATITUDE_MAX <= v <= LATITUDE_MAX):
            msg = "Latitude hors bornes [-90; 90]."
            raise ValueError(msg)
        return v

    @validates("lon")
    def _v_lon(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            msg = "Longitude invalide."
            raise ValueError(msg) from None
        if not (-LONGITUDE_MAX <= v <= LONGITUDE_MAX):
            msg = "Longitude hors bornes [-180; 180]."
            raise ValueError(msg)
        return v

    @validates("tags")
    def _v_tags(self, _k, value):
        return self._norm_text(value)

    def to_dict(self) -> dict[str, Any]:
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

    @override
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

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    type = Column(String(50), nullable=False, default="hospital")
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    address: Mapped[str] = mapped_column(String(255), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=True)
    lon: Mapped[float] = mapped_column(Float, nullable=True)

    aliases: Mapped[str] = mapped_column(String(500), nullable=True)
    active = Column(Boolean, nullable=False, default=True)

    created_at = Column(
        DateTime(
            timezone=True),
        server_default=func.now(),
        nullable=False)
    updated_at = Column(
        DateTime(
            timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False)

    services = relationship(
        "MedicalService",
        backref="establishment",
        cascade="all, delete-orphan",
        passive_deletes=True)

    def alias_list(self):
        return [a.strip().lower()
                for a in (self.aliases or "").split(";") if a.strip()]

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

    @override
    def __repr__(self):
        return f"<MedicalEstablishment {self.name!r} @ {self.address!r}>"

    # Validations
    @validates("name", "display_name", "address")
    def _v_text_not_empty(self, key, value):
        v = (value or "").strip()
        if not v:
            msg = f"'{key}' ne peut pas être vide."
            raise ValueError(msg)
        if (key == "name" and len(v) > NAME_MAX_LENGTH) or (
                key in {"display_name", "address"} and len(v) > DISPLAY_NAME_MAX_LENGTH):
            msg = f"'{key}' dépasse la longueur maximale autorisée."
            raise ValueError(msg)
        return v

    @validates("type")
    def _v_type(self, _k, value):
        v = (value or "hospital").strip().lower()
        if len(v) > TYPE_MAX_LENGTH:
            msg = f"Le 'type' dépasse {TYPE_MAX_LENGTH} caractères."
            raise ValueError(msg)
        return v

    @validates("lat")
    def _v_lat(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            msg = "Latitude invalide."
            raise ValueError(msg) from None
        if not (-LATITUDE_MAX <= v <= LATITUDE_MAX):
            msg = "Latitude hors bornes [-90; 90]."
            raise ValueError(msg)
        return v

    @validates("lon")
    def _v_lon(self, _k, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            msg = "Longitude invalide."
            raise ValueError(msg) from None
        if not (-LONGITUDE_MAX <= v <= LONGITUDE_MAX):
            msg = "Longitude hors bornes [-180; 180]."
            raise ValueError(msg)
        return v


class MedicalService(db.Model):
    __tablename__ = "medical_service"
    __table_args__ = (
        UniqueConstraint(
            "establishment_id",
            "name",
            name="uq_med_service_per_estab"),
        CheckConstraint(
            "lat IS NULL OR (lat BETWEEN -90 AND 90)",
            name="chk_med_service_lat"),
        CheckConstraint(
            "lon IS NULL OR (lon BETWEEN -180 AND 180)",
            name="chk_med_service_lon"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    establishment_id = Column(
        Integer,
        ForeignKey(
            "medical_establishment.id",
            ondelete="CASCADE"),
        nullable=False,
        index=True)

    category = Column(String(50), nullable=False, default="Service")
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(200), nullable=True)

    # Localisation / contact
    address_line: Mapped[str] = mapped_column(String(255), nullable=True)
    postcode: Mapped[str] = mapped_column(String(16), nullable=True)
    city: Mapped[str] = mapped_column(String(100), nullable=True)
    building: Mapped[str] = mapped_column(String(120), nullable=True)
    floor: Mapped[str] = mapped_column(String(60), nullable=True)
    site_note: Mapped[str] = mapped_column(String(255), nullable=True)
    phone: Mapped[str] = mapped_column(String(40), nullable=True)
    email: Mapped[str] = mapped_column(String(120), nullable=True)

    lat: Mapped[float] = mapped_column(Float, nullable=True)
    lon: Mapped[float] = mapped_column(Float, nullable=True)

    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(
            timezone=True),
        server_default=func.now(),
        nullable=False)
    updated_at = Column(
        DateTime(
            timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False)

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

    @override
    def __repr__(self):
        return f"<MedicalService {self.name!r} ({self.category}) estab={self.establishment_id}>"

    # Validations
    @validates("name")
    def _v_name(self, _k, value):
        v = (value or "").strip()
        if not v:
            msg = "'name' ne peut pas être vide."
            raise ValueError(msg)
        if len(v) > NAME_MAX_LENGTH:
            msg = f"'name' dépasse {NAME_MAX_LENGTH} caractères."
            raise ValueError(msg)
        return v

    @validates("category")
    def _v_category(self, _k, value):
        v = (value or "Service").strip()
        if len(v) > TYPE_MAX_LENGTH:
            msg = f"'category' dépasse {TYPE_MAX_LENGTH} caractères."
            raise ValueError(msg)
        return v

    @validates("slug")
    def _v_slug(self, _k, value):
        if value is None:
            return value
        v = value.strip()
        if len(v) > NAME_MAX_LENGTH:
            msg = f"'slug' dépasse {NAME_MAX_LENGTH} caractères."
            raise ValueError(msg)
        return v

    @validates("email")
    def _v_email(self, _k, value):
        if not value:
            return value
        v = value.strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            msg = "Email invalide."
            raise ValueError(msg)
        return v

    @validates("phone")
    def _v_phone(self, _k, value):
        if not value:
            return value
        v = value.strip()
        digits = re.sub(r"\D", "", v)
        if len(digits) < MIN_PHONE_DIGITS:
            msg = f"Numéro de téléphone invalide (minimum {MIN_PHONE_DIGITS} chiffres)."
            raise ValueError(msg)
        return v

    @validates("lat")
    def _v_lat(self, _k, value):
        if value is None:
            return value
        try:
            v = float(value)
        except (TypeError, ValueError):
            msg = "Latitude invalide."
            raise ValueError(msg) from None
        if not (-LATITUDE_MAX <= v <= LATITUDE_MAX):
            msg = "Latitude hors bornes [-90; 90]."
            raise ValueError(msg)
        return v

    @validates("lon")
    def _v_lon(self, _k, value):
        if value is None:
            return value
        try:
            v = float(value)
        except (TypeError, ValueError):
            msg = "Longitude invalide."
            raise ValueError(msg) from None
        if not (-LONGITUDE_MAX <= v <= LONGITUDE_MAX):
            msg = "Longitude hors bornes [-180; 180]."
            raise ValueError(msg)
        return v
