# models/company.py
"""
Model Company - Gestion des entreprises de transport.
Extrait depuis models.py (lignes ~420-600).
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, relationship, validates

from ext import db

from .base import _as_bool, _as_dt

if TYPE_CHECKING:
    from .dispatch import DailyStats, DispatchMetrics, DispatchRun


class Company(db.Model):
    __tablename__ = "company"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)

    # Adresse opérationnelle
    address = Column(String(200), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Adresse de domiciliation
    domicile_address_line1 = Column(String(200), nullable=True)
    domicile_address_line2 = Column(String(200), nullable=True)
    domicile_zip = Column(String(10), nullable=True)
    domicile_city = Column(String(100), nullable=True)
    domicile_country = Column(String(2), nullable=True, server_default="CH")

    # Contact
    contact_email = Column(String(100), nullable=True)
    contact_phone = Column(String(20), nullable=True)

    # Légal & Facturation
    iban = Column(String(34), nullable=True, index=True)
    uid_ide = Column(String(20), nullable=True, index=True)
    billing_email = Column(String(100), nullable=True)
    billing_notes = Column(Text, nullable=True)

    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE", name="fk_company_user"), nullable=False, index=True)
    is_approved = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    service_area = Column(String(200), nullable=True)
    max_daily_bookings = Column(Integer, nullable=True, server_default="50")
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    dispatch_enabled = Column(Boolean, nullable=False, server_default="false")
    is_partner = Column(Boolean, nullable=False, server_default="false")
    logo_url = Column(String(255), nullable=True)

    # Relations
    user = relationship("User", back_populates="company", passive_deletes=True)
    clients = relationship("Client", back_populates="company", cascade="all, delete-orphan", passive_deletes=True, foreign_keys="Client.company_id", primaryjoin="Company.id == Client.company_id")
    billed_clients = relationship("Client", back_populates="default_billed_to_company", foreign_keys="Client.default_billed_to_company_id", primaryjoin="Company.id == Client.default_billed_to_company_id")
    drivers = relationship("Driver", back_populates="company", passive_deletes=True)
    dispatch_runs: Mapped[List[DispatchRun]] = relationship("DispatchRun", back_populates="company", cascade="all, delete-orphan", passive_deletes=True)
    dispatch_metrics: Mapped[List[DispatchMetrics]] = relationship("DispatchMetrics", back_populates="company", cascade="all, delete-orphan", passive_deletes=True)
    daily_stats: Mapped[List[DailyStats]] = relationship("DailyStats", back_populates="company", cascade="all, delete-orphan", passive_deletes=True)
    bookings = relationship("Booking", back_populates="company", foreign_keys="Booking.company_id", passive_deletes=True)
    vehicles = relationship("Vehicle", back_populates="company", cascade="all, delete-orphan", passive_deletes=True)

    @property
    def serialize(self):
        created_dt = _as_dt(self.created_at)
        accepted_dt = _as_dt(self.accepted_at)
        return {
            "id": self.id,
            "name": self.name,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "domicile": {
                "line1": self.domicile_address_line1,
                "line2": self.domicile_address_line2,
                "zip": self.domicile_zip,
                "city": self.domicile_city,
                "country": self.domicile_country,
            },
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "iban": self.iban,
            "uid_ide": self.uid_ide,
            "billing_email": self.billing_email,
            "billing_notes": self.billing_notes,
            "logo_url": self.logo_url,
            "is_approved": _as_bool(self.is_approved),
            "is_partner": _as_bool(self.is_partner),
            "user_id": self.user_id,
            "service_area": self.service_area,
            "max_daily_bookings": self.max_daily_bookings,
            "created_at": created_dt.isoformat() if created_dt else None,
            "dispatch_enabled": _as_bool(self.dispatch_enabled),
            "accepted_at": accepted_dt.isoformat() if accepted_dt else None,
            "vehicles": [v.serialize for v in self.vehicles],
        }

    @validates("contact_email", "billing_email")
    def validate_any_email(self, key, value):
        if not value:
            return value
        v = value.strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError(f"Format d'email invalide pour {key}.")
        return v

    @validates("contact_phone")
    def validate_contact_phone(self, key, value):
        if not value:
            return value
        v = value.strip()
        if not re.match(r"^\+?[0-9\s\-\(\)]{7,20}$", v):
            raise ValueError("Numéro de téléphone invalide.")
        return v

    @validates("iban")
    def validate_iban(self, key, value):
        if not value:
            return value
        v = value.replace(" ", "").upper()
        if len(v) < 15 or len(v) > 34 or not v[:2].isalpha() or not v[2:4].isdigit():
            raise ValueError("IBAN invalide (format).")
        rearranged = v[4:] + v[:4]
        try:
            converted = "".join(str(int(ch, 36)) for ch in rearranged)
        except ValueError:
            raise ValueError("IBAN invalide (caractères non autorisés).")
        remainder = 0
        for i in range(0, len(converted), 9):
            remainder = int(str(remainder) + converted[i:i+9]) % 97
        if remainder != 1:
            raise ValueError("IBAN invalide (checksum).")
        return v

    @validates("uid_ide")
    def validate_uid_ide(self, key, value):
        if not value:
            return value
        v = value.strip().upper()
        if not re.match(r"^CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?$|^CHE[- ]?\d{9}(\s*TVA)?$", v, flags=re.IGNORECASE):
            raise ValueError("IDE/UID suisse invalide (ex: CHE-123.456.789).")
        digits = re.sub(r"\D", "", v)[:9]
        v_norm = f"CHE-{digits[0:3]}.{digits[3:6]}.{digits[6:9]}"
        if "TVA" in v:
            v_norm += " TVA"
        return v_norm

    @validates("name")
    def validate_name(self, key, value):
        if not value or len(value.strip()) == 0:
            raise ValueError("Le nom de l'entreprise ne peut pas être vide.")
        if len(value) > 100:
            raise ValueError("Le nom de l'entreprise ne peut pas dépasser 100 caractères.")
        return value.strip()

    @validates("user_id")
    def validate_user_id(self, key, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ID utilisateur invalide.")
        return value

    def toggle_approval(self) -> bool:
        self.is_approved = not _as_bool(self.is_approved)
        return _as_bool(self.is_approved)

    def can_dispatch(self) -> bool:
        return _as_bool(self.is_approved) and _as_bool(self.dispatch_enabled)

    def approve(self):
        self.is_approved = True
        self.accepted_at = datetime.now(UTC)

    def __repr__(self):
        return f"<Company {self.name} | ID: {self.id} | Approved: {self.is_approved}>"

    def to_dict(self):
        return self.serialize.copy()

