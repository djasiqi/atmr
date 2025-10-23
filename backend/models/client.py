# models/client.py
"""
Model Client - Gestion des clients / patients.
Extrait depuis models.py (lignes ~1177-1360).
"""
from __future__ import annotations

import re
from typing import cast

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import relationship, validates

from ext import db

from .base import _as_bool, _as_int, _iso
from .enums import ClientType


class Client(db.Model):
    __tablename__ = "client"
    __table_args__ = (
        UniqueConstraint("user_id", "company_id", name="uq_user_company"),
        Index("ix_client_company_active", "company_id", "is_active"),
        Index('ix_client_company_user', 'company_id', 'user_id'),
        Index('uq_client_user_no_company', 'user_id', unique=True, postgresql_where=text('company_id IS NULL')),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(ForeignKey('user.id', ondelete="CASCADE"), nullable=False, index=True)
    company_id = Column(ForeignKey('company.id', ondelete="SET NULL"), nullable=True, index=True)

    client_type = Column(
        SAEnum(ClientType, name="client_type"),
        nullable=False,
        default=ClientType.SELF_SERVICE,
        server_default=ClientType.SELF_SERVICE.value,
    )

    # Coordonnées de facturation/contacts
    billing_address = Column(String(255), nullable=True)
    billing_lat = Column(Numeric(10, 7), nullable=True)
    billing_lon = Column(Numeric(10, 7), nullable=True)
    contact_email = Column(String(100), nullable=True)
    contact_phone = Column(String(50), nullable=True)

    # Domiciliation
    domicile_address = Column(String(255), nullable=True)
    domicile_zip = Column(String(10), nullable=True)
    domicile_city = Column(String(100), nullable=True)
    domicile_lat = Column(Numeric(10, 7), nullable=True)
    domicile_lon = Column(Numeric(10, 7), nullable=True)

    # Accès logement
    door_code = Column(String(50), nullable=True)
    floor = Column(String(20), nullable=True)
    access_notes = Column(Text, nullable=True)

    # Médecin traitant
    gp_name = Column(String(120), nullable=True)
    gp_phone = Column(String(50), nullable=True)

    # Préférences de facturation par défaut
    default_billed_to_type = Column(String(50), nullable=False, server_default="patient")
    default_billed_to_company_id = Column(Integer, ForeignKey("company.id", ondelete="SET NULL"), nullable=True)
    default_billed_to_contact = Column(String(120), nullable=True)

    # Support institutions
    is_institution = Column(Boolean, nullable=False, server_default="false")
    institution_name = Column(String(200), nullable=True)

    # Établissement de résidence (EMS, Foyer, etc.)
    residence_facility = Column(String(200), nullable=True)  # Nom de l'établissement (ex: "EMS Maison de Vessy")

    # Tarif préférentiel
    preferential_rate = Column(Numeric(10, 2), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relations
    user = relationship("User", back_populates="clients", passive_deletes=True)
    company = relationship("Company", back_populates="clients", foreign_keys=[company_id], passive_deletes=True)
    default_billed_to_company = relationship("Company", back_populates="billed_clients", foreign_keys=[default_billed_to_company_id])
    bookings = relationship("Booking", back_populates="client", passive_deletes=True, lazy=True)
    payments = relationship("Payment", back_populates="client", passive_deletes=True, lazy=True)

    # Validators
    @validates("contact_email")
    def validate_contact_email(self, key, email):
        if cast(ClientType, self.client_type) == ClientType.SELF_SERVICE and not email:
            raise ValueError("L'email est requis pour les clients self-service.")
        if email:
            v = email.strip()
            if "@" not in v:
                raise ValueError("Email invalide.")
            return v
        return email

    @validates("billing_address")
    def validate_billing_address(self, key, value):
        cid = _as_int(getattr(self, "company_id", None), 0)
        if cid > 0 and (not value or not str(value).strip()):
            # Si pas d'adresse de facturation, utiliser l'adresse de domicile comme fallback
            domicile = getattr(self, "domicile_address", None)
            if domicile and str(domicile).strip():
                return domicile
            raise ValueError("L'adresse de facturation (ou de domicile) est obligatoire pour les clients liés à une entreprise.")
        return value

    @validates("contact_phone", "gp_phone")
    def validate_phone_numbers(self, key, value):
        if value:
            v = value.strip()
            digits = re.sub(r"\D", "", v)
            if len(digits) < 6:
                raise ValueError(f"{key} semble invalide.")
            return v
        return value

    @validates("default_billed_to_type")
    def validate_default_billed_to_type(self, key, val):
        val = (val or "patient").strip().lower()
        if val not in ("patient", "clinic", "insurance"):
            raise ValueError("default_billed_to_type invalide (patient|clinic|insurance)")
        if val == "patient":
            self.default_billed_to_company_id = None
        return val

    # Serialization
    @property
    def serialize(self):
        user = self.user
        first_name = getattr(user, "first_name", "") or ""
        last_name = getattr(user, "last_name", "") or ""
        username = getattr(user, "username", "") or ""
        phone_user = getattr(user, "phone", "") or ""
        full_name = (f"{first_name} {last_name}".strip() or username or "Nom non renseigné")

        return {
            "id": self.id,
            "user": user.serialize if user else None,
            "first_name": first_name,
            "last_name": last_name,
            "full_name": full_name,
            "client_type": self.client_type.value,
            "company_id": self.company_id,
            "billing_address": self.billing_address,
            "billing_lat": float(self.billing_lat) if self.billing_lat else None,
            "billing_lon": float(self.billing_lon) if self.billing_lon else None,
            "contact_email": self.contact_email,
            "phone": self.contact_phone or phone_user,
            "domicile": {
                "address": self.domicile_address,
                "zip": self.domicile_zip,
                "city": self.domicile_city,
                "lat": float(self.domicile_lat) if self.domicile_lat else None,
                "lon": float(self.domicile_lon) if self.domicile_lon else None,
            },
            "access": {
                "door_code": self.door_code,
                "floor": self.floor,
                "notes": self.access_notes,
            },
            "gp": {
                "name": self.gp_name,
                "phone": self.gp_phone,
            },
            "default_billing": {
                "billed_to_type": (self.default_billed_to_type or "patient"),
                "billed_to_company": (self.default_billed_to_company.serialize
                                      if self.default_billed_to_company else None),
                "billed_to_contact": self.default_billed_to_contact,
            },
            "is_institution": _as_bool(self.is_institution),
            "institution_name": self.institution_name,
            "residence_facility": self.residence_facility,
            "preferential_rate": float(self.preferential_rate) if self.preferential_rate else None,
            "is_active": _as_bool(self.is_active),
            "created_at": _iso(self.created_at),
        }

    def toggle_active(self) -> bool:
        current = _as_bool(getattr(self, "is_active", False))
        self.is_active = not current
        return bool(self.is_active)

    def is_self_service(self) -> bool:
        return cast(ClientType, self.client_type) == ClientType.SELF_SERVICE

    def __repr__(self):
        return f"<Client id={self.id}, user_id={self.user_id}, type={self.client_type}, active={self.is_active}>"

