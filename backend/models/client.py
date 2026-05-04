# models/client.py
# pyright: reportUnnecessaryTypeIgnoreComment=false

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import logging
import re
from decimal import Decimal
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
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _as_bool, _as_int, _iso
from .enums import ClientType, ManagementMode

logger = logging.getLogger(__name__)

CID_ZERO = 0
MIN_PHONE_DIGITS = 6

"""Model Client - Gestion des clients / patients.
Extrait depuis models.py (lignes ~1177-1360).
"""


class Client(db.Model):
    __tablename__ = "client"
    __table_args__ = (
        UniqueConstraint("user_id", "company_id", name="uq_user_company"),
        Index("ix_client_company_active", "company_id", "is_active"),
        Index("ix_client_company_user", "company_id", "user_id"),
        Index(
            "uq_client_user_no_company",
            "user_id",
            unique=True,
            postgresql_where=text("company_id IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"), nullable=True, index=True
    )

    client_type: Mapped[ClientType] = mapped_column(
        SAEnum(ClientType, name="client_type"),
        nullable=False,
        default=ClientType.PORTAL,
        server_default=ClientType.PORTAL.value,
    )
    management_mode: Mapped[ManagementMode | None] = mapped_column(
        SAEnum(ManagementMode, name="management_mode"),
        nullable=True,
        default=None,
    )

    # Coordonnées de facturation/contacts
    billing_address: Mapped[str | None] = mapped_column(String(255), nullable=True)
    billing_lat: Mapped[Decimal | None] = mapped_column(Numeric(10, 7), nullable=True)
    billing_lon: Mapped[Decimal | None] = mapped_column(Numeric(10, 7), nullable=True)
    contact_email: Mapped[str | None] = mapped_column(String(100), nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Domiciliation
    domicile_address: Mapped[str] = mapped_column(String(255), nullable=True)
    domicile_zip: Mapped[str] = mapped_column(String(10), nullable=True)
    domicile_city: Mapped[str] = mapped_column(String(100), nullable=True)
    domicile_lat: Mapped[Decimal] = mapped_column(Numeric(10, 7), nullable=True)
    domicile_lon: Mapped[Decimal] = mapped_column(Numeric(10, 7), nullable=True)

    # Accès logement
    door_code: Mapped[str] = mapped_column(String(50), nullable=True)
    floor: Mapped[str] = mapped_column(String(20), nullable=True)
    access_notes: Mapped[str] = mapped_column(Text, nullable=True)

    # Médecin traitant
    gp_name: Mapped[str] = mapped_column(String(120), nullable=True)
    gp_phone: Mapped[str] = mapped_column(String(50), nullable=True)

    # Préférences de facturation par défaut
    default_billed_to_type = Column(
        String(50), nullable=False, server_default="patient"
    )
    default_billed_to_company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="SET NULL"), nullable=True
    )
    default_billed_to_contact: Mapped[str] = mapped_column(String(120), nullable=True)

    # Support institutions
    is_institution = Column(Boolean, nullable=False, server_default="false")
    institution_name: Mapped[str] = mapped_column(String(200), nullable=True)

    # Liaison vers une Institution officielle de la plateforme
    linked_institution_id: Mapped[int | None] = mapped_column(
        ForeignKey("institutions.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Établissement de résidence (EMS, Foyer, etc.)
    # Nom de l'établissement (ex: "EMS Maison de Vessy")
    residence_facility: Mapped[str] = mapped_column(String(200), nullable=True)

    # Numéro AVS (assurance vieillesse et survivants)
    avs_number: Mapped[str] = mapped_column(String(20), nullable=True)

    # Tarif préférentiel
    preferential_rate: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ✅ D2: Colonnes chiffrées
    contact_phone_encrypted: Mapped[str] = mapped_column(Text, nullable=True)
    gp_name_encrypted: Mapped[str] = mapped_column(Text, nullable=True)
    gp_phone_encrypted: Mapped[str] = mapped_column(Text, nullable=True)
    billing_address_encrypted: Mapped[str] = mapped_column(Text, nullable=True)
    encryption_migrated = Column(Boolean, default=False, nullable=False)

    # Relations
    user = relationship("User", back_populates="clients", passive_deletes=True)
    company = relationship(
        "Company",
        back_populates="clients",
        foreign_keys=[company_id],
        passive_deletes=True,
    )
    linked_institution = relationship(
        "Institution",
        foreign_keys=[linked_institution_id],
        lazy="joined",
    )
    default_billed_to_company = relationship(
        "Company",
        back_populates="billed_clients",
        foreign_keys=[default_billed_to_company_id],
    )
    bookings = relationship(
        "Booking", back_populates="client", passive_deletes=True, lazy=True
    )
    payments = relationship(
        "Payment", back_populates="client", passive_deletes=True, lazy=True
    )
    stays = relationship(
        "ClientStay",
        back_populates="client",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy=True,
    )
    transport_vouchers = relationship(
        "TransportVoucher",
        back_populates="client",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy=True,
    )

    # Validators
    @validates("contact_email")
    def validate_contact_email(self, _key: str, email: str) -> str:
        if self.management_mode == ManagementMode.SELF_SERVICE and not email:
            msg = "L'email est requis pour les clients self-service."
            raise ValueError(msg)
        if email:
            v = email.strip()
            if "@" not in v:
                msg = "Email invalide."
                raise ValueError(msg)
            return v
        return email

    @validates("billing_address")
    def validate_billing_address(self, _key: str, value: str) -> str:
        cid = _as_int(getattr(self, "company_id", None))
        if cid > CID_ZERO and (not value or not str(value).strip()):
            # Si pas d'adresse de facturation, utiliser l'adresse de domicile
            # comme fallback
            domicile = getattr(self, "domicile_address", None)
            if domicile and str(domicile).strip():
                return cast(str, domicile)
            msg = (
                "L'adresse de facturation (ou de domicile) est obligatoire "
                "pour les clients liés à une entreprise."
            )
            raise ValueError(msg)
        return value

    @validates("contact_phone", "gp_phone")
    def validate_phone_numbers(self, key: str, value: str) -> str:
        if value:
            v = value.strip()
            digits = re.sub(r"\D", "", v)
            if len(digits) < MIN_PHONE_DIGITS:
                msg = f"{key} semble invalide."
                raise ValueError(msg)
            return v
        return value

    @validates("default_billed_to_type")
    def validate_default_billed_to_type(self, _key: str, val: str) -> str:
        val = (val or "patient").strip().lower()
        if val not in ("patient", "clinic", "insurance"):
            msg = "default_billed_to_type invalide (patient|clinic|insurance)"
            raise ValueError(msg)
        if val == "patient":
            self.default_billed_to_company_id = None  # type: ignore[assignment]
        return val

    # ✅ D2: Propriétés hybrides pour chiffrement/déchiffrement automatique
    @hybrid_property
    def contact_phone_secure(self) -> str:  # type: ignore[no-redef]
        """Téléphone de contact déchiffré."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "contact_phone_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception as e:
                    logger.error("[D2] Erreur déchiffrement contact_phone: %s", e)
                    return None  # type: ignore[return]
            return cast(str, getattr(self, "contact_phone", None))
        except ImportError:
            return cast(str, getattr(self, "contact_phone", None))

    @contact_phone_secure.setter  # type: ignore[no-redef]
    def contact_phone_secure(self, value: str):
        """Chiffre le téléphone de contact."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.contact_phone_encrypted = get_encryption_service().encrypt_field(
                    value
                )
                self.encryption_migrated = True
                self.contact_phone = None  # Ancienne colonne dépréciée
            else:
                self.contact_phone_encrypted = None  # type: ignore[assignment]
                self.contact_phone = None
        except ImportError:
            self.contact_phone = value

    @hybrid_property
    def gp_name_secure(self) -> str:  # type: ignore[no-redef]
        """Nom du médecin traitant déchiffré."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "gp_name_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception:
                    return None  # type: ignore[return]
            return cast(str, getattr(self, "gp_name", None))
        except ImportError:
            return cast(str, getattr(self, "gp_name", None))

    @gp_name_secure.setter  # type: ignore[no-redef]
    def gp_name_secure(self, value: str):
        """Chiffre le nom du médecin traitant."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.gp_name_encrypted = get_encryption_service().encrypt_field(value)
                self.encryption_migrated = True
            else:
                self.gp_name_encrypted = None  # type: ignore[assignment]
        except ImportError:
            self.gp_name = value  # type: ignore[assignment]

    @hybrid_property
    def gp_phone_secure(self) -> str:  # type: ignore[no-redef]
        """Téléphone du médecin traitant déchiffré."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "gp_phone_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception:
                    return None  # type: ignore[return]
            return cast(str, getattr(self, "gp_phone", None))
        except ImportError:
            return cast(str, getattr(self, "gp_phone", None))

    @gp_phone_secure.setter  # type: ignore[no-redef]
    def gp_phone_secure(self, value: str):
        """Chiffre le téléphone du médecin traitant."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.gp_phone_encrypted = get_encryption_service().encrypt_field(value)
                self.encryption_migrated = True
                self.gp_phone = None  # type: ignore[assignment]  # Ancienne colonne dépréciée
            else:
                self.gp_phone_encrypted = None  # type: ignore[assignment]
                self.gp_phone = None  # type: ignore[assignment]
        except ImportError:
            self.gp_phone = value  # type: ignore[assignment]

    @hybrid_property
    def billing_address_secure(self) -> str:  # type: ignore[no-redef]
        """Adresse de facturation déchiffrée."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "billing_address_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception:
                    return None  # type: ignore[return]
            return cast(str, getattr(self, "billing_address", None))
        except ImportError:
            return cast(str, getattr(self, "billing_address", None))

    @billing_address_secure.setter  # type: ignore[no-redef]
    def billing_address_secure(self, value: str):
        """Chiffre l'adresse de facturation."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.billing_address_encrypted = get_encryption_service().encrypt_field(
                    value
                )
                self.encryption_migrated = True
            else:
                self.billing_address_encrypted = None  # type: ignore[assignment]
        except ImportError:
            self.billing_address = value

    def _serialize_linked_institution(self) -> dict[str, object] | None:
        """Retourne les données de l'institution officielle liée, ou None."""
        inst = getattr(self, "linked_institution", None)
        if not inst:
            return None
        return {
            "id": inst.id,
            "name": inst.name,
            "institution_type": inst.institution_type,
            "address": inst.address,
            "contact_email": inst.contact_email,
            "contact_phone": inst.contact_phone,
        }

    # Serialization
    @property
    def serialize(self):
        user = self.user
        first_name = getattr(user, "first_name", "") or ""
        last_name = getattr(user, "last_name", "") or ""
        username = getattr(user, "username", "") or ""
        phone_user = getattr(user, "phone", "") or ""
        full_name = (
            f"{first_name} {last_name}".strip() or username or "Nom non renseigné"
        )

        return {
            "id": self.id,
            "user": user.serialize if user else None,
            "first_name": first_name,
            "last_name": last_name,
            "full_name": full_name,
            "client_type": self.client_type.value,
            "company_id": self.company_id,
            "billing_address": self.billing_address,
            "billing_lat": float(self.billing_lat)
            if self.billing_lat is not None
            else None,
            "billing_lon": float(self.billing_lon)
            if self.billing_lon is not None
            else None,
            "contact_email": self.contact_email,
            "phone": self.contact_phone or phone_user,
            "domicile": {
                "address": self.domicile_address,
                "zip": self.domicile_zip,
                "city": self.domicile_city,
                "lat": (
                    float(domicile_lat)
                    if (domicile_lat := getattr(self, "domicile_lat", None)) is not None
                    else None
                ),
                "lon": (
                    float(domicile_lon)
                    if (domicile_lon := getattr(self, "domicile_lon", None)) is not None
                    else None
                ),
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
                "billed_to_company": (
                    self.default_billed_to_company.serialize
                    if self.default_billed_to_company
                    else None
                ),
                "billed_to_contact": self.default_billed_to_contact,
            },
            "is_institution": _as_bool(self.is_institution),
            "institution_name": self.institution_name,
            "linked_institution_id": self.linked_institution_id,
            "linked_institution": self._serialize_linked_institution(),
            "residence_facility": self.residence_facility,
            "preferential_rate": (
                float(preferential_rate)
                if (preferential_rate := getattr(self, "preferential_rate", None))
                is not None
                else None
            ),
            "is_active": _as_bool(self.is_active),
            "created_at": _iso(self.created_at),
        }

    def toggle_active(self) -> bool:
        current = _as_bool(getattr(self, "is_active", False))
        self.is_active = not current
        return bool(self.is_active)

    def is_self_service(self) -> bool:
        return self.management_mode == ManagementMode.SELF_SERVICE

    @override
    def __repr__(self) -> str:
        return (
            f"<Client id={self.id}, user_id={self.user_id}, "
            f"type={self.client_type}, active={self.is_active}>"
        )
