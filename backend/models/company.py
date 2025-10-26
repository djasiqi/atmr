# models/company.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _as_bool, _as_dt
from .enums import DispatchMode

REMAINDER_ONE = 1
VALUE_ZERO = 0
AJUSTEMENTS_THRESHOLD = 10
IBAN_MIN_LENGTH = 15
IBAN_MAX_LENGTH = 34
COMPANY_NAME_MAX_LENGTH = 100

"""Model Company - Gestion des entreprises de transport.
Extrait depuis models.py (lignes ~420-600).
"""


if TYPE_CHECKING:
    from .dispatch import DailyStats, DispatchMetrics, DispatchRun

logger = logging.getLogger(__name__)


class Company(db.Model):
    __tablename__ = "company"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Adresse opérationnelle
    address: Mapped[str] = mapped_column(String(200), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    longitude: Mapped[float] = mapped_column(Float, nullable=True)

    # Adresse de domiciliation
    domicile_address_line1: Mapped[str] = mapped_column(String(200), nullable=True)
    domicile_address_line2: Mapped[str] = mapped_column(String(200), nullable=True)
    domicile_zip: Mapped[str] = mapped_column(String(10), nullable=True)
    domicile_city: Mapped[str] = mapped_column(String(100), nullable=True)
    domicile_country = Column(String(2), nullable=True, server_default="CH")

    # Contact
    contact_email: Mapped[str] = mapped_column(String(100), nullable=True)
    contact_phone: Mapped[str] = mapped_column(String(20), nullable=True)

    # Légal & Facturation
    iban: Mapped[str] = mapped_column(String(34), nullable=True, index=True)
    uid_ide: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    billing_email: Mapped[str] = mapped_column(String(100), nullable=True)
    billing_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    user_id = Column(
        Integer,
        ForeignKey(
            "user.id",
            ondelete="CASCADE",
            name="fk_company_user"),
        nullable=False,
        index=True)
    is_approved = Column(Boolean, nullable=False, server_default="false")
    created_at = Column(
        DateTime(
            timezone=True),
        server_default=func.now(),
        nullable=False)
    service_area: Mapped[str] = mapped_column(String(200), nullable=True)
    max_daily_bookings: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, server_default="50")
    accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    dispatch_enabled = Column(Boolean, nullable=False, server_default="false")
    is_partner = Column(Boolean, nullable=False, server_default="false")
    logo_url: Mapped[str] = mapped_column(String(255), nullable=True)

    # 🆕 Configuration du système de dispatch autonome
    dispatch_mode = Column(
        Enum(DispatchMode),
        default=DispatchMode.SEMI_AUTO,
        nullable=False,
        server_default="semi_auto",
        index=True,
        comment="Mode de fonctionnement du dispatch: manual, semi_auto, fully_auto"
    )
    autonomous_config = Column(
        Text,
        nullable=True,
        comment="Configuration JSON pour le dispatch autonome"
    )

    # Relations
    user = relationship("User", back_populates="company", passive_deletes=True)
    clients = relationship(
        "Client",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Client.company_id",
        primaryjoin="Company.id == Client.company_id")
    billed_clients = relationship(
        "Client",
        back_populates="default_billed_to_company",
        foreign_keys="Client.default_billed_to_company_id",
        primaryjoin="Company.id == Client.default_billed_to_company_id")
    drivers = relationship(
        "Driver",
        back_populates="company",
        passive_deletes=True)
    dispatch_runs: Mapped[List[DispatchRun]] = relationship(
        "DispatchRun", back_populates="company", cascade="all, delete-orphan", passive_deletes=True)
    dispatch_metrics: Mapped[List[DispatchMetrics]] = relationship(
        "DispatchMetrics", back_populates="company", cascade="all, delete-orphan", passive_deletes=True)
    daily_stats: Mapped[List[DailyStats]] = relationship(
        "DailyStats", back_populates="company", cascade="all, delete-orphan", passive_deletes=True)
    bookings = relationship(
        "Booking",
        back_populates="company",
        foreign_keys="Booking.company_id",
        passive_deletes=True)
    vehicles = relationship(
        "Vehicle",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True)

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
            msg = f"Format d'email invalide pour {key}."
            raise ValueError(msg)
        return v

    @validates("contact_phone")
    def validate_contact_phone(self, _key, value):
        if not value:
            return value
        v = value.strip()
        if not re.match(r"^\+?[0-9\s\-\(\)]{7,20}$", v):
            msg = "Numéro de téléphone invalide."
            raise ValueError(msg)
        return v

    @validates("iban")
    def validate_iban(self, _key, value):
        if not value:
            return value
        v = value.replace(" ", "").upper()
        if len(v) < IBAN_MIN_LENGTH or len(v) > IBAN_MAX_LENGTH or not v[:2].isalpha() or not v[2:4].isdigit():
            msg = "IBAN invalide (format)."
            raise ValueError(msg)
        rearranged = v[4:] + v[:4]
        try:
            converted = "".join(str(int(ch, 36)) for ch in rearranged)
        except ValueError as err:
            msg = "IBAN invalide (caractères non autorisés)."
            raise ValueError(msg) from err
        remainder = 0
        for i in range(0, len(converted), 9):
            remainder = int(str(remainder) + converted[i:i + 9]) % 97
        if remainder != REMAINDER_ONE:
            msg = "IBAN invalide (checksum)."
            raise ValueError(msg)
        return v

    @validates("uid_ide")
    def validate_uid_ide(self, _key, value):
        if not value:
            return value
        v = value.strip().upper()
        if not re.match(
                r"^CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?$|^CHE[- ]?\d{9}(\s*TVA)?$", v, flags=re.IGNORECASE):
            msg = "IDE/UID suisse invalide (ex: CHE-123.456789)."
            raise ValueError(msg)
        digits = re.sub(r"\D", "", v)[:9]
        v_norm = f"CHE-{digits[0:3]}.{digits[3:6]}.{digits[6:9]}"
        if "TVA" in v:
            v_norm += " TVA"
        return v_norm

    @validates("name")
    def validate_name(self, _key, value):
        if not value or len(value.strip()) == 0:
            msg = "Le nom de l'entreprise ne peut pas être vide."
            raise ValueError(msg)
        if len(value) > COMPANY_NAME_MAX_LENGTH:
            msg = f"Le nom de l'entreprise ne peut pas dépasser {COMPANY_NAME_MAX_LENGTH} caractères."
            raise ValueError(msg)
        return value.strip()

    @validates("user_id")
    def validate_user_id(self, _key, value):
        if not isinstance(value, int) or value <= VALUE_ZERO:
            msg = "ID utilisateur invalide."
            raise ValueError(msg)
        return value

    def toggle_approval(self) -> bool:
        self.is_approved = not _as_bool(self.is_approved)
        return _as_bool(self.is_approved)

    def can_dispatch(self) -> bool:
        return _as_bool(self.is_approved) and _as_bool(self.dispatch_enabled)

    def approve(self):
        self.is_approved = True
        self.accepted_at = datetime.now(UTC)

    def get_autonomous_config(self) -> Dict[str, Any]:
        """Retourne la configuration autonome avec valeurs par défaut.

        Returns:
            Configuration complète pour le dispatch autonome

        """
        default_config: Dict[str, Any] = {
            "auto_dispatch": {
                "enabled": False,
                "interval_minutes": 5,
                "trigger_on_urgent_booking": True,
                "trigger_on_driver_unavailable": True,
            },
            "realtime_optimizer": {
                "enabled": False,
                "check_interval_minutes": 2,
                "auto_apply_suggestions": False,
            },
            "auto_apply_rules": {
                # Notifications auto (5-20 min retard)
                "customer_notifications": True,
                "minor_time_adjustments": False,   # Ajustements < AJUSTEMENTS_THRESHOLD min
                "reassignments": False,            # Toujours manuel par défaut
                "emergency_notifications": True,   # Alertes urgentes (>30 min)
            },
            "safety_limits": {
                "max_auto_actions_per_hour": 50,
                "max_auto_reassignments_per_day": 10,
                "require_approval_delay_minutes": 30,  # >30 min = validation manuelle
            },
            "re_optimize_triggers": {
                "delay_threshold_minutes": 15,
                "driver_became_unavailable": True,
                "better_driver_available_gain_minutes": 10,
            }
        }

        # Si une config est stockée, la merger avec les valeurs par défaut
        config_value = getattr(self, "autonomous_config", None)
        if config_value and isinstance(config_value, str) and config_value.strip():
            try:
                stored_config = json.loads(config_value)
                # Deep merge récursif

                def deep_merge(
                        base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
                    result = base.copy()
                    for key, value in override.items():
                        if key in result and isinstance(
                                result[key], dict) and isinstance(value, dict):
                            result[key] = deep_merge(result[key], value)
                        else:
                            result[key] = value
                    return result

                return deep_merge(default_config, stored_config)
            except (json.JSONDecodeError, TypeError, AttributeError) as err:
                # Si la config est invalide, retourner la config par défaut
                logger.warning(
                    "[Company] Invalid autonomous_config for company %s: %s",
                    self.id, err
                )
                return default_config

        return default_config

    def set_autonomous_config(self, config: Dict[str, Any]) -> None:
        """Définit la configuration autonome.

        Args:
            config: Configuration à stocker (sera mergée avec les valeurs par défaut)

        """
        self.autonomous_config = json.dumps(config)

    @override
    def __repr__(self):
        return f"<Company {self.name} | ID: {self.id} | Approved: {self.is_approved}>"

    def to_dict(self):
        return self.serialize.copy()
