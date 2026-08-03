# models/company.py
# pyright: reportRedeclaration=false
# Le linter détecte un conflit entre Column(name="iban") et @hybrid_property iban,
# mais c'est un faux positif : Column avec name ne crée pas d'attribut Python.

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db
from security.crypto import get_encryption_service

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
    # ✅ S2: IBAN chiffré en base de données (conformité RGPD)
    # Le champ _iban_raw stocke le texte chiffré (base64), peut être plus
    # long que l'IBAN original
    # Utilisation de Column avec name="iban" pour garder le nom de colonne en base
    _iban_raw = Column(
        String(200), nullable=True, name="iban"
    )  # Augmenté à 200 pour stocker le texte chiffré
    uid_ide: Mapped[str] = mapped_column(String(20), nullable=True, index=True)
    legal_form: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Forme juridique contractuelle (LegalForm)",
    )
    signatory_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    signatory_title: Mapped[str | None] = mapped_column(String(120), nullable=True)
    billing_email: Mapped[str] = mapped_column(String(100), nullable=True)
    billing_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Tarif préférentiel pour les cliniques (ex: 40.00 CHF / trajet)
    preferential_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2),
        nullable=True,
        comment="Tarif préférentiel en CHF pour les cliniques",
    )

    user_id = Column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE", name="fk_company_user"),
        nullable=False,
        index=True,
    )
    is_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    service_area: Mapped[str] = mapped_column(String(200), nullable=True)
    max_daily_bookings: Mapped[int | None] = mapped_column(
        Integer, nullable=True, server_default="50"
    )
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    dispatch_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    is_partner = Column(Boolean, nullable=False, server_default="false")
    logo_url: Mapped[str] = mapped_column(String(255), nullable=True)

    # 🆕 Configuration du système de dispatch autonome
    dispatch_mode: Mapped[DispatchMode] = mapped_column(
        Enum(DispatchMode),
        default=DispatchMode.SEMI_AUTO,
        nullable=False,
        server_default="semi_auto",
        index=True,
        comment="Mode de fonctionnement du dispatch: manual, semi_auto, fully_auto",
    )
    autonomous_config: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Configuration JSON pour le dispatch autonome"
    )

    # ✅ Security V2: Politique de securite entreprise (JSON)
    security_policy: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="JSON: require_2fa_roles, password_expiry_days, max_session_days, enforcement_mode",
    )

    # Plateforme : suspension gouvernance (tenant = Company en V1 — voir docs/platform/DECISIONS.md)
    platform_suspended: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="false",
        comment="Intention persistée : tenant suspendu au sens plateforme",
    )

    # Recouvrement facturation plateforme (jamais via platform_suspended)
    platform_billing_access_state: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default="active",
        comment="active|partial|full — mode restreint commercial",
    )
    platform_billing_state_source: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="automatic_dunning|admin_manual",
    )
    platform_billing_state_reason_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    platform_billing_state_since: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    platform_billing_state_config_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    platform_billing_state_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    dunning_paused_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    dunning_pause_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    dunning_paused_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )

    # Relations
    user = relationship(
        "User",
        back_populates="company",
        passive_deletes=True,
        foreign_keys=[user_id],
    )
    clients = relationship(
        "Client",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Client.company_id",
        primaryjoin="Company.id == Client.company_id",
    )
    billed_clients = relationship(
        "Client",
        back_populates="default_billed_to_company",
        foreign_keys="Client.default_billed_to_company_id",
        primaryjoin="Company.id == Client.default_billed_to_company_id",
    )
    drivers = relationship("Driver", back_populates="company", passive_deletes=True)
    device_tokens = relationship(
        "DeviceToken",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    dispatch_runs: Mapped[List[DispatchRun]] = relationship(
        "DispatchRun",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    dispatch_metrics: Mapped[List[DispatchMetrics]] = relationship(
        "DispatchMetrics",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    daily_stats: Mapped[List[DailyStats]] = relationship(
        "DailyStats",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    bookings = relationship(
        "Booking",
        back_populates="company",
        foreign_keys="Booking.company_id",
        passive_deletes=True,
    )
    vehicles = relationship(
        "Vehicle",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    billing_profile = relationship(
        "CompanyBillingProfile",
        back_populates="company",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    client_stays = relationship(
        "ClientStay",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    transport_vouchers = relationship(
        "TransportVoucher",
        back_populates="company",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

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
            # Adresse de domiciliation (champs directs pour compatibilité frontend)
            "domicile_address_line1": self.domicile_address_line1,
            "domicile_address_line2": self.domicile_address_line2,
            "domicile_zip": self.domicile_zip,
            "domicile_city": self.domicile_city,
            "domicile_country": self.domicile_country,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "iban": self.iban,
            "uid_ide": self.uid_ide,
            "legal_form": self.legal_form,
            "signatory_name": self.signatory_name,
            "signatory_title": self.signatory_title,
            "billing_email": self.billing_email,
            "billing_notes": self.billing_notes,
            "preferential_rate": (
                float(self.preferential_rate)
                if self.preferential_rate is not None
                else None
            ),
            "logo_url": self.logo_url,
            "is_approved": _as_bool(self.is_approved),
            "is_partner": _as_bool(self.is_partner),
            "user_id": self.user_id,
            "service_area": self.service_area,
            "max_daily_bookings": self.max_daily_bookings,
            "created_at": created_dt.isoformat()
            if isinstance(created_dt, datetime)
            else None,
            "dispatch_enabled": _as_bool(self.dispatch_enabled),
            "accepted_at": accepted_dt.isoformat()
            if isinstance(accepted_dt, datetime)
            else None,
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

    @hybrid_property
    def iban(self) -> str | None:
        """✅ S2: Propriété hybride pour déchiffrer automatiquement l'IBAN.

        Returns:
            IBAN en clair ou None si vide
        """
        if not bool(getattr(self, "_iban_raw", None)):
            return None
        try:
            encryption_service = get_encryption_service()
            return encryption_service.decrypt_field(
                str(getattr(self, "_iban_raw", None))
            )
        except Exception as e:
            logger.error(
                "[Company] Erreur déchiffrement IBAN pour company_id=%s: %s", self.id, e
            )
            # En cas d'erreur, retourner None plutôt que de lever une exception
            # pour éviter de casser l'application si des données sont corrompues
            return None

    @iban.setter
    def iban(self, value: str | None) -> None:
        """✅ S2: Setter pour chiffrer automatiquement l'IBAN avant stockage.

        Args:
            value: IBAN en clair ou None
        """
        if not value:
            self._iban_raw = None
            return

        # Valider l'IBAN avant chiffrement
        v = value.replace(" ", "").upper()
        if (
            len(v) < IBAN_MIN_LENGTH
            or len(v) > IBAN_MAX_LENGTH
            or not v[:2].isalpha()
            or not v[2:4].isdigit()
        ):
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
            remainder = int(str(remainder) + converted[i : i + 9]) % 97
        if remainder != REMAINDER_ONE:
            msg = "IBAN invalide (checksum)."
            raise ValueError(msg)

        # Chiffrer l'IBAN validé
        try:
            encryption_service = get_encryption_service()
            self._iban_raw = encryption_service.encrypt_field(v)
        except Exception as e:
            logger.error(
                "[Company] Erreur chiffrement IBAN pour company_id=%s: %s", self.id, e
            )
            raise

    @validates("uid_ide")
    def validate_uid_ide(self, _key, value):
        if not value:
            return value
        v = value.strip().upper()
        if not re.match(
            r"^CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?$|^CHE[- ]?\d{9}(\s*TVA)?$",
            v,
            flags=re.IGNORECASE,
        ):
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
            msg = (
                f"Le nom de l'entreprise ne peut pas dépasser "
                f"{COMPANY_NAME_MAX_LENGTH} caractères."
            )
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
        return bool(self.is_approved)

    def can_dispatch(self) -> bool:
        return bool(_as_bool(self.is_approved) and _as_bool(self.dispatch_enabled))

    def set_dispatch_mode(self, mode: DispatchMode) -> DispatchMode:
        """Change le mode de dispatch en garantissant l'invariant métier.

        Invariant : `dispatch_mode == MANUAL ⇒ dispatch_enabled == False`.
        Au passage en MANUAL, on coupe automatiquement `dispatch_enabled` afin
        qu'aucun déclencheur automatique ne puisse lancer un dispatch.

        Args:
            mode: Le nouveau mode de dispatch.

        Returns:
            Le mode effectivement appliqué.
        """
        self.dispatch_mode = mode
        if mode == DispatchMode.MANUAL:
            self.dispatch_enabled = False
        return mode

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
                # Ajustements < AJUSTEMENTS_THRESHOLD min
                "minor_time_adjustments": False,
                "reassignments": False,  # Toujours manuel par défaut
                "emergency_notifications": True,  # Alertes urgentes (>30 min)
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
            },
        }

        # Si une config est stockée, la merger avec les valeurs par défaut
        config_value = getattr(self, "autonomous_config", None)
        if config_value and isinstance(config_value, str) and config_value.strip():
            try:
                stored_config = json.loads(config_value)
                # Deep merge récursif

                def deep_merge(
                    base: Dict[str, Any], override: Dict[str, Any]
                ) -> Dict[str, Any]:
                    result = base.copy()
                    for key, value in override.items():
                        if (
                            key in result
                            and isinstance(result[key], dict)
                            and isinstance(value, dict)
                        ):
                            result[key] = deep_merge(result[key], value)
                        else:
                            result[key] = value
                    return result

                return deep_merge(default_config, stored_config)
            except (json.JSONDecodeError, TypeError, AttributeError) as err:
                # Si la config est invalide, retourner la config par défaut
                logger.warning(
                    "[Company] Invalid autonomous_config for company %s: %s",
                    self.id,
                    err,
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
