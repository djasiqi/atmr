# --- models.py (en-tête : imports standard) ---
from __future__ import annotations

import os
import uuid
import re
import base64
import binascii


from datetime import datetime, date, timezone
from enum import Enum as PyEnum
from typing import Any, Optional, cast, Type, TypeVar, Dict, List

from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import (
    Column, Integer, String, Float, UniqueConstraint, Enum as SAEnum, ForeignKey,
    DateTime, Date, Boolean, func, Text, Index, CheckConstraint, event,
    text
)

from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy_utils import StringEncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine

from ext import db
from sqlalchemy.dialects.postgresql import JSONB



# garde uniquement les utilitaires temps que tu utilises vraiment
from shared.time_utils import (
    to_geneva_local, iso_utc_z, to_utc_from_db, parse_local_naive, now_local, split_date_time_local
)

TEnum = TypeVar("TEnum", bound=PyEnum)

# --- Helpers de typage/normalisation (pour calmer Pylance) ---
def _as_dt(v: Any) -> Optional[datetime]:
    return v if isinstance(v, datetime) else None

def _as_str(v: Any) -> Optional[str]:
    return v if isinstance(v, str) else None

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def _as_bool(v: Any) -> bool:
    # ⚠️ n’essaie pas de faire "if Column[bool]" → force un bool Python
    return bool(v) if isinstance(v, (bool, int)) else False

def _iso(v: Any) -> Optional[str]:
    dt = _as_dt(v)
    return dt.isoformat() if dt else None

def _coerce_enum(v: Any, enum_cls: Type[TEnum]) -> Optional[TEnum]:
    """Transforme str → Enum (par value OU par name), sinon None."""
    if isinstance(v, enum_cls):
        return v
    if isinstance(v, str):
        try:
            return enum_cls(v)            # ex: "pending" -> BookingStatus.PENDING
        except Exception:
            try:
                return enum_cls[v]        # ex: "PENDING" -> BookingStatus.PENDING
            except Exception:
                return None
    return None



def _load_encryption_key() -> bytes:
    """
    Charge la clé d'encryption de façon canonique :
    - d'abord APP_ENCRYPTION_KEY_B64 (Base64 URL-safe, sans '=')
    - sinon (legacy) ENCRYPTION_KEY_HEX ou ENCRYPTION_KEY (hex strict)
    Valide la longueur AES: 16/24/32 octets.
    """
    # 1) Canonique: Base64 URL-safe
    b64 = (os.getenv("APP_ENCRYPTION_KEY_B64") or "").strip()
    if b64:
        padded = b64 + "=" * (-len(b64) % 4)
        try:
            key = base64.urlsafe_b64decode(padded.encode())
        except (binascii.Error, ValueError):
            raise RuntimeError("APP_ENCRYPTION_KEY_B64 doit être en Base64 URL-safe.")
        if len(key) not in (16, 24, 32):
            raise RuntimeError("APP_ENCRYPTION_KEY_B64 invalide: longueur attendue 16/24/32 octets (AES-128/192/256).")
        return key

    # 2) Legacy hex (uniquement si B64 absente)
    legacy_hex = (os.getenv("ENCRYPTION_KEY_HEX") or os.getenv("ENCRYPTION_KEY") or "").strip()
    if legacy_hex:
        if legacy_hex.lower().startswith("0x"):
            legacy_hex = legacy_hex[2:]
        try:
            key = bytes.fromhex(legacy_hex)
        except ValueError:
            raise RuntimeError("ENCRYPTION_KEY_HEX/ENCRYPTION_KEY est définie mais n'est pas une chaîne hex valide.")
        if len(key) not in (16, 24, 32):
            raise RuntimeError("Clé hex invalide: longueur attendue 16/24/32 octets.")
        return key

    # 3) Fail fast explicite
    raise RuntimeError("APP_ENCRYPTION_KEY_B64 manquante. Fournis une clé Base64 URL-safe (16/24/32 octets).")

# Expose la clé pour le reste de models.py
_encryption_key = _load_encryption_key()
_encryption_key_str = _encryption_key.hex()

# ========== ENUMS ==========

class UserRole(str, PyEnum):
    ADMIN   = "ADMIN"
    CLIENT  = "CLIENT"
    DRIVER  = "DRIVER"
    COMPANY = "COMPANY"
    # alias rétrocompat (garde UserRole.client, etc.)
    admin   = ADMIN
    client  = CLIENT
    driver  = DRIVER
    company = COMPANY

class BookingStatus(str, PyEnum):
    PENDING           = "PENDING"
    ACCEPTED          = "ACCEPTED"
    ASSIGNED          = "ASSIGNED"
    EN_ROUTE          = "EN_ROUTE"
    IN_PROGRESS       = "IN_PROGRESS"
    COMPLETED         = "COMPLETED"
    RETURN_COMPLETED  = "RETURN_COMPLETED"
    CANCELED          = "CANCELED"
    @classmethod
    def choices(cls): return [e.value for e in cls]

class PaymentStatus(str, PyEnum):
    PENDING   = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    @classmethod
    def choices(cls): return [e.value for e in cls]

class GenderEnum(str, PyEnum):
    HOMME = "HOMME"
    FEMME = "FEMME"
    AUTRE = "AUTRE"
    # alias rétrocompat
    homme = HOMME
    femme = FEMME
    autre = AUTRE

class ClientType(str, PyEnum):
    SELF_SERVICE = "SELF_SERVICE"
    PRIVATE      = "PRIVATE"
    CORPORATE    = "CORPORATE"

class InvoiceStatus(str, PyEnum):
    UNPAID   = "UNPAID"
    PAID     = "PAID"
    CANCELED = "CANCELED"
    @classmethod
    def choices(cls): return [e.value for e in cls]

class DriverType(PyEnum):
    REGULAR   = "REGULAR"
    EMERGENCY = "EMERGENCY"

class DriverState(str, PyEnum):
    AVAILABLE = "AVAILABLE"
    BUSY      = "BUSY"
    OFFLINE   = "OFFLINE"

class VacationType(str, PyEnum):
    VACANCES = "VACANCES"
    MALADIE  = "MALADIE"
    CONGES   = "CONGES"
    AUTRE    = "AUTRE"

class SenderRole(str, PyEnum):
    DRIVER  = "DRIVER"
    COMPANY = "COMPANY"
    # alias rétrocompat
    driver  = DRIVER
    company = COMPANY

class RealtimeEventType(str, PyEnum):
    LOCATION_UPDATE  = "LOCATION_UPDATE"
    STATUS_CHANGE    = "STATUS_CHANGE"
    ASSIGNMENT_DELTA = "ASSIGNMENT_DELTA"
    DELAY_DETECTED   = "DELAY_DETECTED"

class RealtimeEntityType(str, PyEnum):
    DRIVER     = "DRIVER"
    BOOKING    = "BOOKING"
    ASSIGNMENT = "ASSIGNMENT"

class AssignmentStatus(str, PyEnum):
    SCHEDULED        = "SCHEDULED"
    EN_ROUTE_PICKUP  = "EN_ROUTE_PICKUP"
    ARRIVED_PICKUP   = "ARRIVED_PICKUP"
    ONBOARD          = "ONBOARD"
    EN_ROUTE_DROPOFF = "EN_ROUTE_DROPOFF"
    ARRIVED_DROPOFF  = "ARRIVED_DROPOFF"
    COMPLETED        = "COMPLETED"
    CANCELLED        = "CANCELLED"
    NO_SHOW          = "NO_SHOW"
    REASSIGNED       = "REASSIGNED"

class DispatchStatus(str, PyEnum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


     

# ========== MODÈLES ==========

class User(db.Model):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    public_id = Column(String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True, unique=True, index=True)
    
    # ↓ Champs présents pour tous les rôles (client, driver, etc.)
    phone = Column(String(20), nullable=True)
    address = Column(String(200), nullable=True)
    birth_date = Column(Date, nullable=True)
    gender = Column(SAEnum(GenderEnum, name="gender"), nullable=True)
    profile_image = Column(String(255), nullable=True)
    
    password = Column(String(255), nullable=False)
    role = Column(
        SAEnum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.CLIENT,
        server_default=UserRole.CLIENT.value,
    )

    reset_token = Column(String(100), unique=True, nullable=True)
    zip_code = Column(String(10), nullable=True)
    city = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    force_password_change = Column(Boolean, default=False, nullable=False)

    # ✅ Ajout de l'index sur `public_id` pour optimiser les recherches
    __table_args__ = (
        Index('idx_public_id', 'public_id'),
    )

    # ✅ Relations bidirectionnelles avec suppression en cascade
    clients = relationship('Client', back_populates='user', cascade="all, delete-orphan")
    driver = relationship('Driver', back_populates='user', uselist=False, cascade="all, delete-orphan", passive_deletes=True)
    company = relationship('Company', back_populates='user', uselist=False, cascade="all, delete-orphan", passive_deletes=True)
    invoices = relationship("Invoice", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

    # 🔒 Gestion des mots de passe
    def set_password(self, password, force_change=False):
        self.password = generate_password_hash(password)
        self.force_password_change = force_change


    def check_password(self, password: str) -> bool:
        # Récupère la valeur runtime (qui sera bien une string en pratique)
        pw_any = getattr(self, "password", "")  # évite les warnings d'attr
        if isinstance(pw_any, (bytes, bytearray)):
            pw_str = pw_any.decode("utf-8", "ignore")
        else:
            pw_str = cast(str, pw_any or "")
        return check_password_hash(pw_str, password)


    # Validation du téléphone
    @validates('phone')
    def validate_phone(self, key, phone):
        if phone is None or phone.strip() == "":
            return None
        phone = phone.strip()
        if not re.match(r"^\+?\d{7,15}$", phone):
            raise ValueError("Numéro de téléphone invalide. Doit contenir 7 à 15 chiffres avec option '+'.")
        return phone



    # Validation de la date de naissance
    @validates('birth_date')
    def validate_birth_date(self, key, birth_date):
        """Vérifie que la date de naissance est valide et raisonnable."""
        if birth_date and birth_date > date.today():
            raise ValueError("La date de naissance ne peut pas être dans le futur.")
        return birth_date
    
    # Validation de l'adresse
    @validates('address')
    def validate_address(self, key, address):
        if address is not None:  # Vérifie si la valeur n'est pas None
            if address.strip() == '':
                raise ValueError("L'adresse ne peut pas être vide.")
            if len(address) > 200:
                raise ValueError("L'adresse ne peut pas dépasser 200 caractères.")
        return address
    
    @validates('first_name', 'last_name')
    def validate_name(self, key, name):
        if name is not None:  # ✅ Vérifie si la valeur n'est pas None
            if len(name.strip()) == 0:
                raise ValueError(f"Le champ {key} ne peut pas être vide.")
        return name
    
     # 📌 Vérification du genre
    @validates('gender')
    def validate_gender(self, key, gender_value):
        """
        Valide/convertit la valeur vers GenderEnum.
        Évite d'utiliser le nom 'gender' (collision avec l'attribut mappé).
        """
        if gender_value is None:
            return None
        coerced = _coerce_enum(gender_value, GenderEnum)
        if coerced is None:
            raise ValueError("Genre invalide.")
        return coerced
    
    @validates('role')
    def validate_role(self, key, role_value):
        """Coerce str → UserRole, évite d’évaluer un Column en bool."""
        coerced = _coerce_enum(role_value, UserRole)
        if coerced is None:
            raise ValueError("Invalid role value. Allowed values: admin, client, driver, company.")
        return coerced
    
    @validates('email')
    def validate_email(self, key, email):
        """
        Valide le format si fourni.
        ⚠️ La règle 'self-service => email requis' est déjà appliquée dans Client.validate_contact_email.
        On évite ici toute logique cross-model (et donc les tests sur self.clients / self.role).
        """
        if email is None or email.strip() == "":
            return None
        import re
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email.strip()):
            raise ValueError("Format d'email invalide.")
        return email.strip()
    
    # Propriété pour la sérialisation
    @property
    def serialize(self):
        return {
            "id": self.id,
            "user_id": self.id,  # ✅ correction ici
            "public_id": self.public_id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name or "Non spécifié",
            "last_name": self.last_name or "Non spécifié",
            "phone": self.phone or "Non spécifié",
            "address": self.address or "Non spécifié",
            "birth_date": (self.birth_date.strftime('%Y-%m-%d') if isinstance(self.birth_date, date) else None),
            "gender": (self.gender.value if isinstance(self.gender, GenderEnum) else "Non spécifié"),
            "profile_image": self.profile_image or None,
            "role": (self.role.value if isinstance(self.role, UserRole) else str(self.role)),
            "zip_code": self.zip_code or "Non spécifié",
            "city": self.city or "Non spécifié",
            "created_at": _iso(self.created_at),
            "force_password_change": self.force_password_change
        }

    @property
    def full_name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip()

    # 📌 Représentation pour le debug
    def __repr__(self):
        return f"<User {self.username} ({self.email}) - Role: {self.role.value}>"
    
    def to_dict(self):
        return self.serialize

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
    domicile_country = Column(String(2), nullable=True, server_default="CH")  # ISO 3166-1

    # Contact
    contact_email = Column(String(100), nullable=True)
    contact_phone = Column(String(20), nullable=True)

    # Légal & Facturation
    iban = Column(String(34), nullable=True, index=True)       # IBAN (max 34 chars)
    uid_ide = Column(String(20), nullable=True, index=True)    # ex: CHE-123.456.789
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
    billed_clients = relationship("Client", back_populates="default_billed_to_company", foreign_keys="Client.default_billed_to_company_id", primaryjoin="Company.id == Client.default_billed_to_company_id",)
    drivers = relationship("Driver", back_populates="company", passive_deletes=True)
    dispatch_runs: Mapped[List["DispatchRun"]] = relationship( "DispatchRun", back_populates="company", cascade="all, delete-orphan", passive_deletes=True,)

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

    # --------- Sérialisation ---------
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
            "is_approved": _as_bool(self.is_approved),
            "is_partner": _as_bool(self.is_partner),
            "user_id": self.user_id,
            "service_area": self.service_area,
            "max_daily_bookings": self.max_daily_bookings,
            "created_at": created_dt.isoformat() if created_dt else None,
            "dispatch_enabled": _as_bool(self.dispatch_enabled),
            "accepted_at": accepted_dt.isoformat() if accepted_dt else None,
            "logo_url": self.logo_url,
            "vehicles": [v.serialize for v in self.vehicles],
        }

    # --------- Validateurs ---------
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

    # --------- Méthodes métier ---------
    def toggle_approval(self) -> bool:
        self.is_approved = not _as_bool(self.is_approved)
        return _as_bool(self.is_approved)

    def can_dispatch(self) -> bool:
        return _as_bool(self.is_approved) and _as_bool(self.dispatch_enabled)

    def approve(self):
        self.is_approved = True
        self.accepted_at = datetime.now(timezone.utc)

    def __repr__(self):
        return f"<Company {self.name} | ID: {self.id} | Approved: {self.is_approved}>"

    def to_dict(self):
        return self.serialize.copy()


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
    model = Column(String(120), nullable=False)          # ex: "VW Caddy Maxi"
    license_plate = Column(String(20), nullable=False, index=True)  # ex: "GE 123456"
    year = Column(Integer, nullable=True)                # ex: 2021
    vin = Column(String(32), nullable=True)              # optionnel

    # Capacités
    seats = Column(Integer, nullable=True)               # nb de places assises
    wheelchair_accessible = Column(Boolean, nullable=False, server_default="false")

    # Suivi administratif
    insurance_expires_at = Column(DateTime(timezone=True), nullable=True)
    inspection_expires_at = Column(DateTime(timezone=True), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relations
    company = relationship("Company", back_populates="vehicles", passive_deletes=True)

    # -------- Validations --------
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

    # -------- Sérialisation --------
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


class Driver(db.Model):
    __tablename__ = "driver"
    __table_args__ = (
        # Accélère les listes par entreprise + filtres de disponibilité
        Index("ix_driver_company_active", "company_id", "is_active", "is_available"),
        # Optionnel mais utile si tu fais des requêtes par position
        Index("ix_driver_geo", "company_id", "latitude", "longitude"),
        # Garde-fous géographiques (NULL autorisé)
        CheckConstraint("(latitude IS NULL OR (latitude BETWEEN -90 AND 90))", name="chk_driver_lat"),
        CheckConstraint("(longitude IS NULL OR (longitude BETWEEN -180 AND 180))", name="chk_driver_lon"),
    )

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True)

    # Véhicule (métadonnées libres)
    vehicle_assigned = Column(String(100), nullable=True)
    brand = Column(String(100), nullable=True)

    # Plaque chiffrée (sqlalchemy-utils / AES)
    license_plate = Column(
        StringEncryptedType(
            String,               # base type
            _encryption_key_str,  # clé hex
            AesEngine,
            "pkcs5",
        ),
        nullable=True,
    )

    # États
    is_active = Column(Boolean, nullable=False, server_default="true")
    is_available = Column(Boolean, nullable=False, server_default="true")

    driver_type = Column(SAEnum(DriverType, name="driver_type"), 
                     nullable=False, 
                     server_default="REGULAR")

    # Localisation temps réel
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    last_position_update = Column(DateTime(timezone=True), nullable=True)

    # Média & notifications
    driver_photo = Column(Text, nullable=True)
    push_token = Column(String(255), nullable=True, index=True)

    # Relations
    user = relationship("User", back_populates="driver", passive_deletes=True)
    company = relationship("Company", back_populates="drivers", passive_deletes=True)
    bookings = relationship("Booking", back_populates="driver", passive_deletes=True)
    working_config = relationship("DriverWorkingConfig", backref="driver", uselist=False)
    vacations = relationship("DriverVacation", back_populates="driver", cascade="all, delete-orphan", passive_deletes=True,)


    # ------------ Validations ------------
    @validates("user_id")
    def _v_user_id(self, _key: str, user_id: int) -> int:
        if user_id is None or int(user_id) <= 0:
            raise ValueError("L'ID utilisateur est invalide.")

        user = db.session.get(User, int(user_id))
        if not user:
            raise ValueError("Utilisateur non trouvé.")

        # ⚠️ Ne JAMAIS faire "if user.role == ..." directement.
        role_value = _coerce_enum(getattr(user, "role", None), UserRole)
        if role_value is None or role_value is not UserRole.DRIVER:
            raise ValueError("L'utilisateur associé n'a pas le rôle 'driver'.")

        return int(user_id)

    @validates("company_id")
    def _v_company_id(self, _key: str, company_id: int) -> int:
        if company_id is None or int(company_id) <= 0:
            raise ValueError("L'ID de l'entreprise est invalide.")
        company = db.session.get(Company, int(company_id))
        if not company:
            raise ValueError("Entreprise non trouvée.")
        return int(company_id)

    @validates("vehicle_assigned", "brand", "license_plate")
    def _v_vehicle_info(self, key: str, value: str | None) -> str | None:
        if value is not None and not str(value).strip():
            raise ValueError(f"Le champ {key} ne peut pas être vide.")
        return value

    @validates("latitude", "longitude")
    def _v_gps(self, key: str, value: float | None) -> float | None:
        if value is None:
            return None
        v = float(value)
        if key == "latitude" and not (-90 <= v <= 90):
            raise ValueError("Latitude hors limites [-90; 90].")
        if key == "longitude" and not (-180 <= v <= 180):
            raise ValueError("Longitude hors limites [-180; 180].")
        return v

    # ------------ Utilitaires ------------
    @property
    def serialize(self) -> dict:
        dt = getattr(self, "driver_type", None)
        return {
            "id": self.id,
            "user_id": self.user_id,
            # évite d'évaluer self.user en bool; utilise getattr
            "username": getattr(self.user, "username", "Non spécifié"),
            "first_name": getattr(self.user, "first_name", "Non spécifié"),
            "last_name": getattr(self.user, "last_name", "Non spécifié"),
            "phone": getattr(self.user, "phone", "Non spécifié"),
            "photo": self.driver_photo or getattr(self.user, "profile_image", "/images/default-driver.png"),
            "company_id": self.company_id,
            "company_name": getattr(self.company, "name", "Non spécifié"),
            "is_active": _as_bool(self.is_active),
            "is_available": _as_bool(self.is_available),
            # ⬇️ pas de truthiness sur un champ mappé
            "driver_type": (dt.value if isinstance(dt, DriverType) else (str(dt) if dt is not None else None)),
            "vehicle_assigned": self.vehicle_assigned or "Non spécifié",
            "brand": self.brand or "Non spécifié",
            "license_plate": self.license_plate or "Non spécifié",
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

    @property
    def serialize_position(self):
        return {
            "id": self.id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "is_available": _as_bool(self.is_available),
        }

    def toggle_availability(self) -> bool:
        current = _as_bool(getattr(self, "is_available", False))
        self.is_available = not current
        return not current

    def update_location(self, latitude: float, longitude: float):
        if latitude is None or longitude is None:
            raise ValueError("Les coordonnées GPS ne peuvent pas être nulles.")
        self.latitude = self._v_gps("latitude", latitude)
        self.longitude = self._v_gps("longitude", longitude)
        self.last_position_update = datetime.now(timezone.utc)

    def deactivate(self):
        self.is_active = False
        self.is_available = False
        db.session.commit()

    def activate(self):
        self.is_active = True
        self.is_available = True
        db.session.commit()

    def __repr__(self):
        uname = self.user.username if self.user else "N/A"
        cname = self.company.name if self.company else "N/A"
        return f"<Driver id={self.id} user={uname} company={cname}>"

    def to_dict(self):
        return self.serialize

class DriverWorkingConfig(db.Model):
    """
    Config horaire d'un chauffeur (minutes depuis minuit).
    """
    __tablename__ = "driver_working_config"
    __table_args__ = (
        # bornes 0..1440
        CheckConstraint("earliest_start BETWEEN 0 AND 1440", name="chk_dwc_earliest"),
        CheckConstraint("latest_start BETWEEN 0 AND 1440", name="chk_dwc_latest"),
        CheckConstraint("total_working_minutes BETWEEN 0 AND 1440", name="chk_dwc_total"),
        CheckConstraint("break_duration BETWEEN 0 AND 1440", name="chk_dwc_break_dur"),
        CheckConstraint("break_earliest BETWEEN 0 AND 1440", name="chk_dwc_break_earliest"),
        CheckConstraint("break_latest BETWEEN 0 AND 1440", name="chk_dwc_break_latest"),
        # logique de fenêtre
        CheckConstraint("earliest_start < latest_start", name="chk_dwc_start_window"),
        CheckConstraint("break_earliest < break_latest", name="chk_dwc_break_window"),
        # cohérence des durées
        CheckConstraint("break_duration <= total_working_minutes", name="chk_dwc_break_vs_total"),
        # un seul config par driver
        UniqueConstraint("driver_id", name="uq_dwc_driver"),
        Index("ix_dwc_driver", "driver_id"),
    )

    id = Column(Integer, primary_key=True)

    driver_id = Column(
        Integer,
        ForeignKey("driver.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # minutes depuis 00:00 (non-null avec valeurs par défaut côté DB)
    earliest_start = Column(Integer, nullable=False, server_default="360")   # 06:00
    latest_start = Column(Integer, nullable=False, server_default="600")     # 10:00
    total_working_minutes = Column(Integer, nullable=False, server_default="510")  # 8h30
    break_duration = Column(Integer, nullable=False, server_default="60")    # 1h
    break_earliest = Column(Integer, nullable=False, server_default="600")   # 10:00
    break_latest = Column(Integer, nullable=False, server_default="900")     # 15:00

    @property
    def serialize(self):
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "earliest_start": self.earliest_start,
            "latest_start": self.latest_start,
            "total_working_minutes": self.total_working_minutes,
            "break_duration": self.break_duration,
            "break_earliest": self.break_earliest,
            "break_latest": self.break_latest,
        }

    def __repr__(self):
        return (
            f"<DriverWorkingConfig id={self.id} driver_id={self.driver_id} "
            f"start={self.earliest_start}-{self.latest_start} "
            f"maxWork={self.total_working_minutes} "
            f"break={self.break_duration} ({self.break_earliest}-{self.break_latest})>"
        )

    # ---- Validations ORM (complémentaires aux CheckConstraints) ----
    @validates(
        "earliest_start",
        "latest_start",
        "total_working_minutes",
        "break_duration",
        "break_earliest",
        "break_latest",
    )
    def validate_working_times(self, key, value):
        if value is None:
            raise ValueError(f"'{key}' ne peut pas être NULL")
        v = int(value)
        if v < 0 or v > 1440:
            raise ValueError(f"'{key}' doit être entre 0 et 1440")
        return v

    def validate_config(self):
        earliest = _as_int(self.earliest_start)
        latest = _as_int(self.latest_start)
        if earliest >= latest:
            raise ValueError("earliest_start doit être avant latest_start")

        be = _as_int(self.break_earliest)
        bl = _as_int(self.break_latest)
        if be >= bl:
            raise ValueError("break_earliest doit être avant break_latest")

        if _as_int(self.break_duration) > _as_int(self.total_working_minutes):
            raise ValueError("La pause ne peut pas excéder le temps de travail total")

    # Hook ORM pour garantir la cohérence même sans passer par l’API
    @staticmethod
    def _enforce_config(_mapper, _connection, target: "DriverWorkingConfig") -> None:
        target.validate_config()

event.listen(DriverWorkingConfig, "before_insert", DriverWorkingConfig._enforce_config)
event.listen(DriverWorkingConfig, "before_update", DriverWorkingConfig._enforce_config)


class DriverVacation(db.Model):
    """
    Période d'absence / vacances d'un chauffeur.
    """
    __tablename__ = "driver_vacations"
    __table_args__ = (
        # ordre logique des dates
        CheckConstraint("start_date <= end_date", name="chk_vacation_dates_order"),
        # accélère les requêtes: chevauchement, filtrage par driver/date
        Index("ix_vacation_driver_start", "driver_id", "start_date"),
        Index("ix_vacation_driver_end", "driver_id", "end_date"),
        # évite les doublons exacts (même driver, même fenêtre, même type)
        UniqueConstraint("driver_id", "start_date", "end_date", "vacation_type",
                         name="uq_vacation_exact_period"),
    )

    id = Column(Integer, primary_key=True)
    driver_id = Column(Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, index=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    vacation_type = Column(SAEnum(VacationType, name="vacation_type"),
                           nullable=False, server_default=VacationType.VACANCES.value)

    # Relations (préférence: back_populates pour la symétrie)
    driver = relationship("Driver", back_populates="vacations", passive_deletes=True)

    def __repr__(self) -> str:
            vt = getattr(self.vacation_type, "value", self.vacation_type)
            # sd / ed peuvent être None au niveau typage statique → protège l'isoformat()
            sd = getattr(self, "start_date", None)
            ed = getattr(self, "end_date", None)
            sd_str = sd.isoformat() if isinstance(sd, date) else "?"
            ed_str = ed.isoformat() if isinstance(ed, date) else "?"
            return f"<DriverVacation id={self.id}, driver_id={self.driver_id}, {sd_str} → {ed_str}, type={vt}>"

    @validates("start_date", "end_date")
    def validate_dates(self, key: str, value: date) -> date:
        if value is None or not isinstance(value, date):
            raise ValueError(f"{key} doit être une date valide.")
        return value

    def validate_logic(self) -> None:
            # ⚠️ Ne pas faire "if self.start_date and self.end_date ..."
            sd = getattr(self, "start_date", None)
            ed = getattr(self, "end_date", None)
            if isinstance(sd, date) and isinstance(ed, date) and sd > ed:
                raise ValueError("La date de début ne peut pas être après la date de fin.")

    @staticmethod
    def _enforce_logic(_mapper, _connection, target: "DriverVacation") -> None:
        target.validate_logic()

    def overlaps(self, other_start: date, other_end: date) -> bool:
        """Retourne True si les fenêtres se chevauchent (bornes incluses)."""
        sd = getattr(self, "start_date", None)
        ed = getattr(self, "end_date", None)
        if not (isinstance(sd, date) and isinstance(ed, date)):
            # Si l'instance n'a pas de vraies dates Python (cas extrêmes), on ne peut pas décider → False
            return False
        # Comparaisons entre VRAIES dates Python → bool natif (pas de ColumnElement)
        return (sd <= other_end) and (ed >= other_start)

    @property
    def serialize(self) -> dict:
        sd = getattr(self, "start_date", None)
        ed = getattr(self, "end_date", None)
        vt = getattr(self, "vacation_type", None)
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "start_date": (sd.isoformat() if isinstance(sd, date) else None),
            "end_date": (ed.isoformat() if isinstance(ed, date) else None),
            "vacation_type": getattr(vt, "value", vt),
        }

event.listen(DriverVacation, "before_insert", DriverVacation._enforce_logic)
event.listen(DriverVacation, "before_update", DriverVacation._enforce_logic)


class Client(db.Model):
    __tablename__ = "client"
    __table_args__ = (
        UniqueConstraint("user_id", "company_id", name="uq_user_company"),
        # Accélère les listes par propriétaire et par activité
        Index("ix_client_company_active", "company_id", "is_active"),
        Index('ix_client_company_user', 'company_id', 'user_id'),
        Index('uq_client_user_no_company', 'user_id', unique=True, postgresql_where=text('company_id IS NULL')),
    )

    id = Column(Integer, primary_key=True)

    # --- rattachements
    user_id = Column(ForeignKey('user.id', ondelete="CASCADE"), nullable=False, index=True)

    # propriétaire (entreprise) -> visibilité/ACL
    company_id = Column(ForeignKey('company.id', ondelete="SET NULL"), nullable=True, index=True)

    client_type = Column(
        SAEnum(ClientType, name="client_type"),
        nullable=False,
        default=ClientType.SELF_SERVICE,
        server_default=ClientType.SELF_SERVICE.value,
    )


    # Coordonnées de facturation/contacts "côté entreprise"
    billing_address = Column(String(255), nullable=True)
    contact_email = Column(String(100), nullable=True)
    contact_phone = Column(String(50), nullable=True)

    # --- NOUVEAU: Domiciliation (propre à l'entreprise)
    domicile_address = Column(String(255), nullable=True)
    domicile_zip = Column(String(10), nullable=True)
    domicile_city = Column(String(100), nullable=True)

    # --- NOUVEAU: Accès logement
    door_code = Column(String(50), nullable=True)     # ex: B1234#
    floor = Column(String(20), nullable=True)         # ex: "RDC", "3", "2B"
    access_notes = Column(Text, nullable=True)        # consignes d'accès détaillées

    # --- NOUVEAU: Médecin traitant
    gp_name = Column(String(120), nullable=True)
    gp_phone = Column(String(50), nullable=True)

    # --- NOUVEAU: Préférences de facturation par défaut (préremplissage Booking)
    default_billed_to_type = Column(String(50), nullable=False, server_default="patient")  # "patient"|"clinic"|"insurance"
    default_billed_to_company_id = Column(Integer, ForeignKey("company.id", ondelete="SET NULL"), nullable=True)
    default_billed_to_contact = Column(String(120), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # 🔁 Relations (désambiguïsées)
    user = relationship("User", back_populates="clients", passive_deletes=True)
    company = relationship("Company", back_populates="clients", foreign_keys=[company_id], passive_deletes=True)
    default_billed_to_company = relationship("Company", back_populates="billed_clients", foreign_keys=[default_billed_to_company_id])


    bookings = relationship("Booking", back_populates="client", passive_deletes=True, lazy=True)
    payments = relationship("Payment", back_populates="client", passive_deletes=True, lazy=True)

    # ---------------- Validators ---------------- #

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
            raise ValueError("L'adresse de facturation est obligatoire pour les clients liés à une entreprise.")
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

    # ---------------- Serialization ---------------- #

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
            "contact_email": self.contact_email,
            "phone": self.contact_phone or phone_user,
            "domicile": {
                "address": self.domicile_address,
                "zip": self.domicile_zip,
                "city": self.domicile_city,
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
            "is_active": _as_bool(self.is_active),
            "created_at": _iso(self.created_at),
        }

    # ---------------- Utils ---------------- #

    def toggle_active(self) -> bool:
        current = _as_bool(getattr(self, "is_active", False))
        self.is_active = not current
        return bool(self.is_active)

    def is_self_service(self) -> bool:
        return cast(ClientType, self.client_type) == ClientType.SELF_SERVICE

    def __repr__(self):
        return f"<Client id={self.id}, user_id={self.user_id}, type={self.client_type}, active={self.is_active}>"


class Booking(db.Model):
    __tablename__ = 'booking'
    __table_args__ = (
        # bornes GPS côté DB (en plus des validations Python)
        CheckConstraint("pickup_lat IS NULL OR (pickup_lat BETWEEN -90 AND 90)",  name="chk_booking_pickup_lat"),
        CheckConstraint("pickup_lon IS NULL OR (pickup_lon BETWEEN -180 AND 180)", name="chk_booking_pickup_lon"),
        CheckConstraint("dropoff_lat IS NULL OR (dropoff_lat BETWEEN -90 AND 90)",  name="chk_booking_drop_lat"),
        CheckConstraint("dropoff_lon IS NULL OR (dropoff_lon BETWEEN -180 AND 180)", name="chk_booking_drop_lon"),
        # optionnel mais utile pour les requêtes fréquentes
        Index('ix_booking_company_scheduled', 'company_id', 'scheduled_time'),
        Index('ix_booking_status_scheduled', 'status', 'scheduled_time'),
        Index('ix_booking_driver_status', 'driver_id', 'status'),
    )

    id = Column(Integer, primary_key=True)
    customer_name = Column(String(100), nullable=False)
    pickup_location = Column(String(200), nullable=False)
    dropoff_location = Column(String(200), nullable=False)
    booking_type = Column(String(200), nullable=False, server_default='standard')  # "standard" | "manual"

    # Stockage local naïf (Europe/Zurich) — assumé par tes utilitaires.
    scheduled_time = Column(DateTime(timezone=False), nullable=True)

    amount = Column(Float, nullable=False)
    status = Column(
        SAEnum(BookingStatus, name="booking_status"),
        index=True, nullable=False,
        default=BookingStatus.PENDING,
        server_default=BookingStatus.PENDING.value,
    )


    user_id = Column(Integer, ForeignKey('user.id', ondelete="CASCADE"), nullable=False)

    # JSONB: default Python (callable) + default DB pour les insert “brut”
    rejected_by = Column(JSONB, nullable=False, default=list,
                         server_default=text("'[]'::jsonb"))

    duration_seconds = Column(Integer)
    distance_meters = Column(Integer)

    client_id = Column(Integer, ForeignKey('client.id', ondelete="CASCADE"), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=True, index=True)
    driver_id = Column(Integer, ForeignKey('driver.id', ondelete="SET NULL"), nullable=True, index=True)

    is_round_trip = Column(Boolean, nullable=False, server_default=text('false'))
    is_return     = Column(Boolean, nullable=False, server_default=text('false'))

    boarded_at   = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    parent_booking_id = Column(Integer, ForeignKey('booking.id', ondelete="SET NULL"), nullable=True)

    medical_facility = Column(String(200))
    doctor_name      = Column(String(200))
    hospital_service = Column(String(100))
    notes_medical    = Column(Text)
    is_urgent        = Column(Boolean, nullable=False, server_default=text('false'))

    pickup_lat  = Column(Float)
    pickup_lon  = Column(Float)
    dropoff_lat = Column(Float)
    dropoff_lon = Column(Float)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Qui paie cette course ?
    billed_to_type       = Column(String(50), nullable=False, server_default="patient")  # "patient" | "clinic" | "insurance"
    billed_to_company_id = Column(Integer, ForeignKey('company.id', ondelete="SET NULL"), nullable=True)
    billed_to_contact    = Column(String(120))

    # --- Relations
    client  = relationship('Client',  back_populates='bookings', passive_deletes=True)
    company = relationship('Company', back_populates='bookings', foreign_keys=[company_id], passive_deletes=True)
    driver  = relationship('Driver',  back_populates='bookings', passive_deletes=True)
    invoice = relationship("Invoice", back_populates="booking", cascade="all, delete-orphan", passive_deletes=True, uselist=False)
    payments = relationship('Payment', back_populates='booking', passive_deletes=True, lazy=True)

    billed_to_company = relationship('Company', foreign_keys=[billed_to_company_id])

    # Aller-retour auto-référent
    return_trip = relationship(
        "Booking",
        backref="original_booking",
        remote_side=[id],
        foreign_keys=[parent_booking_id],
        uselist=False
    )

    # --- Propriétés & méthodes métier (inchangées sauf uniformisation mineure) ---
    @property
    def customer_full_name(self) -> str:
        cust = _as_str(self.customer_name)
        if cust:
            return cust
        if self.client and self.client.user:
            u = self.client.user
            if (u.first_name or u.last_name):
                return f"{u.first_name or ''} {u.last_name or ''}".strip()
            return u.username
        return "Non spécifié"

    def get_effective_payer(self) -> dict:
        btype = (_as_str(self.billed_to_type) or "patient").lower()
        if btype != "patient" and getattr(self, "billed_to_company", None):
            comp = self.billed_to_company
            return {
                "type": btype,
                "name": comp.name,
                "address": getattr(comp, "address", None),
                "email": getattr(comp, "contact_email", None),
                "phone": getattr(comp, "contact_phone", None),
                "company_id": comp.id
            }

        cli = getattr(self, "client", None)
        if cli and getattr(cli, "user", None):
            u = cli.user
            full = f"{(u.first_name or '').strip()} {(u.last_name or '').strip()}".strip()
            return {
                "type": "patient",
                "name": full or (u.username or "Client"),
                "address": getattr(cli, "billing_address", None),
                "email": getattr(cli, "contact_email", None) or getattr(u, "email", None),
                "phone": getattr(cli, "contact_phone", None) or getattr(u, "phone", None),
                "client_id": cli.id
            }
        return {"type": "patient", "name": self.customer_full_name}

    @property
    def serialize(self):
        scheduled_dt = _as_dt(self.scheduled_time)
        created_dt   = _as_dt(self.created_at)
        updated_dt   = _as_dt(self.updated_at)
        boarded_dt   = _as_dt(self.boarded_at)
        completed_dt = _as_dt(self.completed_at)

        date_local, time_local = split_date_time_local(scheduled_dt) if scheduled_dt else (None, None)
        created_loc = to_geneva_local(created_dt) if created_dt else None
        updated_loc = to_geneva_local(updated_dt) if updated_dt else None

        amt = _as_float(self.amount, 0.0)
        status_val = cast(BookingStatus, self.status)

        cli = self.client
        cli_user = getattr(cli, "user", None)

        return {
            "id": self.id,
            "customer_name": self.customer_full_name,
            "client_name": self.customer_full_name,
            "pickup_location": self.pickup_location,
            "dropoff_location": self.dropoff_location,
            "amount": round(amt, 2),
            "scheduled_time": scheduled_dt.isoformat() if scheduled_dt else None,
            "date_formatted": date_local or "Non spécifié",
            "time_formatted": time_local or "Non spécifié",
            "status": getattr(status_val, "value", "unknown").lower(),
            "client": {
                "id": getattr(cli, "id", None),
                "first_name": getattr(cli_user, "first_name", "") if cli_user else "",
                "last_name":  getattr(cli_user, "last_name",  "") if cli_user else "",
                "email":      getattr(cli_user, "email",      "") if cli_user else "",
                "full_name": self.customer_full_name,
            },
            "company": self.company.name if self.company else "Non assignée",
            "driver": self.driver.user.username if (self.driver and self.driver.user) else "Non assigné",
            "driver_id": self.driver_id,
            "duration_seconds": self.duration_seconds,
            "distance_meters": self.distance_meters,
            "medical_facility": self.medical_facility or "Non spécifié",
            "doctor_name": self.doctor_name or "Non spécifié",
            "hospital_service": self.hospital_service or "Non spécifié",
            "notes_medical": self.notes_medical or "Aucune note",
            "created_at": created_loc.strftime("%d/%m/%Y %H:%M") if created_loc else "Non spécifié",
            "updated_at": updated_loc.strftime("%d/%m/%Y %H:%M") if updated_loc else "Non spécifié",
            "rejected_by": self.rejected_by,
            "is_round_trip": _as_bool(self.is_round_trip),
            "is_return": _as_bool(self.is_return),
            "has_return": self.return_trip is not None,
            "boarded_at": iso_utc_z(to_utc_from_db(boarded_dt)) if boarded_dt else None,
            "completed_at": iso_utc_z(to_utc_from_db(completed_dt)) if completed_dt else None,
            "duree_minutes": (int((completed_dt - boarded_dt).total_seconds() // 60)
                              if (completed_dt and boarded_dt) else None),
            "duration_in_minutes": self.duration_in_minutes,
            "billing": {
                "billed_to_type": (_as_str(self.billed_to_type) or "patient"),
                "billed_to_company": self.billed_to_company.serialize if self.billed_to_company else None,
                "billed_to_contact": self.billed_to_contact
            },
            "patient_name": _as_str(self.customer_name),
        }

    @staticmethod
    def auto_geocode_if_needed(_booking):
        return False

    @validates('user_id')
    def validate_user_id(self, _key, user_id):
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("L'ID utilisateur doit être un entier positif.")
        return user_id

    @validates('is_return')
    def validate_is_return(self, _key, val):
        return bool(val)

    @validates('amount')
    def validate_amount(self, _key, amount):
        if amount <= 0:
            raise ValueError("Le montant doit être supérieur à 0")
        return round(amount, 2)

    @validates('scheduled_time')
    def validate_scheduled_time(self, _key, scheduled_time):
        st = parse_local_naive(scheduled_time)
        if st and st < now_local():
            raise ValueError("Heure prévue dans le passé.")
        return st

    @validates('customer_name')
    def validate_customer_name(self, _key, name):
        if not name or len(name.strip()) == 0:
            raise ValueError("Le nom du client ne peut pas être vide")
        if len(name) > 100:
            raise ValueError("Le nom du client ne peut pas dépasser 100 caractères")
        return name

    @validates('pickup_location', 'dropoff_location')
    def validate_location(self, key, location):
        if not location or len(location.strip()) == 0:
            raise ValueError(f"{key} ne peut pas être vide")
        if len(location) > 200:
            raise ValueError(f"{key} ne peut pas dépasser 200 caractères")
        return location

    @validates('status')
    def validate_status(self, _key, status):
        if isinstance(status, str):
            status = status.upper()
            try:
                status = BookingStatus[status]
            except KeyError:
                raise ValueError(f"Statut invalide : {status}. Doit être l'un de {list(BookingStatus.__members__.keys())}")
        if not isinstance(status, BookingStatus):
            raise ValueError(f"Statut invalide : {status}. Doit être un BookingStatus valide.")
        return status

    @validates('driver_id')
    def validate_driver_id(self, _key, value):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("driver_id doit être un entier positif ou null")
        return value

    def is_future(self) -> bool:
        st = _as_dt(self.scheduled_time)
        return bool(st and st > now_local())

    def update_status(self, new_status):
        if not isinstance(new_status, BookingStatus):
            raise ValueError("Statut invalide.")
        self.status = new_status

    @property
    def duration_in_minutes(self) -> Optional[int]:
        b = _as_dt(self.boarded_at)
        c = _as_dt(self.completed_at)
        if b and c:
            return int((c - b).total_seconds() // 60)
        return None

    def to_dict(self):
        return self.serialize

    def is_assignable(self) -> bool:
        st = _as_dt(self.scheduled_time)
        status_val = cast(BookingStatus, self.status)
        return (status_val in (BookingStatus.PENDING, BookingStatus.ACCEPTED)) and bool(st and st > now_local())

    def assign_driver(self, driver_id: int):
        if not self.is_assignable():
            raise ValueError("La réservation ne peut pas être attribuée actuellement.")
        current_driver_id = _as_int(getattr(self, "driver_id", None), 0)
        target_driver_id  = _as_int(driver_id, 0)
        if current_driver_id == target_driver_id:
            return
        self.driver_id = driver_id
        self.status = BookingStatus.ASSIGNED
        self.updated_at = datetime.now(timezone.utc)

    def cancel_booking(self):
        if self.status not in [BookingStatus.ASSIGNED, BookingStatus.ACCEPTED]:
            raise ValueError("Seules les réservations en cours peuvent être annulées.")
        self.status = BookingStatus.CANCELED
        self.updated_at = datetime.now(timezone.utc)

    @validates('billed_to_type')
    def _v_billed_to_type(self, _key, value):
        v = (value or 'patient').lower().strip()
        if v not in ('patient', 'clinic', 'insurance'):
            raise ValueError("billed_to_type invalide (patient|clinic|insurance)")
        return v

    @validates('billed_to_company_id')
    def _v_billed_to_company_id(self, _key, value):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("billed_to_company_id doit être un entier positif ou NULL")
        current_type = _as_str(getattr(self, "billed_to_type", None)) or "patient"
        if current_type.strip().lower() == "patient":
            return None
        return value

    @staticmethod
    def _enforce_billing_exclusive(_mapper, _connection, target: "Booking") -> None:
        btype = (_as_str(getattr(target, "billed_to_type", None)) or "patient").strip().lower()
        if btype == "patient":
            target.billed_to_company_id = None
            return
        company_id = _as_int(getattr(target, "billed_to_company_id", None), 0)
        if company_id <= 0:
            raise ValueError("billed_to_company_id est obligatoire si billed_to_type n'est pas 'patient'")

# Hooks ORM
event.listen(Booking, "before_insert", Booking._enforce_billing_exclusive)
event.listen(Booking, "before_update", Booking._enforce_billing_exclusive)

class Payment(db.Model):
    __tablename__ = 'payment'
    __table_args__ = (
        CheckConstraint('amount > 0', name='chk_payment_amount_positive'),
        Index('ix_payment_booking_status', 'booking_id', 'status'),
        Index('ix_payment_client_date', 'client_id', 'date'),
    )

    id = Column(Integer, primary_key=True)

    amount = Column(Float, nullable=False)

    # date du paiement et MAJ
    date = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False,
                        server_default=func.now(), onupdate=func.now())

    # méthode + statut (Enums PG)
    method = Column(
        SAEnum('credit_card', 'paypal', 'bank_transfer', 'cash', name='payment_method'),
        nullable=False
    )
    status = Column(
        SAEnum(PaymentStatus, name="payment_status"),
        nullable=False,
        default=PaymentStatus.PENDING,
        server_default=PaymentStatus.PENDING.value,
    )


    # FKs (indexées) — uniformisées en Column(...)
    user_id    = Column(Integer, ForeignKey('user.id', ondelete='CASCADE'),    nullable=False, index=True)
    client_id  = Column(Integer, ForeignKey('client.id', ondelete='CASCADE'),  nullable=False, index=True)
    booking_id = Column(Integer, ForeignKey('booking.id', ondelete='CASCADE'), nullable=False, index=True)

    # Relations
    client  = relationship('Client',  back_populates='payments', passive_deletes=True)
    booking = relationship('Booking', back_populates='payments', passive_deletes=True)
    # (facultatif) user = relationship('User', passive_deletes=True)

    @property
    def serialize(self):
        amt = _as_float(self.amount, 0.0)
        bk = self.booking
        return {
            "id": self.id,
            "amount": amt,
            "method": str(self.method),  # SAEnum -> str
            "status": (self.status.value if isinstance(self.status, PaymentStatus) else str(self.status)),
            "date": _iso(self.date),
            "updated_at": _iso(self.updated_at),
            "client_id": self.client_id,
            "booking_id": self.booking_id,
            "booking_info": {
                "pickup_location": bk.pickup_location if bk else None,
                "dropoff_location": bk.dropoff_location if bk else None,
                "scheduled_time": _iso(bk.scheduled_time) if bk else None,
            }
        }

    # -------- Validations --------
    @validates('user_id')
    def validate_user_id(self, _key, user_id):
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("L'ID utilisateur pour Payment doit être un entier positif.")
        return user_id

    @validates('amount')
    def validate_amount(self, _key, amount):
        if amount <= 0:
            raise ValueError("Le montant doit être supérieur à 0")
        return round(amount, 2)

    @validates('method')
    def validate_method(self, _key, method):
        # Si tu gardes SAEnum ci-dessus, SQLAlchemy/PG bloquent déjà les valeurs invalides.
        allowed = {'credit_card', 'paypal', 'bank_transfer', 'cash'}
        if isinstance(method, str):
            method = method.strip()
        if method not in allowed:
            raise ValueError(f"Méthode de paiement invalide. Autorisées : {', '.join(sorted(allowed))}")
        return method

    @validates('date')
    def validate_date(self, _key, value):
        if value is None:
            raise ValueError("La date de paiement ne peut pas être nulle")
        if value > datetime.now(timezone.utc):
            raise ValueError("La date de paiement ne peut pas être dans le futur")
        return value

    @validates('status')
    def validate_status(self, _key, status):
        if isinstance(status, str):
            key = status.upper().strip()
            if key not in PaymentStatus.__members__:
                raise ValueError(f"Statut de paiement invalide : {status}. Attendu: {list(PaymentStatus.__members__.keys())}")
            status = PaymentStatus[key]
        if not isinstance(status, PaymentStatus):
            raise ValueError("Statut invalide (PaymentStatus attendu).")
        return status


    # -------- Métier --------
    def update_status(self, new_status):
        if isinstance(new_status, str):
            try:
                new_status = PaymentStatus(new_status.lower())
            except Exception:
                raise ValueError("Statut de paiement invalide.")
        if not isinstance(new_status, PaymentStatus):
            raise ValueError("Statut de paiement invalide.")
        self.status = new_status


class Invoice(db.Model):
    __tablename__ = "invoice"
    __table_args__ = (
        CheckConstraint('amount > 0', name='chk_invoice_amount_positive'),
        # 👉 Active ceci si **une seule** facture par booking :
        UniqueConstraint('booking_id', name='uq_invoice_booking_one'),
        Index('ix_invoice_company_status_due', 'company_id', 'status', 'due_date'),
        Index('ix_invoice_user_created', 'user_id', 'created_at'),
    )

    id = Column(Integer, primary_key=True)
    reference = Column(String(32), unique=True, nullable=False, index=True,
                       default=lambda: Invoice.generate_reference())
    amount = Column(Float, nullable=False)

    user_id = Column(Integer, ForeignKey('user.id', ondelete="CASCADE",  name="fk_invoice_user"),
                     nullable=False, index=True)
    booking_id = Column(Integer, ForeignKey('booking.id', ondelete="SET NULL", name="fk_invoice_booking"),
                        nullable=True, index=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE", name="fk_invoice_company"),
                        nullable=False, index=True)

    details = Column(Text)
    pdf_url = Column(String(255))
    due_date = Column(DateTime(timezone=True))
    paid_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    status = Column(
        SAEnum(InvoiceStatus, name="invoice_status"),
        nullable=False,
        default=InvoiceStatus.UNPAID,
        server_default=InvoiceStatus.UNPAID.value,
    )


    # Relations
    user = relationship("User", back_populates="invoices", passive_deletes=True)
    # ⚠️ si tu veux du 1↔1 : ajoute uselist=False côté Booking (voir note plus bas)
    booking = relationship("Booking", back_populates="invoice", passive_deletes=True)
    # Option: passer aussi en back_populates côté Company pour uniformiser
    company = relationship("Company", backref="invoices", passive_deletes=True)

    # ---------- Sérialisation ----------
    @property
    def serialize(self):
        st = self.status
        status_str = getattr(st, "value", None) or (str(st) if st is not None else "unknown")
        return {
            "id": self.id,
            "reference": self.reference,
            "amount": self.amount,
            "user_id": self.user_id,
            "user": self.user.serialize if self.user else None,
            "booking_id": self.booking_id,
            "booking": self.booking.serialize if self.booking else None,
            "company_id": self.company_id,
            "company": self.company.serialize if self.company else None,
            "details": self.details,
            "pdf_url": self.pdf_url,
            "status": status_str,
            "status_human": self.status_human,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "due_date": _iso(self.due_date),
            "paid_at": _iso(self.paid_at),
        }

    @property
    def status_human(self) -> str:
        status_map = {"unpaid": "Non payée", "paid": "Payée", "canceled": "Annulée"}
        raw = getattr(self.status, "value", self.status)
        key = cast(str, raw if isinstance(raw, str) else str(raw))
        return status_map.get(key, "Inconnu")

    # ---------- Référence ----------
    @staticmethod
    def generate_reference():
        import random
        import string
        prefix = "FCT"
        suffix = ''.join(random.choices(string.digits, k=7))
        return f"{prefix}-{suffix}"

    # ---------- Validations ----------
    @validates('user_id')
    def validate_user_id(self, _key, user_id):
        if user_id is None or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("ID utilisateur invalide")
        return user_id

    @validates('amount')
    def validate_amount(self, _key, amount):
        if amount is None or amount <= 0:
            raise ValueError("Le montant doit être supérieur à 0")
        return round(amount, 2)

    @validates('booking_id')
    def validate_booking_id(self, _key, booking_id):
        if booking_id is not None and booking_id <= 0:
            raise ValueError("ID de réservation invalide")
        return booking_id

    @validates('company_id')
    def validate_company_id(self, _key, company_id):
        if not isinstance(company_id, int) or company_id <= 0:
            raise ValueError("ID de l'entreprise invalide")
        return company_id

    @validates('reference')
    def validate_reference(self, _key, ref):
        if not ref or len(ref) > 32:
            raise ValueError("Référence de facture invalide")
        return ref
    
    @validates('status')
    def validate_status(self, _key, status):
        if isinstance(status, str):
            key = status.upper().strip()
            if key not in InvoiceStatus.__members__:
                raise ValueError(f"Statut de facture invalide : {status}. Attendu: {list(InvoiceStatus.__members__.keys())}")
            status = InvoiceStatus[key]
        if not isinstance(status, InvoiceStatus):
            raise ValueError("Statut de facture invalide.")
        return status


    # ---------- Actions ----------
    def mark_as_paid(self):
        self.status = InvoiceStatus.PAID
        self.paid_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_as_unpaid(self):
        self.status = InvoiceStatus.UNPAID
        self.paid_at = None
        self.updated_at = datetime.now(timezone.utc)

    def cancel(self):
        self.status = InvoiceStatus.CANCELED
        self.updated_at = datetime.now(timezone.utc)

    def is_overdue(self):
        return self.due_date and datetime.now(timezone.utc) > self.due_date and self.status != InvoiceStatus.PAID

    def __repr__(self):
        return f"<Invoice {self.reference} | {self.amount:.2f} CHF | {getattr(self.status, 'value', self.status)}>"

class Message(db.Model):
    __tablename__ = "message"
    __table_args__ = (
        # tri/filtrage fréquents : société → destinataire → non lus → récents
        Index("ix_msg_company_receiver_unread_ts", "company_id", "receiver_id", "is_read", "timestamp"),
        CheckConstraint("sender_role IN ('DRIVER','COMPANY')", name='check_sender_role_valid'),
    )

    id = Column(Integer, primary_key=True)

    company_id = Column(Integer,
                        ForeignKey('company.id', ondelete="CASCADE"),
                        nullable=False, index=True)

    sender_id = Column(Integer,
                       ForeignKey('user.id', ondelete="SET NULL"),
                       nullable=True, index=True)

    receiver_id = Column(Integer,
                         ForeignKey('user.id', ondelete="SET NULL"),
                         nullable=True, index=True)

    sender_role = Column(SAEnum(SenderRole, name="sender_role"), nullable=False)  # "driver" | "company"
    content = Column(Text, nullable=False)

    # timestamptz côté PG grâce à timezone=True
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    is_read = Column(Boolean, nullable=False, default=False)

    # Relations
    sender = relationship('User', foreign_keys=[sender_id], lazy='joined')
    receiver = relationship('User', foreign_keys=[receiver_id], lazy='joined')
    company = relationship('Company', lazy='joined')

    def __repr__(self):
        return f"<Message {self.id} from {getattr(self.sender_role, 'value', self.sender_role)} ({self.sender_id})>"

    @property
    def serialize(self):
        return {
            "id": self.id,
            "company_id": self.company_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "sender_role": getattr(self.sender_role, "value", self.sender_role),
            "content": self.content,
            "timestamp": _iso(self.timestamp),
            "is_read": _as_bool(self.is_read),
        }

    # -------- Validateurs --------
    @validates('company_id')
    def _v_company_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("company_id invalide.")
        return v

    @validates('sender_role')
    def _v_sender_role(self, _k, v):
        # accepte string et convertit vers l'enum
        if isinstance(v, str):
            try:
                v = SenderRole(v.lower())
            except ValueError:
                raise ValueError("Le rôle de l'expéditeur doit être 'DRIVER' ou 'COMPANY'")
        if not isinstance(v, SenderRole):
            raise ValueError("sender_role invalide.")
        return v

    @validates('content')
    def _v_content(self, _k, text):
        if not text or not str(text).strip():
            raise ValueError("Le contenu du message ne peut pas être vide.")
        return text.strip()

class FavoritePlace(db.Model):
    __tablename__ = "favorite_place"
    __table_args__ = (
        # Évite les doublons d’adresse au sein d’une même entreprise
        UniqueConstraint("company_id", "address", name="uq_fav_company_address"),
        # Accélère la recherche par libellé pour une entreprise
        Index("ix_fav_company_label", "company_id", "label"),
        # Accélère les requêtes par coordonnées (proches)
        Index("ix_fav_company_coords", "company_id", "lat", "lon"),
        # Verrouille les bornes géographiques
        CheckConstraint("lat BETWEEN -90 AND 90", name="chk_fav_lat"),
        CheckConstraint("lon BETWEEN -180 AND 180", name="chk_fav_lon"),
        # OPTIONNEL (PG) : unicité “case-insensitive” si vous normalisez label en lower()
        # Index("uq_fav_company_label_lower", text("company_id"), text("lower(label)"), unique=True),
    )

    id = Column(Integer, primary_key=True)

    # Entreprise propriétaire du favori
    company_id = Column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Libellé affiché (ex. "HUG – Urgences")
    label = Column(String(200), nullable=False)

    # Adresse canonique (ex. "Rue Gabrielle-Perret-Gentil 4, 1205 Genève")
    address = Column(String(255), nullable=False)

    # Coordonnées GPS
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    # Tags libres (ex. "hospital;emergency")
    tags = Column(String(200))

    # Timestamps (pratiques pour audit/tri) – côté Postgres
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ---------- Normalisation & validations ----------

    @staticmethod
    def _norm_text(s: str) -> str:
        return (s or "").strip()

    @staticmethod
    def _norm_address(s: str) -> str:
        # Normalisation simple (trim + collapse espaces)
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
        # Trim simple; garder le format "tag1;tag2" si tu le souhaites
        return self._norm_text(value)

    # ---------- Helpers ----------

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

    # Backward-compat pour ton code existant
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
        # OPTIONNEL (PG) : si tu veux une recherche plus rapide par nom :
        # Index("ix_med_estab_name", "name"),
    )

    id = Column(Integer, primary_key=True)

    # type générique: "hospital", "clinic", "ems", ...
    type = Column(String(50), nullable=False, default="hospital")

    name = Column(String(200), nullable=False)           # "HUG"
    display_name = Column(String(255), nullable=False)   # "HUG - Hôpitaux Universitaires de Genève"
    address = Column(String(255), nullable=False)        # adresse canonique
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    # Alias de recherche: "hug;hôpital cantonal;hopital geneve"
    aliases = Column(String(500), nullable=True)
    active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    services = db.relationship(
        "MedicalService",
        backref="establishment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # ---------- Helpers ----------

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

    # ---------- Validations ----------

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
        # liste ouverte mais normalisée
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
        # Optionnel : accélère les recherches par nom au sein d’un établissement
        # Index("ix_med_service_estab_name", "establishment_id", "name"),
    )

    id = Column(Integer, primary_key=True)
    establishment_id = Column(
        Integer,
        ForeignKey("medical_establishment.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # "Service", "Département", "Unité", "Laboratoire", "Groupe"
    category = Column(String(50), nullable=False, default="Service")
    name = Column(String(200), nullable=False)
    slug = Column(String(200), nullable=True)

    # --- Localisation / contact ---
    address_line = Column(String(255), nullable=True)   # ex. "Rue Gabrielle-Perret-Gentil 4"
    postcode = Column(String(16), nullable=True)
    city = Column(String(100), nullable=True)
    building = Column(String(120), nullable=True)       # ex. "Bât Prévost"
    floor = Column(String(60), nullable=True)           # ex. "étage P" / "3e étage"
    site_note = Column(String(255), nullable=True)      # ex. "Maternité", "Hôpital des enfants"
    phone = Column(String(40), nullable=True)
    email = Column(String(120), nullable=True)

    lat = Column(Float, nullable=True)                  # optionnel
    lon = Column(Float, nullable=True)

    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relation côté établissement déjà fournie par MedicalEstablishment.services (backref="establishment")

    # ---------- Sérialisation ----------
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

    # ---------- Validations ----------
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


# =========================
#  Dispatch / Temps Réel
# =========================


class DispatchRun(db.Model):
    __tablename__ = "dispatch_run"
    __table_args__ = (
        UniqueConstraint('company_id', 'day', name='uq_dispatch_run_company_day'),
        Index('ix_dispatch_run_company_day', 'company_id', 'day'),
        Index('ix_dispatch_run_company_status_day', 'company_id', 'status', 'day'),
        # ❌ plus besoin de CheckConstraint manuel : SAEnum crée un CHECK
    )
    __mapper_args__ = {"eager_defaults": True}

    # ---------- Colonnes typées ----------
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('company.id', ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date, nullable=False)

    # ✅ Enum fort + CHECK auto (native_enum=False => CHECK plutôt qu'un type ENUM DB)
    status: Mapped[DispatchStatus] = mapped_column(
        SAEnum(DispatchStatus, name="dispatch_status", native_enum=False),
        default=DispatchStatus.PENDING,
        nullable=False,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
    )

    # 🔁 MutableDict pour suivre les modifs in-place
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(MutableDict.as_mutable(JSONB()))
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(MutableDict.as_mutable(JSONB()))

    # Optionnel : optimistic locking (décommente si tu veux l'activer)
    # version_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # __mapper_args__ = {"version_id_col": version_id, "version_id_generator": True}

    # ---------- Relations typées ----------
    company: Mapped["Company"] = relationship(back_populates='dispatch_runs', passive_deletes=True)
    assignments: Mapped[List["Assignment"]] = relationship(
        back_populates='dispatch_run',
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    # ---------- Méthodes métier ----------
    def mark_started(self) -> None:
        self.status = DispatchStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self, metrics: Optional[dict] = None) -> None:
        self.status = DispatchStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if metrics:
            self.metrics = {**(self.metrics or {}), **dict(metrics)}

    def mark_failed(self, reason: Optional[str] = None) -> None:
        self.status = DispatchStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        if reason:
            self.metrics = {**(self.metrics or {}), "error": reason}

    # ---------- Validations ----------
    @validates('company_id')
    def _v_company_id(self, _k: str, v: Any) -> int:
        if not isinstance(v, int) or v <= 0:
            raise ValueError("company_id invalide")
        return v

    @validates('day')
    def _v_day(self, _k: str, v: Any) -> date:
        if not isinstance(v, date):
            raise ValueError("'day' doit être un objet date")
        return v

    @validates('status')
    def _v_status(self, _k: str, v: Any) -> DispatchStatus:
        if isinstance(v, DispatchStatus):
            return v
        if isinstance(v, str):
            s = v.strip()
            # d'abord par NAME (PENDING/RUNNING/...)
            try:
                return DispatchStatus[s.upper()]
            except KeyError:
                pass
            # puis par VALUE (également en upper)
            try:
                return DispatchStatus(s.upper())
            except Exception:
                pass
        raise ValueError(f"status invalide: {v}")


    @validates('config', 'metrics')
    def _v_jsonb(self, _k: str, v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("config/metrics doivent être des objets JSON (dict)")
        return v

    # ---------- Sérialisation ----------
    @property
    def serialize(self) -> Dict[str, Any]:
        # status.value si Enum, sinon str
        status_str = self.status.value if isinstance(self.status, DispatchStatus) else str(self.status) or "pending"
        return {
            "id": self.id,
            "company_id": self.company_id,
            "day": self.day.isoformat(),
            "status": status_str,
            "started_at": _iso(_as_dt(self.started_at)),
            "completed_at": _iso(_as_dt(self.completed_at)),
            "created_at": _iso(_as_dt(self.created_at)),
            "metrics": self.metrics or {},
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DispatchRun id={self.id} company={self.company_id} day={self.day} status={self.status}>"



class Assignment(db.Model):
    __tablename__ = "assignment"
    __table_args__ = (
        UniqueConstraint('dispatch_run_id', 'booking_id', name='uq_assignment_run_booking'),
        Index('ix_assignment_driver_status', 'driver_id', 'status'),
        CheckConstraint('delay_seconds >= 0', name='ck_assignment_delay_nonneg'),
    )

    id = Column(Integer, primary_key=True)

    dispatch_run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("dispatch_run.id", ondelete="SET NULL"), index=True, nullable=True
    )
    booking_id = Column(Integer,
                        ForeignKey('booking.id', ondelete="CASCADE"),
                        nullable=False, index=True)
    driver_id = Column(Integer,
                       ForeignKey('driver.id', ondelete="SET NULL"),
                       nullable=True, index=True)

    status = Column(SAEnum(AssignmentStatus, name="assignment_status"),
                nullable=False, 
                default=AssignmentStatus.SCHEDULED)

    # Planifié (plan) & réel (terrain)
    planned_pickup_at = Column(DateTime(timezone=True), nullable=True)
    planned_dropoff_at = Column(DateTime(timezone=True), nullable=True)
    actual_pickup_at = Column(DateTime(timezone=True), nullable=True)
    actual_dropoff_at = Column(DateTime(timezone=True), nullable=True)

    # ETA + retard estimé (secondes)
    eta_pickup_at = Column(DateTime(timezone=True), nullable=True)
    eta_dropoff_at = Column(DateTime(timezone=True), nullable=True)
    delay_seconds = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True),
                        nullable=False,
                        default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True),
                        onupdate=lambda: datetime.now(timezone.utc))

    # Relations
    dispatch_run: Mapped[Optional["DispatchRun"]] = relationship(
        "DispatchRun",
        back_populates="assignments",
    )
    booking = relationship('Booking', backref='assignments', passive_deletes=True)
    driver = relationship('Driver', backref='assignments', passive_deletes=True)

    # --------- Helpers / sérialisation ----------
    @property
    def serialize(self):
        status_val = getattr(self.status, "value", self.status)
        delay: int = _as_int(getattr(self, "delay_seconds", 0), 0)
        return {
            "id": self.id,
            "dispatch_run_id": self.dispatch_run_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "status": status_val,
            "planned_pickup_at": _iso(_as_dt(self.planned_pickup_at)),
            "planned_dropoff_at": _iso(_as_dt(self.planned_dropoff_at)),
            "actual_pickup_at": _iso(_as_dt(self.actual_pickup_at)),
            "actual_dropoff_at": _iso(_as_dt(self.actual_dropoff_at)),
            "eta_pickup_at": _iso(_as_dt(self.eta_pickup_at)),
            "eta_dropoff_at": _iso(_as_dt(self.eta_dropoff_at)),
            "delay_seconds": delay,
            "created_at": _iso(_as_dt(self.created_at)),
            "updated_at": _iso(_as_dt(self.updated_at)),
        }

    def __repr__(self):
        status_str = getattr(self.status, "value", self.status)
        return f"<Assignment id={self.id} booking={self.booking_id} driver={self.driver_id} status={status_str}>"

    # --------- Validations ----------
    @validates('dispatch_run_id')
    def _v_dispatch_run_id(self, _k, v):
        if v is None:
            return None
        if not isinstance(v, int) or v <= 0:
            raise ValueError("dispatch_run_id doit être NULL ou un entier positif.")
        return v

    @validates('booking_id')
    def _v_booking_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("booking_id doit être un entier positif.")
        return v



    @validates('driver_id')
    def _v_driver_id(self, _k, v):
        if v is None:
            return None
        if not isinstance(v, int) or v <= 0:
            raise ValueError("driver_id doit être NULL ou un entier positif.")
        return v

    @validates('status')
    def _v_status(self, _k, value):
        """
        Accepte un AssignmentStatus OU une str (par value 'scheduled' ou par name 'SCHEDULED').
        Évite d'évaluer un Column en truthiness.
        """
        coerced = _coerce_enum(value, AssignmentStatus)
        if coerced is None and isinstance(value, str):
            # tentative par NAME en UPPER (ex: "SCHEDULED")
            try:
                coerced = AssignmentStatus[value.upper()]
            except KeyError:
                pass
        if coerced is None:
            allowed = ", ".join(AssignmentStatus.__members__.keys())
            raise ValueError(f"Statut invalide : {value}. Doit être l'un de {allowed}")
        return coerced

    @validates('delay_seconds')
    def _v_delay(self, _k: str, v: Any) -> int:
        # v peut être None, str, float, etc. → on force un int Python
        val = _as_int(v, 0)   # ton helper déjà défini en haut du fichier
        if val < 0:
            raise ValueError("delay_seconds ne peut pas être négatif.")
        return val
      

    # (optionnel) petite vérification chrono côté app
    def validate_chronology(self):
        """
        ⚠️ On force des datetime Python (ou None) pour éviter les ColumnElement[bool].
        """
        pp = _as_dt(getattr(self, "planned_pickup_at", None))
        pd = _as_dt(getattr(self, "planned_dropoff_at", None))
        ap = _as_dt(getattr(self, "actual_pickup_at", None))
        ad = _as_dt(getattr(self, "actual_dropoff_at", None))

        if pp and pd and pd < pp:
            raise ValueError("planned_dropoff_at < planned_pickup_at")
        if ap and ad and ad < ap:
            raise ValueError("actual_dropoff_at < actual_pickup_at")


class DriverStatus(db.Model):
    __tablename__ = "driver_status"
    __table_args__ = (
        Index('ix_driver_status_state_nextfree', 'state', 'next_free_at'),
        CheckConstraint("latitude IS NULL OR (latitude BETWEEN -90 AND 90)", name="ck_driver_status_lat"),
        CheckConstraint("longitude IS NULL OR (longitude BETWEEN -180 AND 180)", name="ck_driver_status_lon"),
        CheckConstraint("heading IS NULL OR (heading >= 0 AND heading <= 360)", name="ck_driver_status_heading"),
        CheckConstraint("speed IS NULL OR speed >= 0", name="ck_driver_status_speed"),
    )

    id = Column(Integer, primary_key=True)

    driver_id = Column(Integer, ForeignKey('driver.id', ondelete="CASCADE"),
                       nullable=False, unique=True, index=True)

    state = Column(
        SAEnum(DriverState, name="driver_state"),
        nullable=False,
        default=DriverState.AVAILABLE,
        server_default=DriverState.AVAILABLE.value,
    )


    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)   # degrés [0..360]
    speed = Column(Float, nullable=True)     # m/s, ≥ 0
    next_free_at = Column(DateTime(timezone=True), nullable=True)

    current_assignment_id = Column(
        Integer,
        ForeignKey('assignment.id', ondelete="SET NULL"),
        nullable=True
    )

    last_update = Column(DateTime(timezone=True),
                         nullable=False,
                         default=lambda: datetime.now(timezone.utc))

    # Relations (1↔1 avec Driver)
    driver = relationship('Driver', backref='status', uselist=False, passive_deletes=True)
    current_assignment = relationship('Assignment', passive_deletes=True)

    # -------- Sérialisation --------

    @property
    def serialize(self):
        st_any = getattr(self, "state", None)
        state_str = st_any.value if isinstance(st_any, DriverState) else _as_str(st_any)

        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "state": state_str,  # ✅ plus de truthiness sur une Column
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": self.heading,
            "speed": self.speed,
            "next_free_at": _iso(self.next_free_at),
            "current_assignment_id": self.current_assignment_id,
            "last_update": _iso(self.last_update),
        }

    def __repr__(self):
        st_any = getattr(self, "state", None)
        state_str = st_any.value if isinstance(st_any, DriverState) else _as_str(st_any)
        return f"<DriverStatus driver={self.driver_id} state={state_str} next_free_at={_iso(self.next_free_at)}>"


    # -------- Validateurs --------
    @validates('driver_id')
    def _v_driver_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("driver_id doit être un entier positif.")
        return v

    @validates('current_assignment_id')
    def _v_current_assignment_id(self, _k, v):
        if v is None:
            return None
        if not isinstance(v, int) or v <= 0:
            raise ValueError("current_assignment_id doit être NULL ou un entier positif.")
        return v

    @validates('state')
    def _v_state(self, _k, state):
        if isinstance(state, str):
            try:
                state = DriverState[state.upper()]
            except KeyError:
                raise ValueError(f"state invalide. Valeurs autorisées : {[s.value for s in DriverState]}")
        if not isinstance(state, DriverState):
            raise ValueError("state invalide (Enum attendu).")
        return state

    @validates('latitude')
    def _v_lat(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < -90 or v > 90:
            raise ValueError("latitude hors bornes [-90, 90].")
        return v

    @validates('longitude')
    def _v_lon(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < -180 or v > 180:
            raise ValueError("longitude hors bornes [-180, 180].")
        return v

    @validates('heading')
    def _v_heading(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < 0 or v > 360:
            raise ValueError("heading doit être entre 0 et 360 degrés.")
        return v

    @validates('speed')
    def _v_speed(self, _k, v):
        if v is None:
            return None
        v = float(v)
        if v < 0:
            raise ValueError("speed ne peut pas être négative.")
        return v

    # -------- Helpers métier (optionnels) --------
    def mark_available(self, when: Optional[datetime] = None):
        self.state = DriverState.AVAILABLE
        self.next_free_at = when
        self.last_update = datetime.now(timezone.utc)

    def mark_busy(self, next_free_at: Optional[datetime] = None):
        self.state = DriverState.BUSY
        self.next_free_at = next_free_at
        self.last_update = datetime.now(timezone.utc)

    def touch_location(self, lat: float, lon: float, heading: Optional[float] = None, speed: Optional[float] = None):
        self.latitude = lat
        self.longitude = lon
        if heading is not None:
            self.heading = heading
        if speed is not None:
            self.speed = speed
        self.last_update = datetime.now(timezone.utc)


class RealtimeEvent(db.Model):
    __tablename__ = "realtime_event"
    __table_args__ = (
        Index('idx_realtime_event_company_type_time', 'company_id', 'event_type', 'timestamp'),
        Index('idx_realtime_event_entity_time', 'entity_type', 'entity_id', 'timestamp'),
        CheckConstraint("entity_id > 0", name="ck_realtime_entity_id_positive"),
        Index('ix_realtime_event_data_gin', 'data', postgresql_using='gin'),
    )

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"),
                        nullable=False, index=True)

    event_type  = Column(SAEnum(RealtimeEventType,  name="realtime_event_type"),  nullable=False)
    entity_type = Column(SAEnum(RealtimeEntityType, name="realtime_entity_type"), nullable=False)


    entity_id = Column(Integer, nullable=False)
    data = Column(JSONB, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    company = relationship('Company', passive_deletes=True)

    def __repr__(self):
        et_any = getattr(self, "event_type", None)
        et_str = et_any.value if isinstance(et_any, RealtimeEventType) else _as_str(et_any) or "?"
        en_any = getattr(self, "entity_type", None)
        en_str = en_any.value if isinstance(en_any, RealtimeEntityType) else _as_str(en_any) or "?"
        return f"<RealtimeEvent id={self.id} company={self.company_id} type={et_str} entity={en_str}:{self.entity_id}>"

    @property
    def serialize(self):
        et_any = getattr(self, "event_type", None)
        et_str = et_any.value if isinstance(et_any, RealtimeEventType) else _as_str(et_any)

        en_any = getattr(self, "entity_type", None)
        en_str = en_any.value if isinstance(en_any, RealtimeEntityType) else _as_str(en_any)

        return {
            "id": self.id,
            "company_id": self.company_id,
            "event_type": et_str,     # ✅ plus de "if self.event_type"
            "entity_type": en_str,    # ✅ idem
            "entity_id": self.entity_id,
            "data": self.data,
            "timestamp": _iso(self.timestamp),
        }

    # -------- Validateurs --------
    @validates('entity_id')
    def _v_entity_id(self, _k, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("entity_id doit être un entier positif.")
        return v

    @validates('event_type')
    def _v_event_type(self, _k, v):
        if isinstance(v, str):
            try:
                v = RealtimeEventType[v.upper()]
            except KeyError:
                raise ValueError(f"event_type invalide. Valeurs autorisées : "
                                 f"{[e.value for e in RealtimeEventType]}")
        if not isinstance(v, RealtimeEventType):
            raise ValueError("event_type invalide (Enum attendu).")
        return v

    @validates('entity_type')
    def _v_entity_type(self, _k, v):
        if isinstance(v, str):
            try:
                v = RealtimeEntityType[v.upper()]
            except KeyError:
                raise ValueError(f"entity_type invalide. Valeurs autorisées : "
                                 f"{[e.value for e in RealtimeEntityType]}")
        if not isinstance(v, RealtimeEntityType):
            raise ValueError("entity_type invalide (Enum attendu).")
        return v
