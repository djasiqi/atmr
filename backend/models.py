import os
import uuid
import re
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, Float, UniqueConstraint, Enum as SQLEnum, ForeignKey, DateTime, Boolean, inspect, func, Text, event, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy_utils import StringEncryptedType 
from sqlalchemy import CheckConstraint
from sqlalchemy_utils.types.encrypted.encrypted_type import AesEngine
from datetime import datetime, date, timezone
from enum import Enum
from enum import Enum as PyEnum # Importer la classe Enum
import pytz
from ext import db
import json
from sqlalchemy.types import TypeDecorator, TEXT
from sqlalchemy.orm import validates
import enum
from sqlalchemy import Enum as SQLAlchemyEnum
from shared.time_utils import to_utc, now_utc, to_geneva_local, format_geneva, iso_utc_z, to_utc_from_db, parse_local_naive, now_local, split_date_time_local
# --- models.py (en-tête : imports standard) ---
import os
import base64
import binascii

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
        # normalise le padding si absent
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

    # 3) Fail fast explicite (ne JAMAIS exiger ENCRYPTION_KEY si B64 est présente)
    raise RuntimeError("APP_ENCRYPTION_KEY_B64 manquante. Fournis une clé Base64 URL-safe (16/24/32 octets).")

# Expose la clé pour le reste de models.py
_encryption_key = _load_encryption_key()
_encryption_key_str = _encryption_key.hex()

# Définition des énumérations
# ========== ENUMS ==========
class UserRole(str, PyEnum):
    admin = "admin"
    client = "client"
    driver = "driver"
    company = "company"

class BookingStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    ASSIGNED = "assigned"
    EN_ROUTE = "en_route"        # 🚗 Allé ou retour
    IN_PROGRESS = "in_progress"  # 🚕 Client à bord (allé ou retour)
    COMPLETED = "completed"      # 🏁 Allé terminé
    RETURN_COMPLETED = "return_completed"  # 🏁 Retour terminé
    CANCELED = "canceled"


    @classmethod
    def choices(cls):
        return [status.value for status in cls]

class PaymentStatus(str, PyEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

    @classmethod
    def choices(cls):
        return [status.value for status in cls]

class GenderEnum(str, PyEnum):
    homme = "Homme"
    femme = "Femme"
    autre = "Autre"

class ClientType(PyEnum):
    SELF_SERVICE = "self_service"
    PRIVATE = "private"
    CORPORATE = "corporate"

class InvoiceStatus(str, PyEnum):
    UNPAID = "unpaid"
    PAID = "paid"
    CANCELED = "canceled"

    @classmethod
    def choices(cls):
        return [status.value for status in cls]


class DriverType(enum.Enum):
    REGULAR = "REGULAR"
    EMERGENCY = "EMERGENCY"

# ===== Dispatch / Realtime =====
class AssignmentStatus(str, Enum):
    SCHEDULED = "scheduled"
    EN_ROUTE_PICKUP = "en_route_pickup"
    ARRIVED_PICKUP = "arrived_pickup"
    ONBOARD = "onboard"
    EN_ROUTE_DROPOFF = "en_route_dropoff"
    ARRIVED_DROPOFF = "arrived_dropoff"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    REASSIGNED = "reassigned"

# ========== TYPES PERSONNALISÉS ==========
class JSONEncodedList(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return json.dumps(value or [])

    def process_result_value(self, value, dialect):
        try:
            return json.loads(value) if value else []
        except (json.JSONDecodeError, TypeError):
            return []
        

# ========== MODÈLES ==========
class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(36), default=lambda: str(uuid.uuid4()), unique=True, nullable=False, index=True)
    username = db.Column(db.String(100), nullable=False, unique=True, index=True)
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(100), nullable=True, unique=True, index=True)
    
    # ↓ Champs présents pour tous les rôles (client, driver, etc.)
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.String(200), nullable=True)
    birth_date = db.Column(db.Date, nullable=True)
    gender = db.Column(db.Enum(GenderEnum), nullable=True)
    profile_image = db.Column(db.String(255), nullable=True)
    
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(UserRole, name="user_role"), nullable=False, default=UserRole.client)
    reset_token = db.Column(db.String(100), unique=True, nullable=True)
    zip_code = db.Column(db.String(10), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())
    force_password_change = db.Column(db.Boolean, default=False, nullable=False)

    # ✅ Ajout de l'index sur `public_id` pour optimiser les recherches
    __table_args__ = (
        db.Index('idx_public_id', 'public_id'),
    )

    # ✅ Relations bidirectionnelles avec suppression en cascade
    clients = relationship('Client', back_populates='user', uselist=False, passive_deletes=True, cascade="all, delete-orphan")
    driver = relationship('Driver', back_populates='user', uselist=False, cascade="all, delete-orphan", passive_deletes=True)
    company = relationship('Company', back_populates='user', uselist=False, cascade="all, delete-orphan", passive_deletes=True)
    invoices = relationship("Invoice", back_populates="user", cascade="all, delete-orphan", passive_deletes=True)

    # 🔒 Gestion des mots de passe
    def set_password(self, password, force_change=False):
        self.password = generate_password_hash(password)
        self.force_password_change = force_change

    def check_password(self, password):
        pw_hash = self.password
        if isinstance(pw_hash, bytes):
            pw_hash = pw_hash.decode('utf-8')
        return check_password_hash(pw_hash, password)

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
    def validate_gender(self, key, gender):
        if gender and gender not in [g.value for g in GenderEnum]:
            raise ValueError("Genre invalide.")
        return gender
    
    @validates('role')
    def validate_role(self, key, value):
        # Si la valeur est une chaîne, on essaie de la convertir en UserRole.
        if isinstance(value, str):
            try:
                value = UserRole[value.lower()]
            except KeyError:
                raise ValueError("Invalid role value. Allowed values: admin, client, driver, company.")
        return value
    
    @validates('email')
    def validate_email(self, key, email):
        """
        Valide l'email de l'utilisateur.
        Si l'utilisateur est un client self-service, l'email est obligatoire.
        """
        # Si l'email est fourni, on valide le format simple
        if email:
            import re
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                raise ValueError("Format d'email invalide.")
            return email

        # Si l'email n'est pas fourni, on vérifie si c'est un client self-service
        if self.role == UserRole.client and self.clients:
            if self.clients.client_type == ClientType.SELF_SERVICE:
                raise ValueError("L'email est requis pour les clients self-service.")

        return email  # Peut être None pour les autres cas
    
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
            "birth_date": self.birth_date.strftime('%Y-%m-%d') if self.birth_date else None,
            "gender": self.gender.value if self.gender else "Non spécifié",
            "profile_image": self.profile_image or None,
            "role": self.role.value,
            "zip_code": self.zip_code or "Non spécifié",
            "city": self.city or "Non spécifié",
            "created_at": self.created_at.isoformat() if self.created_at else None,
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
    __tablename__ = 'company'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)

    # Adresse opérationnelle (déjà présente)
    address = Column(String(200), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # 📌 NOUVEAU — Adresse de domiciliation (si différente)
    domicile_address_line1 = Column(String(200), nullable=True)
    domicile_address_line2 = Column(String(200), nullable=True)
    domicile_zip = Column(String(10), nullable=True)
    domicile_city = Column(String(100), nullable=True)
    domicile_country = Column(String(2), nullable=True, default="CH")  # ISO 3166-1 alpha-2

    # Contact
    contact_email = Column(String(100), nullable=True)
    contact_phone = Column(String(20), nullable=True)

    # Légal & Facturation
    iban = Column(String(34), nullable=True, index=True)        # IBAN (max 34 chars)
    uid_ide = Column(String(20), nullable=True, index=True)     # IDE Suisse (UID), ex: CHE-123.456.789
    billing_email = Column(String(100), nullable=True)          # email facturation si distinct
    billing_notes = Column(Text, nullable=True)                 # instructions facture (réf, délai, etc.)

    user_id = Column(Integer, ForeignKey('user.id', ondelete="CASCADE", name="fk_company_user"), nullable=False, index=True)
    is_approved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    service_area = Column(String(200), nullable=True)
    max_daily_bookings = Column(Integer, nullable=True, default=50)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    dispatch_enabled = Column(Boolean, default=False)
    is_partner = Column(Boolean, default=False)
    logo_url = Column(String(255), nullable=True) 

    # Relations
    user = relationship('User', back_populates='company', passive_deletes=True)

    clients = relationship(
        'Client',
        back_populates='company',
        passive_deletes=True,
        cascade="all, delete-orphan",
        foreign_keys='Client.company_id'
    )

    drivers = relationship('Driver', back_populates='company')

    bookings = relationship(
        'Booking',
        back_populates='company',
        foreign_keys='Booking.company_id'
    )

    # 🚗 NOUVEAU — véhicules
    vehicles = relationship(
        'Vehicle',
        back_populates='company',
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    @property
    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,

            # opérationnel
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,

            # domiciliation
            "domicile": {
                "line1": self.domicile_address_line1,
                "line2": self.domicile_address_line2,
                "zip": self.domicile_zip,
                "city": self.domicile_city,
                "country": self.domicile_country,
            },

            # contact
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,

            # facturation & légal
            "iban": self.iban,
            "uid_ide": self.uid_ide,
            "billing_email": self.billing_email,
            "billing_notes": self.billing_notes,

            "is_approved": self.is_approved,
            "is_partner": self.is_partner,
            "user_id": self.user_id,
            "service_area": self.service_area,
            "max_daily_bookings": self.max_daily_bookings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "dispatch_enabled": self.dispatch_enabled,
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            
            # logo
            "logo_url": self.logo_url,
            # véhicules (liste)
            "vehicles": [v.serialize for v in self.vehicles],
        }

    # -------- Validateurs --------
    @validates('contact_email', 'billing_email')
    def validate_any_email(self, key, value):
        if not value:
            return value
        v = value.strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError(f"Format d'email invalide pour {key}.")
        return v

    @validates('contact_phone')
    def validate_contact_phone(self, key, value):
        if not value:
            return value
        v = value.strip()
        # +, espaces, -, (), 7 à 20 caractères
        if not re.match(r'^\+?[0-9\s\-\(\)]{7,20}$', v):
            raise ValueError("Numéro de téléphone invalide.")
        return v

    @validates('iban')
    def validate_iban(self, key, value):
        if not value:
            return value
        v = value.replace(' ', '').upper()
        # Format de base
        if len(v) < 15 or len(v) > 34 or not v[:2].isalpha() or not v[2:4].isdigit():
            raise ValueError("IBAN invalide (format).")
        # Checksum IBAN (mod 97) – conversion A=10..Z=35
        rearranged = v[4:] + v[:4]
        try:
            converted = ''.join(str(int(ch, 36)) for ch in rearranged)
        except ValueError:
            raise ValueError("IBAN invalide (caractères non autorisés).")
        # Calcul mod 97 par tranches (évite les très grands entiers)
        remainder = 0
        for i in range(0, len(converted), 9):
            remainder = int(str(remainder) + converted[i:i+9]) % 97
        if remainder != 1:
            raise ValueError("IBAN invalide (checksum).")
        return v

    @validates('uid_ide')
    def validate_uid_ide(self, key, value):
        if not value:
            return value
        v = value.strip().upper()
        # Accepte: CHE-123.456.789 / CHE123456789 (+ optionnel 'TVA')
        if not re.match(r'^CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?$|^CHE[- ]?\d{9}(\s*TVA)?$', v, flags=re.IGNORECASE):
            raise ValueError("IDE/UID suisse invalide (ex: CHE-123.456.789).")
        # Normalisation vers "CHE-123.456.789" (+ ' TVA' si présent)
        digits = re.sub(r'\D', '', v)[:9]
        v_norm = f"CHE-{digits[0:3]}.{digits[3:6]}.{digits[6:9]}"
        if 'TVA' in v:
            v_norm += ' TVA'
        return v_norm

    @validates('name')
    def validate_name(self, key, name):
        if not name or len(name.strip()) == 0:
            raise ValueError("Le nom de l'entreprise ne peut pas être vide.")
        if len(name) > 100:
            raise ValueError("Le nom de l'entreprise ne peut pas dépasser 100 caractères.")
        return name.strip()

    @validates('user_id')
    def validate_user_id(self, key, user_id):
        if not user_id or user_id <= 0:
            raise ValueError("ID utilisateur invalide.")
        return user_id

    def toggle_approval(self):
        self.is_approved = not self.is_approved
        return self.is_approved

    def approve(self):
        from datetime import datetime, timezone
        self.is_approved = True
        self.accepted_at = datetime.now(timezone.utc)

    def can_dispatch(self):
        return self.is_approved and self.dispatch_enabled

    def __repr__(self):
        return f"<Company {self.name} | ID: {self.id} | Approved: {self.is_approved}>"

    def to_dict(self):
        return self.serialize.copy()

class Vehicle(db.Model):
    __tablename__ = 'vehicle'
    __table_args__ = (
        UniqueConstraint('company_id', 'license_plate', name='uq_company_plate'),
    )

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=False, index=True)

    # Infos véhicule
    model = Column(String(120), nullable=False)           # ex: "VW Caddy Maxi"
    license_plate = Column(String(20), nullable=False)    # ex: "GE 123456"
    year = Column(Integer, nullable=True)                 # ex: 2021
    vin = Column(String(32), nullable=True)               # optionnel

    # Capacités
    seats = Column(Integer, nullable=True)                # nb de places assises
    wheelchair_accessible = Column(Boolean, default=False, nullable=False)

    # Suivi administratif
    insurance_expires_at = Column(DateTime(timezone=True), nullable=True)
    inspection_expires_at = Column(DateTime(timezone=True), nullable=True)

    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    company = relationship('Company', back_populates='vehicles')

    @validates('license_plate')
    def validate_license_plate(self, key, plate):
        if not plate or len(plate.strip()) < 3:
            raise ValueError("Numéro de plaque invalide.")
        return plate.strip().upper()

    @property
    def serialize(self):
        return {
            "id": self.id,
            "company_id": self.company_id,
            "model": self.model,
            "license_plate": self.license_plate,
            "year": self.year,
            "vin": self.vin,
            "seats": self.seats,
            "wheelchair_accessible": self.wheelchair_accessible,
            "insurance_expires_at": self.insurance_expires_at.isoformat() if self.insurance_expires_at else None,
            "inspection_expires_at": self.inspection_expires_at.isoformat() if self.inspection_expires_at else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class Driver(db.Model):
    __tablename__ = 'driver'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), unique=True, nullable=False, index=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=False, index=True)
    vehicle_assigned = Column(String(100), nullable=True)  # Nom ou modèle du véhicule
    brand = Column(String(100), nullable=True)  # Marque du véhicule
    license_plate = db.Column(
        StringEncryptedType(
            db.String,
            _encryption_key_str,
            AesEngine,
            'pkcs5'
        ),
        nullable=True
    )
    is_active = Column(Boolean, default=True, nullable=False)  # Statut du chauffeur
    is_available = Column(Boolean, default=True, nullable=False)  # 🔥 Nouveau champ pour disponibilité

    driver_type = db.Column(SQLAlchemyEnum(DriverType), default=DriverType.REGULAR, nullable=False)

    latitude = Column(Float, nullable=True)  # 📍 Localisation en temps réel
    longitude = Column(Float, nullable=True)
    last_position_update = db.Column(db.DateTime(timezone=True), nullable=True)

    driver_photo = Column(Text, nullable=True)
    push_token = Column(String(255), nullable=True, index=True)

    # 🔹 Relations bidirectionnelles
    user = relationship('User', back_populates='driver', passive_deletes=True)
    company = relationship('Company', back_populates='drivers', passive_deletes=True)
    bookings = relationship('Booking', back_populates='driver', passive_deletes=True)
    working_config = db.relationship(
        'DriverWorkingConfig',
        backref='driver',
        uselist=False
    )

    @property
    def serialize(self):
        """
        Sérialisation des données du chauffeur pour le frontend.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.user.username if self.user else "Non spécifié",
            "first_name": getattr(self.user, "first_name", "Non spécifié"),
            "last_name": getattr(self.user, "last_name", "Non spécifié"),
            "phone": getattr(self.user, "phone", "Non spécifié"),
            "photo": self.driver_photo or (getattr(self.user, "profile_image", "/images/default-driver.png")),
            "company_id": self.company_id,
            "company_name": self.company.name if self.company else "Non spécifié",
            "is_active": self.is_active,
            "is_available": self.is_available,  # ✅ Ajout pour savoir si le chauffeur est libre
            "driver_type": self.driver_type.value if self.driver_type else None,
            "vehicle_assigned": self.vehicle_assigned or "Non spécifié",
            "brand": self.brand or "Non spécifié",
            "license_plate": self.license_plate or "Non spécifié",
            "latitude": self.latitude,
            "longitude": self.longitude
            
        }

    @validates('user_id')
    def validate_user_id(self, key, user_id):
        """
        Vérifie que l'utilisateur existe et possède le rôle 'driver'.
        """
        if not user_id or user_id <= 0:
            raise ValueError("L'ID utilisateur est invalide.")
        user = db.session.get(User, user_id)
        if not user:
            raise ValueError("Utilisateur non trouvé.")
        if user.role != UserRole.driver:
            raise ValueError("L'utilisateur associé n'a pas le rôle 'driver'.")
        return user_id

    @validates('company_id')
    def validate_company_id(self, key, company_id):
        """
        Vérifie que l'entreprise associée existe bien.
        """
        if not company_id or company_id <= 0:
            raise ValueError("L'ID de l'entreprise est invalide.")
        company = db.session.get(Company, company_id)
        if not company:
            raise ValueError("Entreprise non trouvée.")
        return company_id

    @validates('vehicle_assigned', 'brand', 'license_plate')
    def validate_vehicle_info(self, key, value):
        """
        Vérifie que les informations du véhicule ne sont pas vides.
        """
        if value is not None and not value.strip():
            raise ValueError(f"Le champ {key} ne peut pas être vide.")
        return value
    
    @property
    def serialize_position(self):
        return {
            "id": self.id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "is_available": self.is_available
        }


    def toggle_availability(self):
        """
        Active ou désactive la disponibilité du chauffeur.
        """
        self.is_available = not self.is_available
        return self.is_available


    def update_location(self, latitude, longitude):
        """
        Met à jour la position du chauffeur en temps réel.
        """
        if latitude is None or longitude is None:
            raise ValueError("Les coordonnées GPS ne peuvent pas être nulles.")
        
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Les coordonnées GPS sont hors limites valides.")

        self.latitude = latitude
        self.longitude = longitude

    
    def deactivate(self):
        """Désactive le chauffeur."""
        self.is_active = False
        self.is_available = False
        db.session.commit()

    def activate(self):
        """Active le chauffeur."""
        self.is_active = True
        self.is_available = True
        db.session.commit()

    def __repr__(self):
        return f"<Driver {self.id} - {self.user.username if self.user else 'N/A'} - {self.company.name if self.company else 'N/A'}>"
    
    def to_dict(self):
        return self.serialize

class DriverWorkingConfig(db.Model):
    """
    Stocke la configuration d'horaire de travail pour un chauffeur :
      - earliest_start : heure min en minutes depuis minuit (ex. 6h = 360)
      - latest_start   : heure max pour débuter la journée (ex. 10h = 600)
      - total_working_minutes : durée totale de travail max (ex. 510 = 8h30)
      - break_duration : durée de la pause (ex. 60 minutes)
      - break_earliest / break_latest : plage horaire min/max où la pause peut démarrer
    """
    __tablename__ = 'driver_working_config'

    id = Column(Integer, primary_key=True)
    
    # Relie chaque config à un chauffeur unique.
    driver_id = Column(
        Integer,
        ForeignKey('driver.id', ondelete="CASCADE"),
        nullable=False,
        unique=True,    # un seul config par chauffeur
        index=True
    )

    # Heure la plus tôt pour démarrer la journée (en minutes depuis minuit)
    earliest_start = Column(Integer, default=360)  # 360 => 6h
    # Heure la plus tard pour démarrer la journée (en minutes)
    latest_start = Column(Integer, default=600)    # 600 => 10h

    # Durée de travail maximum (en minutes), ex : 510 => 8h30
    total_working_minutes = Column(Integer, default=510)

    # Durée de la pause (en minutes), ex: 60 => 1h
    break_duration = Column(Integer, default=60)

    # Plage horaire pour commencer la pause
    break_earliest = Column(Integer, default=600)  # 600 => 10h
    break_latest   = Column(Integer, default=900)  # 900 => 15h

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
        return (f"<DriverWorkingConfig id={self.id} driver_id={self.driver_id} "
                f"start={self.earliest_start}-{self.latest_start} "
                f"maxWork={self.total_working_minutes} "
                f"break={self.break_duration} ({self.break_earliest}-{self.break_latest})>")

    @validates('earliest_start', 'latest_start', 'total_working_minutes', 'break_duration', 'break_earliest', 'break_latest')
    def validate_working_times(self, key, value):
        if value is None or value < 0 or value > 1440:
            raise ValueError(f"'{key}' doit être entre 0 et 1440 (minutes de la journée)")
        return value
    
    def validate_config(self):
        if self.earliest_start >= self.latest_start:
            raise ValueError("earliest_start doit être avant latest_start")
        if self.break_earliest >= self.break_latest:
            raise ValueError("break_earliest doit être avant break_latest")
        if self.break_duration > self.total_working_minutes:
            raise ValueError("La pause ne peut pas excéder le temps de travail total")

    def readable_hours(self):
        def min_to_hour(m): return f"{m//60:02d}:{m%60:02d}"
        return {
            "work_start": f"{min_to_hour(self.earliest_start)} → {min_to_hour(self.latest_start)}",
            "break_window": f"{min_to_hour(self.break_earliest)} → {min_to_hour(self.break_latest)}",
            "pause": f"{self.break_duration} min",
            "total_working_time": f"{self.total_working_minutes} min"
        }

class DriverVacation(db.Model):
    """
    Représente une période d'absence ou de vacances pour un chauffeur.
    """
    __tablename__ = 'driver_vacations'

    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.Integer, db.ForeignKey('driver.id', ondelete="CASCADE"), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    vacation_type = db.Column(db.String(50), default="VACANCES")  # Ex: VACANCES, MALADIE, CONGÉS

    # 🔁 Relation
    driver = db.relationship('Driver', backref='vacations', passive_deletes=True)

    def __repr__(self):
        return (
            f"<DriverVacation id={self.id}, driver_id={self.driver_id}, "
            f"{self.start_date.isoformat()} → {self.end_date.isoformat()}, "
            f"type={self.vacation_type}>"
        )

    @validates('start_date', 'end_date')
    def validate_dates(self, key, value):
        if not value:
            raise ValueError(f"{key} ne peut pas être vide.")
        if not isinstance(value, date):
            raise ValueError(f"{key} doit être une date valide.")
        return value

    def validate_logic(self):
        if self.start_date > self.end_date:
            raise ValueError("La date de début ne peut pas être après la date de fin.")

    def overlaps(self, other_start: date, other_end: date) -> bool:
        """
        Vérifie si cette période chevauche une autre période donnée.
        """
        return self.start_date <= other_end and self.end_date >= other_start

    @property
    def serialize(self):
        return {
            "id": self.id,
            "driver_id": self.driver_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "vacation_type": self.vacation_type
        }

class Client(db.Model):
    __tablename__ = 'client'

    id = Column(Integer, primary_key=True)

    # --- rattachements
    user_id = Column(Integer, ForeignKey('user.id', ondelete="CASCADE"), nullable=False, index=True)

    # propriétaire (entreprise) -> visibilité/ACL
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=True, index=True)

    client_type = Column(SQLEnum(ClientType, name="client_type"), nullable=False, default=ClientType.SELF_SERVICE)

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
    default_billed_to_company_id = Column(Integer, ForeignKey('company.id', ondelete="SET NULL"), nullable=True)
    default_billed_to_contact = Column(String(120), nullable=True)

    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        db.UniqueConstraint('user_id', 'company_id', name='uq_user_company'),
    )

    # 🔁 Relations (désambiguïsées)
    user = relationship('User', back_populates='clients', passive_deletes=True)

    # ⚠️ Deux FKs vers Company: il faut préciser foreign_keys pour éviter l'ambiguïté
    company = relationship(
        'Company',
        back_populates='clients',
        foreign_keys=[company_id],
        passive_deletes=True
    )

    default_billed_to_company = relationship(
        'Company',
        foreign_keys=[default_billed_to_company_id],
        backref='billed_clients'  # optionnel: accès inverse
    )

    bookings = relationship('Booking', back_populates='client', lazy=True)
    payments = relationship('Payment', back_populates='client', lazy=True)

    # ---------------- Validators ---------------- #

    @validates('contact_email')
    def validate_contact_email(self, key, email):
        if self.client_type == ClientType.SELF_SERVICE and not email:
            raise ValueError("L'email est requis pour les clients self-service.")
        if email:
            email = email.strip()
            if '@' not in email:
                raise ValueError("Email invalide.")
        return email

    @validates('billing_address')
    def validate_billing_address(self, key, value):
        if self.company_id and (not value or not value.strip()):
            raise ValueError("L'adresse de facturation est obligatoire pour les clients liés à une entreprise.")
        return value

    @validates('contact_phone', 'gp_phone')
    def validate_phone_numbers(self, key, value):
        if value:
            v = value.strip()
            # autorise +, espace, -, (), mais au moins 6 chiffres
            import re
            digits = re.sub(r'\D', '', v)
            if len(digits) < 6:
                raise ValueError(f"{key} semble invalide.")
            return v
        return value

    @validates('default_billed_to_type')
    def validate_default_billed_to_type(self, key, val):
        val = (val or 'patient').strip().lower()
        if val not in ('patient', 'clinic', 'insurance'):
            raise ValueError("default_billed_to_type invalide (patient|clinic|insurance)")
        # cohérence: si patient => pas de société par défaut
        if val == 'patient':
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
                "billed_to_company": (self.default_billed_to_company.serialize if self.default_billed_to_company else None),
                "billed_to_contact": self.default_billed_to_contact,
            },

            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    # ---------------- Utils ---------------- #

    def toggle_active(self):
        self.is_active = not self.is_active
        return self.is_active

    def is_self_service(self):
        return self.client_type == ClientType.SELF_SERVICE

    def __repr__(self):
        return f"<Client id={self.id}, user_id={self.user_id}, type={self.client_type}, active={self.is_active}>"

class Booking(db.Model):
    __tablename__ = 'booking'

    id = Column(Integer, primary_key=True)
    customer_name = Column(String(100), nullable=False)
    pickup_location = Column(String(200), nullable=False)
    dropoff_location = Column(String(200), nullable=False)
    booking_type = Column(String(200), default='standard')  # standard ou manual
    scheduled_time = Column(DateTime(timezone=False), nullable=True)  # Naive Europe/Zurich
    amount = Column(Float, nullable=False)
    status = db.Column(SQLEnum(BookingStatus), index=True, default=BookingStatus.PENDING, nullable=False)
    user_id = Column(Integer, ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    rejected_by = Column(JSONEncodedList, default=[])
    duration_seconds = db.Column(db.Integer)  # ou duration_minutes
    distance_meters = db.Column(db.Integer)

    client_id = Column(Integer, ForeignKey('client.id', ondelete="CASCADE"), nullable=False, index=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=True, index=True)
    driver_id = Column(Integer, ForeignKey('driver.id', ondelete="SET NULL"), nullable=True, index=True)
    
    is_round_trip = Column(Boolean, default=False)  # 🚗 Aller-retour prévu ?
    is_return = Column(Boolean, default=False)  # ✅ Permet de distinguer un retour
    boarded_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    parent_booking_id = Column(Integer, ForeignKey('booking.id', ondelete="SET NULL"), nullable=True)

    medical_facility = Column(String(200), nullable=True)
    doctor_name = Column(String(200), nullable=True)
    hospital_service = Column(String(100), nullable=True)  # Service médical (ex: cardiologie, pédiatrie, etc.)
    notes_medical = Column(Text, nullable=True)  # Notes médicales supplémentaires
    is_urgent = db.Column(db.Boolean, nullable=False, server_default='0')

    pickup_lat = Column(Float, nullable=True)
    pickup_lon = Column(Float, nullable=True)
    dropoff_lat = Column(Float, nullable=True)
    dropoff_lon = Column(Float, nullable=True)


    created_at = Column(DateTime(timezone=True), default=func.current_timestamp())
    updated_at = Column(DateTime(timezone=True), default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Qui paie cette course ?
    billed_to_type = Column(String(50), default="patient")  # "patient" | "clinic" | "insurance"
    billed_to_company_id = Column(Integer, ForeignKey('company.id', ondelete="SET NULL"), nullable=True)
    billed_to_contact = Column(String(120), nullable=True)  # email ou personne de ref (optionnel)


    # 🔹 Relations
    client = relationship('Client', back_populates='bookings', passive_deletes=True)
    company = relationship('Company', back_populates='bookings', foreign_keys=[company_id], passive_deletes=True)
    driver = relationship('Driver', back_populates='bookings', passive_deletes=True)
    payments = relationship('Payment', back_populates='booking', passive_deletes=True, lazy=True)
    invoice = relationship("Invoice", back_populates="booking", cascade="all, delete-orphan", passive_deletes=True)
    billed_to_company = relationship('Company', foreign_keys=[billed_to_company_id])

    # 🔄 Relation aller-retour
    return_trip = relationship(
        "Booking",
        backref="original_booking",
        remote_side=[id],
        foreign_keys=[parent_booking_id],
        uselist=False
    )

    # Version CORRIGÉE (plus robuste)
    @property
    def customer_full_name(self):
        # Priorité n°1 : Toujours utiliser le nom qui a été explicitement sauvegardé dans la course.
        if self.customer_name:
            return self.customer_name
        
        # Priorité n°2 (fallback) : Si le premier est vide, essayer de le construire depuis la relation.
        if self.client and self.client.user:
            if self.client.user.first_name or self.client.user.last_name:
                return f"{self.client.user.first_name or ''} {self.client.user.last_name or ''}".strip()
            return self.client.user.username
            
        # Fallback final si tout le reste échoue.
        return "Non spécifié"
    
    def get_effective_payer(self):
        """
        Retourne un dict normalisé pour l'adresse/identité du payeur.
        Priorité: billed_to_company si billed_to_type != 'patient',
        sinon le Client lié à la booking.
        """
        if (self.billed_to_type or 'patient') != 'patient' and getattr(self, "billed_to_company", None):
            comp = self.billed_to_company
            return {
                "type": self.billed_to_type,
                "name": comp.name,
                "address": getattr(comp, "address", None),
                "email": getattr(comp, "contact_email", None),
                "phone": getattr(comp, "contact_phone", None),
                "company_id": comp.id
            }

        # Fallback: client relié à la réservation
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
    
    # 🔹 Propriété pour la sérialisation
    @property
    # Sécuriser les relations
    def serialize(self):
            # -- Normalisations temps (depuis DB) --
            # Pas besoin de conversion, on utilise directement le datetime naïf
            scheduled_time = self.scheduled_time
            date_local, time_local = split_date_time_local(scheduled_time)  # ('YYYY-MM-DD','HH:MM')

            created_loc = to_geneva_local(self.created_at) if self.created_at else None
            updated_loc = to_geneva_local(self.updated_at) if self.updated_at else None
            boarded_iso = iso_utc_z(self.boarded_at)
            completed_iso = iso_utc_z(self.completed_at)        
            cli = self.client
            cli_user = getattr(cli, "user", None)

            data = {
                "id": self.id,
                "customer_name": self.customer_name or self.customer_full_name,
                "client_name": self.customer_name or self.customer_full_name,

                "pickup_location": self.pickup_location,
                "dropoff_location": self.dropoff_location,

                # ⬇️ ISO UTC machine (avec 'Z')
                "amount": round(self.amount or 0, 2),
                "scheduled_time": scheduled_time.isoformat() if scheduled_time else None,

                # ⬇️ Affichage local Europe/Zurich pour le front
                "date_formatted": date_local or "Non spécifié",
                "time_formatted": time_local or "Non spécifié",            
                "status": getattr(self.status, "value", "unknown").lower(),

                "client": {
                    "id": getattr(cli, "id", None),
                    "first_name": getattr(cli_user, "first_name", "") if cli_user else "",
                    "last_name": getattr(cli_user, "last_name", "") if cli_user else "",
                    "email": getattr(cli_user, "email", "") if cli_user else "",
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

                # Dates système rendues en local pour l’affichage
                "created_at": created_loc.strftime("%d/%m/%Y %H:%M") if created_loc else "Non spécifié",
                "updated_at": updated_loc.strftime("%d/%m/%Y %H:%M") if updated_loc else "Non spécifié",

                "rejected_by": self.rejected_by,

                "is_round_trip": bool(self.is_round_trip),
                "is_return": bool(self.is_return),
                "has_return": self.return_trip is not None,

                # Événements de course (ISO UTC)
                "boarded_at": iso_utc_z(to_utc_from_db(self.boarded_at)) if self.boarded_at else None,
                "completed_at": iso_utc_z(to_utc_from_db(self.completed_at)) if self.completed_at else None,

                # garde si tu as bien ce champ côté modèle/service
                "duree_minutes": (
                    int((self.completed_at - self.boarded_at).total_seconds() // 60)
                    if (self.completed_at and self.boarded_at) else None
                ),
                "duration_in_minutes": getattr(self, "duration_in_minutes", None),

                "billing": {
                    "billed_to_type": self.billed_to_type or "patient",
                    "billed_to_company": self.billed_to_company.serialize if self.billed_to_company else None,
                    "billed_to_contact": self.billed_to_contact
                },
                "patient_name": self.customer_name,  # pour clarté front
            }
            return data

    
    @staticmethod
    def auto_geocode_if_needed(booking):
        """
        NO-OP: désactivé.
        On ne géocode plus côté serveur.
        Les coordonnées doivent être fournies par le frontend
        (autocomplete, favoris, etc.) ou sinon gérées par le fallback
        dans enrich_booking_coords (coords entreprise ou Genève).
        """
        return False


    
    @validates('user_id')
    def validate_user_id(self, key, user_id):
        state = inspect(self)
        if user_id is None and state.deleted:
            return user_id
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("L'ID utilisateur doit être un entier positif.")
        return user_id
    
    @validates('is_return')
    def validate_is_return(self, key, val):
        return bool(val)  # force le type

    
    @validates('amount')
    def validate_amount(self, key, amount):
        if amount <= 0:
            raise ValueError("Le montant doit être supérieur à 0")
        return round(amount, 2)

    @validates('scheduled_time')
    def validate_scheduled_time(self, key, scheduled_time):
        st = parse_local_naive(scheduled_time)  # ← normalisation en naive Europe/Zurich
        if st and st < now_local():
            raise ValueError("Heure prévue dans le passé.")
        return st

    @validates('customer_name')
    def validate_customer_name(self, key, customer_name):
        if not customer_name or len(customer_name.strip()) == 0:
            raise ValueError("Le nom du client ne peut pas être vide")
        if len(customer_name) > 100:
            raise ValueError("Le nom du client ne peut pas dépasser 100 caractères")
        return customer_name

    @validates('pickup_location', 'dropoff_location')
    def validate_location(self, key, location):
        if not location or len(location.strip()) == 0:
            raise ValueError(f"{key} ne peut pas être vide")
        if len(location) > 200:
            raise ValueError(f"{key} ne peut pas dépasser 200 caractères")
        return location

    @validates('status')
    def validate_status(self, key, status):
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
    def validate_driver_id(self, key, value):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ValueError("driver_id doit être un entier positif ou null")
        return value
    
    
    def is_future(self) -> bool:
        return self.scheduled_time > now_local()

    def update_status(self, new_status):
        """Met à jour le statut de la réservation."""
        if not isinstance(new_status, BookingStatus):
            raise ValueError("Statut invalide.")
        self.status = new_status

    @property
    def duration_in_minutes(self):
        if self.boarded_at and self.completed_at:
            delta = self.completed_at - self.boarded_at
            return int(delta.total_seconds() // 60)
        return None

    def to_dict(self):
        return self.serialize
    
    # Dans models.py, dans la classe Booking
    def is_assignable(self):
        # On importe la fonction to_utc au début de votre fichier models.py
        # from services.dispatch import to_utc
        return (
            self.status in [BookingStatus.PENDING, BookingStatus.ACCEPTED]
            and self.scheduled_time > now_local()
        )

    def assign_driver(self, driver_id: int):
        """
        Attribue un chauffeur à la réservation si c’est possible.
        Change le statut à 'ASSIGNED' si tout est ok.
        """
        if not self.is_assignable():
            raise ValueError("La réservation ne peut pas être attribuée actuellement.")
        
        if self.driver_id == driver_id:
            return  # Ne rien faire si déjà assigné au même chauffeur
        
        self.driver_id = driver_id
        self.status = BookingStatus.ASSIGNED
        self.updated_at = datetime.now(timezone.utc)

    def cancel_booking(self):
        if self.status not in [BookingStatus.ASSIGNED, BookingStatus.ACCEPTED]:
            raise ValueError("Seules les réservations en cours peuvent être annulées.")
        self.status = BookingStatus.CANCELED
        self.updated_at = datetime.now(timezone.utc)

    @validates('billed_to_type')
    def _v_billed_to_type(self, key, value):
        v = (value or 'patient').lower().strip()
        if v not in ('patient', 'clinic', 'insurance'):
            raise ValueError("billed_to_type invalide (patient|clinic|insurance)")
        return v

    @validates('billed_to_company_id')
    def _v_billed_to_company_id(self, key, value):
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("billed_to_company_id doit être un entier positif ou NULL")
        # Si le type courant est 'patient', on *renvoie* None (pas d'assignation !)
        if (self.billed_to_type or 'patient') == 'patient':
            return None
        return value

    # --- verrou de cohérence avant INSERT/UPDATE (pas de récursion ici) ---
    def _enforce_billing_exclusive(mapper, connection, target: "Booking"):
        btype = (target.billed_to_type or 'patient').lower()
        if btype == 'patient':
            # patient paie → pas de société
            target.billed_to_company_id = None
        else:
            # clinic/insurance → société obligatoire
            if not target.billed_to_company_id:
                raise ValueError("billed_to_company_id est obligatoire si billed_to_type n'est pas 'patient'")

class Payment(db.Model):
    __tablename__ = 'payment'

    id = Column(Integer, primary_key=True)
    amount = Column(Float, nullable=False)
    date = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    method = Column(String(50), nullable=False)
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    client_id = Column(Integer, ForeignKey('client.id', ondelete="CASCADE"), nullable=False)
    booking_id = Column(Integer, ForeignKey('booking.id', ondelete="CASCADE"), nullable=False)

    # Relations
    client = relationship('Client', back_populates='payments', passive_deletes=True)
    booking = relationship('Booking', back_populates='payments', passive_deletes=True)

    @property
    def serialize(self):
        """Sérialise les infos de Payment + un extrait de la réservation."""
        return {
            "id": self.id,
            "amount": self.amount,
            "method": self.method,
            "status": self.status.value,
            "date": self.date.isoformat() if self.date else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "client_id": self.client_id,
            "booking_id": self.booking_id,
            "booking_info": {
                "pickup_location": self.booking.pickup_location if self.booking else None,
                "dropoff_location": self.booking.dropoff_location if self.booking else None,
                "scheduled_time": (iso_utc_z(to_utc_from_db(self.booking.scheduled_time))),
                "scheduled_time": (
                                    self.booking.scheduled_time.isoformat()
                                    if (self.booking and self.booking.scheduled_time) else None
                                )
                            } 
            }
    @validates('user_id')
    def validate_user_id(self, key, user_id):
        state = inspect(self)
        # Si l'objet est en cours de suppression et user_id est None, on autorise cette valeur
        if user_id is None and state.deleted:
            return user_id
        if not user_id or user_id <= 0:
            raise ValueError("L'ID utilisateur pour Payment doit être un entier positif.")
        return user_id

    @validates('amount')
    def validate_amount(self, key, amount):
        """Vérifie que le montant est positif."""
        if amount <= 0:
            raise ValueError("Le montant doit être supérieur à 0")
        return round(amount, 2)  # ✅ Arrondi pour éviter les erreurs

    # 🔹 Validation de la méthode de paiement
    @validates('method')
    def validate_method(self, key, method):
        """Vérifie que la méthode de paiement est valide."""
        allowed_methods = ['credit_card', 'paypal', 'bank_transfer', 'cash']
        if method not in allowed_methods:
            raise ValueError(f"Méthode de paiement invalide. Méthodes autorisées : {', '.join(allowed_methods)}")
        return method

    # 🔹 Validation de la date
    @validates('date')
    def validate_date(self, key, date):
        """Vérifie que la date du paiement n’est pas dans le futur."""
        if date is None:
            raise ValueError("La date de paiement ne peut pas être nulle")
        if date > datetime.now(timezone.utc):
            raise ValueError("La date de paiement ne peut pas être dans le futur")
        return date


    # 🔹 Validation du statut
    @validates('status')
    def validate_status(self, key, status):
        """Vérifie que le statut est valide."""
        if isinstance(status, str):
            status = status.lower()
            if status not in PaymentStatus.choices():
                raise ValueError(f"Statut de paiement invalide : {status}. Doit être l'un de {PaymentStatus.choices()}")

        if not isinstance(status, PaymentStatus):
            raise ValueError(f"Statut invalide : {status}. Doit être un PaymentStatus valide.")
        return status
    

    # 🔹 Méthode pour changer le statut de paiement
    def update_status(self, new_status):
        """Met à jour le statut du paiement."""
        if not isinstance(new_status, PaymentStatus):
            raise ValueError("Statut de paiement invalide.")
        self.status = new_status
    
class Invoice(db.Model):
    __tablename__ = "invoice"

    id = Column(Integer, primary_key=True)
    reference = Column(String(32), unique=True, nullable=False, index=True, default=lambda: Invoice.generate_reference())
    amount = Column(Float, nullable=False)
    user_id = Column(Integer, ForeignKey('user.id', ondelete="CASCADE", name="fk_invoice_user"), nullable=False, index=True)
    booking_id = Column(Integer, ForeignKey("booking.id", ondelete="SET NULL", name="fk_invoice_booking"), nullable=True, index=True)
    company_id = Column(Integer, ForeignKey("company.id", ondelete="CASCADE", name="fk_invoice_company"), nullable=False, index=True)
    details = Column(Text, nullable=True)  # Détails libres (motif, référence externe...)
    pdf_url = Column(String(255), nullable=True)  # Lien vers PDF généré (optionnel)
    due_date = Column(DateTime(timezone=True), nullable=True)
    paid_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)    
    status = Column(SQLEnum(InvoiceStatus, name="invoice_status"), default=InvoiceStatus.UNPAID, nullable=False)

    # Relations
    user = relationship("User", back_populates="invoices", passive_deletes=True)
    booking = relationship("Booking", back_populates="invoice", passive_deletes=True)
    company = relationship("Company", backref="invoices", passive_deletes=True)

    # ========== Sérialisation enrichie ==========
    @property
    def serialize(self):
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
            "status": self.status.value if self.status else "unknown",
            "status_human": self.status_human,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
        }

    @property
    def status_human(self):
        status_map = {
            "unpaid": "Non payée",
            "paid": "Payée",
            "canceled": "Annulée"
        }
        return status_map.get(self.status.value, "Inconnu")

    # ========== Génération référence unique ==========
    @staticmethod
    def generate_reference():
        import random, string
        prefix = "FCT"
        suffix = ''.join(random.choices(string.digits, k=7))
        return f"{prefix}-{suffix}"

    # ========== Validations ==========
    @validates('user_id')
    def validate_user_id(self, key, user_id):
        if user_id is None or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("ID utilisateur invalide")
        return user_id

    @validates('amount')
    def validate_amount(self, key, amount):
        if amount is None or amount <= 0:
            raise ValueError("Le montant doit être supérieur à 0")
        return round(amount, 2)

    @validates('booking_id')
    def validate_booking_id(self, key, booking_id):
        if booking_id is not None and booking_id <= 0:
            raise ValueError("ID de réservation invalide")
        return booking_id

    @validates('company_id')
    def validate_company_id(self, key, company_id):
        if not isinstance(company_id, int) or company_id <= 0:
            raise ValueError("ID de l'entreprise invalide")
        return company_id

    @validates('reference')
    def validate_reference(self, key, ref):
        if not ref or len(ref) > 32:
            raise ValueError("Référence de facture invalide")
        return ref

    @validates('status')
    def validate_status(self, key, status):
        if isinstance(status, str):
            status = status.lower()
            if status not in InvoiceStatus.choices():
                raise ValueError(f"Statut de facture invalide : {status}")
        if not isinstance(status, InvoiceStatus):
            raise ValueError("Statut de facture invalide.")
        return status

    # ========== Actions Métiers ==========

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
        return f"<Invoice {self.reference} | {self.amount:.2f} CHF | {self.status.value}>"

class Message(db.Model):
    __tablename__ = "message"

    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id', ondelete="CASCADE"), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="SET NULL"), nullable=True)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="SET NULL"), nullable=True)
    sender_role = db.Column(db.String(20), nullable=False)  # 'driver' ou 'company'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    is_read = db.Column(db.Boolean, default=False)

    __table_args__ = (
        CheckConstraint("sender_role IN ('driver', 'company')", name='check_sender_role_valid'),
    )

    # Relations
    sender = db.relationship('User', foreign_keys=[sender_id], lazy='joined')
    receiver = db.relationship('User', foreign_keys=[receiver_id], lazy='joined')
    company = db.relationship('Company', lazy='joined')

    def __repr__(self):
        return f"<Message {self.id} from {self.sender_role} ({self.sender_id})>"

    @property
    def serialize(self):
        return {
            "id": self.id,
            "company_id": self.company_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "sender_role": self.sender_role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_read": self.is_read
        }

    @validates('sender_role')
    def validate_sender_role(self, key, value):
        if value not in ['driver', 'company']:
            raise ValueError("Le rôle de l'expéditeur doit être 'driver' ou 'company'")
        return value

    @validates('content')
    def validate_content(self, key, content):
        if not content or not content.strip():
            raise ValueError("Le contenu du message ne peut pas être vide")
        return content

class FavoritePlace(db.Model):
    __tablename__ = "favorite_place"
    __table_args__ = (
        # Évite les doublons d’adresse au sein d’une même entreprise
        db.UniqueConstraint("company_id", "address", name="uq_fav_company_address"),
        # Accélère la recherche par libellé pour une entreprise
        db.Index("ix_fav_company_label", "company_id", "label"),
        # Accélère les requêtes par coordonnées (proches)
        db.Index("ix_fav_company_coords", "company_id", "lat", "lon"),
        # Verrouille les bornes géographiques
        db.CheckConstraint("lat BETWEEN -90 AND 90", name="chk_fav_lat"),
        db.CheckConstraint("lon BETWEEN -180 AND 180", name="chk_fav_lon"),
    )

    id = db.Column(db.Integer, primary_key=True)

    # Entreprise propriétaire du favori
    company_id = db.Column(
        db.Integer,
        db.ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Libellé affiché (ex. "HUG – Urgences")
    label = db.Column(db.String(200), nullable=False)

    # Adresse canonique (ex. "Rue Gabrielle-Perret-Gentil 4, 1205 Genève")
    address = db.Column(db.String(255), nullable=False)

    # Coordonnées GPS
    lat = db.Column(db.Float, nullable=False)
    lon = db.Column(db.Float, nullable=False)

    # Tags libres (ex. "hospital;emergency")
    tags = db.Column(db.String(200))

    # Timestamps (pratiques pour audit/tri)
    created_at = db.Column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
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
        # Normalisation ultra-simple (évite les espaces doubles, trim)
        s = (s or "").strip()
        s = " ".join(s.split())
        return s

    @validates("label")
    def _v_label(self, key, value):
        value = self._norm_text(value)
        if not value:
            raise ValueError("Le champ 'label' ne peut pas être vide.")
        return value

    @validates("address")
    def _v_address(self, key, value):
        value = self._norm_address(value)
        if not value:
            raise ValueError("Le champ 'address' ne peut pas être vide.")
        if len(value) > 255:
            raise ValueError("Le champ 'address' dépasse 255 caractères.")
        return value

    @validates("lat")
    def _v_lat(self, key, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Latitude invalide.")
        if not (-90.0 <= v <= 90.0):
            raise ValueError("Latitude hors bornes [-90; 90].")
        return v

    @validates("lon")
    def _v_lon(self, key, value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError("Longitude invalide.")
        if not (-180.0 <= v <= 180.0):
            raise ValueError("Longitude hors bornes [-180; 180].")
        return v

    @validates("tags")
    def _v_tags(self, key, value):
        # Trim simple; tu peux forcer un format "tag1;tag2"
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
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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

# --- ÉTABLISSEMENTS & SERVICES MÉDICAUX ---

class MedicalEstablishment(db.Model):
    __tablename__ = "medical_establishment"
    __table_args__ = (
        db.UniqueConstraint("name", name="uq_med_estab_name"),
        db.UniqueConstraint("address", name="uq_med_estab_address"),
        db.CheckConstraint("lat BETWEEN -90 AND 90", name="chk_med_estab_lat"),
        db.CheckConstraint("lon BETWEEN -180 AND 180", name="chk_med_estab_lon"),
    )

    id = db.Column(db.Integer, primary_key=True)
    # type générique: "hospital", "clinic", "ems", ...
    type = db.Column(db.String(50), nullable=False, default="hospital")

    name = db.Column(db.String(200), nullable=False)           # "HUG"
    display_name = db.Column(db.String(255), nullable=False)   # "HUG - Hôpitaux Universitaires de Genève"
    address = db.Column(db.String(255), nullable=False)        # adresse canonique
    lat = db.Column(db.Float, nullable=False)
    lon = db.Column(db.Float, nullable=False)

    # Alias de recherche: "hug;hôpital cantonal;hopital geneve"
    aliases = db.Column(db.String(500), nullable=True)
    active = db.Column(db.Boolean, nullable=False, default=True)

    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    services = db.relationship("MedicalService", backref="establishment", cascade="all, delete-orphan", passive_deletes=True)

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

# models.py (complément dans MedicalService)
class MedicalService(db.Model):
    __tablename__ = "medical_service"
    __table_args__ = (
        db.UniqueConstraint("establishment_id", "name", name="uq_med_service_per_estab"),
        db.CheckConstraint("lat IS NULL OR (lat BETWEEN -90 AND 90)", name="chk_med_service_lat"),
        db.CheckConstraint("lon IS NULL OR (lon BETWEEN -180 AND 180)", name="chk_med_service_lon"),
    )

    id = db.Column(db.Integer, primary_key=True)
    establishment_id = db.Column(db.Integer, db.ForeignKey("medical_establishment.id", ondelete="CASCADE"), nullable=False, index=True)

    # "Service", "Département", "Unité", "Laboratoire", "Groupe"
    category = db.Column(db.String(50), nullable=False, default="Service")
    name = db.Column(db.String(200), nullable=False)
    slug = db.Column(db.String(200))

    # --- Nouveaux champs de localisation / contact ---
    address_line = db.Column(db.String(255))   # ex. "Rue Gabrielle-Perret-Gentil 4"
    postcode = db.Column(db.String(16))
    city = db.Column(db.String(100))
    building = db.Column(db.String(120))       # ex. "Bât Prévost"
    floor = db.Column(db.String(60))           # ex. "étage P" / "3e étage"
    site_note = db.Column(db.String(255))      # ex. "Maternité", "Hôpital des enfants", etc.
    phone = db.Column(db.String(40))
    email = db.Column(db.String(120))

    lat = db.Column(db.Float)                  # optionnel si tu veux pointer précis
    lon = db.Column(db.Float)

    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

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

# =========================
#  Dispatch / Temps Réel
# =========================

class DispatchRun(db.Model):
    __tablename__ = "dispatch_run"
    __table_args__ = (
        UniqueConstraint('company_id', 'day', name='uq_dispatch_run_company_day'),
        Index('ix_dispatch_run_company_day', 'company_id', 'day'),
    )

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=False, index=True)
    # stocké en date locale du business (Europe/Zurich) mais persisté comme DATE
    day = Column(db.Date, nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending|running|completed|failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    config = Column(db.JSON, nullable=True)   # configuration utilisée (flags UI)
    metrics = Column(db.JSON, nullable=True)  # KPI calcul run

    company = relationship('Company', backref='dispatch_runs', passive_deletes=True)
    assignments = relationship('Assignment', back_populates='dispatch_run', cascade="all, delete-orphan", passive_deletes=True)

    def mark_started(self):
        self.status = "running"
        self.started_at = datetime.now(timezone.utc)

    def mark_completed(self, metrics: dict | None = None):
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc)
        if metrics:
            self.metrics = (self.metrics or {}) | dict(metrics)

class Assignment(db.Model):
    __tablename__ = "assignment"
    __table_args__ = (
        UniqueConstraint('dispatch_run_id', 'booking_id', name='uq_assignment_run_booking'),
        Index('ix_assignment_driver_status', 'driver_id', 'status'),
    )

    id = Column(Integer, primary_key=True)
    dispatch_run_id = Column(Integer, ForeignKey('dispatch_run.id', ondelete="CASCADE"), nullable=False, index=True)
    booking_id = Column(Integer, ForeignKey('booking.id', ondelete="CASCADE"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey('driver.id', ondelete="SET NULL"), nullable=True, index=True)

    status = Column(SQLEnum(AssignmentStatus, name="assignment_status"), nullable=False, default=AssignmentStatus.SCHEDULED)

    # Planifié (plan) & réel (terrain)
    planned_pickup_at = Column(DateTime(timezone=True), nullable=True)
    planned_dropoff_at = Column(DateTime(timezone=True), nullable=True)
    actual_pickup_at = Column(DateTime(timezone=True), nullable=True)
    actual_dropoff_at = Column(DateTime(timezone=True), nullable=True)

    # ETA + retard estimé (secondes)
    eta_pickup_at = Column(DateTime(timezone=True), nullable=True)
    eta_dropoff_at = Column(DateTime(timezone=True), nullable=True)
    delay_seconds = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    # Relations
    dispatch_run = relationship('DispatchRun', back_populates='assignments', passive_deletes=True)
    booking = relationship('Booking', backref='assignments', passive_deletes=True)
    driver = relationship('Driver', backref='assignments', passive_deletes=True)

class DriverStatus(db.Model):
    __tablename__ = "driver_status"
    __table_args__ = (
        Index('ix_driver_status_state_nextfree', 'state', 'next_free_at'),
    )

    id = Column(Integer, primary_key=True)
    driver_id = Column(Integer, ForeignKey('driver.id', ondelete="CASCADE"), nullable=False, unique=True, index=True)
    state = Column(String(20), nullable=False, default="available")  # available|busy|offline
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)  # degrés
    speed = Column(Float, nullable=True)    # m/s
    next_free_at = Column(DateTime(timezone=True), nullable=True)
    current_assignment_id = Column(Integer, ForeignKey('assignment.id', ondelete="SET NULL"), nullable=True)
    last_update = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    driver = relationship('Driver', backref='status', uselist=False, passive_deletes=True)
    current_assignment = relationship('Assignment', passive_deletes=True)


class RealtimeEvent(db.Model):
    __tablename__ = "realtime_event"
    __table_args__ = (
        Index('idx_realtime_event_company_type_time', 'company_id', 'event_type', 'timestamp'),
    )

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('company.id', ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)    # location_update | status_change | assignment_delta | delay_detected | …
    entity_type = Column(String(20), nullable=False)   # driver | booking | assignment
    entity_id = Column(Integer, nullable=False)
    data = Column(db.JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    company = relationship('Company', passive_deletes=True)