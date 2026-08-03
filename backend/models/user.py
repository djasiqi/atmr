# models/user.py
# pyright: reportUnnecessaryTypeIgnoreComment=false
"""Model User - Gestion des utilisateurs (tous rôles).
Extrait depuis models.py (lignes 249-418).
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import cast

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override
from werkzeug.security import (  # pyright: ignore[reportMissingImports]
    check_password_hash,
    generate_password_hash,
)

from ext import db

from .base import _coerce_enum, _iso
from .enums import GenderEnum, InstitutionRole, UserRole

logger = logging.getLogger(__name__)

ADDRESS_MAX_LENGTH = 200


class User(db.Model):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id = Column(
        String(36),
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )
    username = Column(String(100), nullable=True, index=True)
    first_name: Mapped[str] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str] = mapped_column(String(100), nullable=True)
    email = Column(String(255), nullable=True, index=True)

    # ↓ Champs présents pour tous les rôles (client, driver, etc.)
    phone: Mapped[str] = mapped_column(String(255), nullable=True)
    address: Mapped[str] = mapped_column(String(200), nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[GenderEnum] = mapped_column(
        SAEnum(GenderEnum, name="gender"), nullable=True
    )
    profile_image: Mapped[str] = mapped_column(String(255), nullable=True)

    # 🔔 Token push (Expo/FCM/APNs via Expo) pour les comptes entreprise (dispatch)
    # Utilisé par backend/services/events/fanout.py (_send_push_to_company)
    push_token: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )
    # 🔔 Mode discret push : "detailed" (nom client sur lockscreen) | "discreet" (pas de nom)
    push_privacy_mode: Mapped[str | None] = mapped_column(String(20), nullable=True)

    password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SAEnum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.CLIENT,
        server_default=UserRole.CLIENT.value,
    )

    reset_token = Column(String(100), unique=True, nullable=True)
    zip_code: Mapped[str] = mapped_column(String(10), nullable=True)
    city: Mapped[str] = mapped_column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    force_password_change = Column(Boolean, default=False, nullable=False)
    # Lot 0 SEC-02: invalide tous les access tokens JWT après changement de mot de passe
    token_version: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    # ✅ S3: Date d'expiration du mot de passe (optionnel)
    password_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # ✅ Security V2: TOTP 2FA
    totp_secret_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    totp_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )
    totp_enabled_at = Column(DateTime(timezone=True), nullable=True)
    recovery_codes_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    recovery_codes_remaining: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )

    # ✅ D2: Colonnes chiffrées (stockage)
    phone_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    email_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_name_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_name_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    address_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    encryption_migrated = Column(Boolean, default=False, nullable=False)

    # ✅ Institution: Support multi-tenant institutionnel
    # Un user peut appartenir à une institution (clinique/EMS/IMAD/hôpital)
    # avec un rôle spécifique au sein de cette institution
    institution_id: Mapped[int | None] = mapped_column(
        Integer,
        db.ForeignKey("institutions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    institution_role: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # institution_admin, institution_requester, institution_reader, institution_billing

    # Fonction / metier (descriptif, organisationnel) — independant du role LIRIE.
    # Aucune permission attachee ; sert aux exports, audits et statistiques.
    job_title: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # ✅ Invitation par email pour institution users
    account_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True, default=None
    )  # None (legacy/active), "pending_activation", "invited", "active", "disabled"
    invite_token_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )  # sha256 du token d'invitation
    invite_expires_at = Column(DateTime(timezone=True), nullable=True)
    invite_sent_at = Column(DateTime(timezone=True), nullable=True)

    # Institution identity management
    authentication_method: Mapped[str | None] = mapped_column(
        String(20), nullable=True, default="email", server_default="email"
    )  # email, username, sso, ldap
    temporary_password_created_at = Column(DateTime(timezone=True), nullable=True)
    last_password_reset_at = Column(DateTime(timezone=True), nullable=True)
    temp_password_generation_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    first_login_completed_at = Column(DateTime(timezone=True), nullable=True)
    disabled_at = Column(DateTime(timezone=True), nullable=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)

    # CP-PR1 : classification d'origine des données (défaut unknown)
    data_origin: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="unknown"
    )
    data_origin_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_origin_confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    classified_at = Column(DateTime(timezone=True), nullable=True)
    classified_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    classification_evidence_json = Column(JSONB, nullable=True)

    # ✅ Ajout de l'index sur `public_id` pour optimiser les recherches
    __table_args__ = (
        Index("idx_public_id", "public_id"),
        Index("idx_user_institution_id", "institution_id"),
        Index("idx_user_data_origin", "data_origin"),
    )

    # ✅ Relations bidirectionnelles avec suppression en cascade
    clients = relationship(
        "Client", back_populates="user", cascade="all, delete-orphan"
    )
    driver = relationship(
        "Driver",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    company = relationship(
        "Company",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Company.user_id",
    )
    # ✅ Institution: Relation vers Institution (Many-to-One)
    institution = relationship(
        "Institution",
        back_populates="users",
        foreign_keys=[institution_id],
    )

    # 🔒 Gestion des mots de passe
    def set_password(self, password, force_change=False):
        """✅ S3: Définit un nouveau mot de passe et l'ajoute à l'historique.

        Args:
            password: Nouveau mot de passe en clair
            force_change: Si True, force le changement au prochain login
        """
        # Sauvegarder l'ancien hash dans l'historique avant de le changer
        old_password_hash = getattr(self, "password", None)
        if old_password_hash and hasattr(self, "id") and self.id:
            try:
                from security.password_history import PasswordHistoryService

                PasswordHistoryService.add_password_to_history(
                    self.id, old_password_hash
                )
            except Exception as e:
                logger.warning("[User] ⚠️ Erreur lors de l'ajout à l'historique: %s", e)
                # Ne pas bloquer le changement de mot de passe si l'historique échoue

        # Générer le nouveau hash
        self.password = generate_password_hash(password)
        self.force_password_change = force_change

        # ✅ S3: Mettre à jour la date d'expiration si configuré
        password_expiration_days = int(os.getenv("PASSWORD_EXPIRATION_DAYS", "0"))
        if password_expiration_days > 0:
            self.password_expires_at = datetime.now(UTC) + timedelta(
                days=password_expiration_days
            )

    def check_password(self, password: str) -> bool:
        # Récupère la valeur runtime (qui sera bien une string en pratique)
        pw_any = getattr(self, "password", "")  # évite les warnings d'attr
        if isinstance(pw_any, (bytes, bytearray)):
            pw_str = pw_any.decode("utf-8", "ignore")
        else:
            pw_str = cast("str", pw_any or "")
        return check_password_hash(pw_str, password)

    # Validation du téléphone

    @validates("phone")
    def validate_phone(self, _key, phone):
        # Accepter None ou chaîne vide
        if phone is None:
            return None
        if not isinstance(phone, str):
            return None
        phone = phone.strip()
        if phone == "":
            return None
        # Validation du format si non vide
        if not re.match(r"^\+?\d{7,15}$", phone):
            msg = (
                "Numéro de téléphone invalide. "
                "Doit contenir 7 à 15 chiffres avec option '+'."
            )
            raise ValueError(msg)
        return phone

    # Validation de la date de naissance

    @validates("birth_date")
    def validate_birth_date(self, _key, birth_date):
        """Vérifie que la date de naissance est valide et raisonnable."""
        if birth_date and birth_date > date.today():
            msg = "La date de naissance ne peut pas être dans le futur."
            raise ValueError(msg)
        return birth_date

    # Validation de l'adresse
    @validates("address")
    def validate_address(self, _key, address):
        if address is not None:  # Vérifie si la valeur n'est pas None
            if address.strip() == "":
                msg = "L'adresse ne peut pas être vide."
                raise ValueError(msg)
            if len(address) > ADDRESS_MAX_LENGTH:
                msg = f"L'adresse ne peut pas dépasser {ADDRESS_MAX_LENGTH} caractères."
                raise ValueError(msg)
        return address

    @validates("first_name", "last_name")
    def validate_name(self, _key, name):
        if name is not None and len(name.strip()) == 0:
            msg = f"Le champ {_key} ne peut pas être vide."
            raise ValueError(msg)
        return name

    # 📌 Vérification du genre
    @validates("gender")
    def validate_gender(self, _key, gender_value):
        """Valide/convertit la valeur vers GenderEnum.
        Évite d'utiliser le nom 'gender' (collision avec l'attribut mappé).
        """
        if gender_value is None:
            return None
        try:
            return _coerce_enum(gender_value, GenderEnum)
        except (ValueError, KeyError):
            msg = "Genre invalide."
            raise ValueError(msg) from None

    @validates("role")
    def validate_role(self, _key, role_value):
        """Coerce str → UserRole, évite d'évaluer un Column en bool."""
        try:
            return _coerce_enum(role_value, UserRole)
        except (ValueError, KeyError):
            msg = "Invalid role value. Allowed values: admin, client, driver, company, institution."
            raise ValueError(msg) from None

    @validates("institution_role")
    def validate_institution_role(self, _key, role_value):
        """Valide le rôle institution si fourni."""
        if role_value is None:
            return None
        valid_roles = [e.value for e in InstitutionRole]
        if role_value not in valid_roles:
            msg = f"Invalid institution_role. Allowed values: {', '.join(valid_roles)}"
            raise ValueError(msg)
        return role_value

    @validates("email")
    def validate_email(self, _key, email):
        """Valide le format si fourni.
        ⚠️ La règle 'self-service => email requis' est déjà appliquée
        dans Client.validate_contact_email.
        On évite ici toute logique cross-model
        (et donc les tests sur self.clients / self.role).
        """
        if email is None or email.strip() == "":
            return None
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email.strip()):
            msg = "Format d'email invalide."
            raise ValueError(msg)
        return email.strip()

    # ✅ D2: Propriétés hybrides pour chiffrement/déchiffrement automatique
    @hybrid_property
    def phone_secure(self) -> str | None:  # type: ignore[no-redef]
        """Récupère le téléphone déchiffré."""
        try:
            from security.crypto import get_encryption_service

            # Vérifier si migration effectuée et données chiffrées
            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "phone_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception as e:
                    logger.error("[D2] Erreur déchiffrement phone: %s", e)
                    return None
            # Fallback sur ancienne colonne (migration progressive)
            return getattr(self, "phone", None)
        except ImportError:
            return getattr(self, "phone", None)

    @phone_secure.setter  # type: ignore[no-redef]
    def phone_secure(self, value: str | None):
        """Chiffre et stocke le téléphone."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.phone_encrypted = get_encryption_service().encrypt_field(value)
                self.encryption_migrated = True
                # Garder l'ancienne colonne vide (dépréciée)
                self.phone = None  # type: ignore[assignment]
            else:
                self.phone_encrypted = None
                self.phone = None  # type: ignore[assignment]
        except ImportError:
            # Fallback si le service n'est pas disponible
            self.phone = value  # type: ignore[assignment]

    @hybrid_property
    def email_secure(self) -> str | None:  # type: ignore[no-redef]
        """Récupère l'email déchiffré."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "email_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception:
                    return None
            return cast(str | None, getattr(self, "email", None))
        except ImportError:
            return cast(str | None, getattr(self, "email", None))

    @email_secure.setter  # type: ignore[no-redef]
    def email_secure(self, value: str | None):
        """Chiffre et stocke l'email."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.email_encrypted = get_encryption_service().encrypt_field(value)
                self.encryption_migrated = True
            else:
                self.email_encrypted = None
        except ImportError:
            self.email = value

    @hybrid_property
    def first_name_secure(  # type: ignore[no-redef]
        self,
    ) -> str | None:
        """Récupère le prénom déchiffré."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "first_name_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception:
                    return None
            return cast(str | None, getattr(self, "first_name", None))
        except ImportError:
            return cast(str | None, getattr(self, "first_name", None))

    @first_name_secure.setter  # type: ignore[no-redef]
    def first_name_secure(self, value: str | None):
        """Chiffre et stocke le prénom."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.first_name_encrypted = get_encryption_service().encrypt_field(
                    value
                )
                self.encryption_migrated = True
            else:
                self.first_name_encrypted = None
        except ImportError:
            self.first_name = value  # type: ignore[assignment]

    @hybrid_property
    def last_name_secure(self) -> str | None:  # type: ignore[no-redef]
        """Récupère le nom déchiffré."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "last_name_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception:
                    return None
            return cast(str | None, getattr(self, "last_name", None))
        except ImportError:
            return cast(str | None, getattr(self, "last_name", None))

    @last_name_secure.setter  # type: ignore[no-redef]
    def last_name_secure(self, value: str | None):
        """Chiffre et stocke le nom."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.last_name_encrypted = get_encryption_service().encrypt_field(value)
                self.encryption_migrated = True
            else:
                self.last_name_encrypted = None
        except ImportError:
            self.last_name = value  # type: ignore[assignment]

    @hybrid_property
    def address_secure(self) -> str | None:  # type: ignore[no-redef]
        """Récupère l'adresse déchiffrée."""
        try:
            from security.crypto import get_encryption_service

            is_migrated = bool(getattr(self, "encryption_migrated", False))
            encrypted_val = getattr(self, "address_encrypted", None)
            if is_migrated and encrypted_val:
                try:
                    return get_encryption_service().decrypt_field(encrypted_val)
                except Exception:
                    return None
            return cast(str | None, getattr(self, "address", None))
        except ImportError:
            return cast(str | None, getattr(self, "address", None))

    @address_secure.setter  # type: ignore[no-redef]
    def address_secure(self, value: str | None):
        """Chiffre et stocke l'adresse."""
        try:
            from security.crypto import get_encryption_service

            if value:
                self.address_encrypted = get_encryption_service().encrypt_field(value)
                self.encryption_migrated = True
            else:
                self.address_encrypted = None
        except ImportError:
            self.address = value  # type: ignore[assignment]

    # Propriété pour la sérialisation
    @property
    def serialize(self):
        role_val = getattr(self, "role", None)
        result = {
            "id": self.id,
            "user_id": self.id,  # ✅ correction ici
            "public_id": self.public_id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name or "Non spécifié",
            "last_name": self.last_name or "Non spécifié",
            "phone": self.phone or "Non spécifié",
            "address": self.address or "Non spécifié",
            "birth_date": (
                self.birth_date.strftime("%Y-%m-%d") if self.birth_date else None
            ),
            "gender": (self.gender.value if self.gender else "Non spécifié"),
            "profile_image": self.profile_image or None,
            "role": (role_val.value if role_val else str(role_val)),
            "zip_code": self.zip_code or "Non spécifié",
            "city": self.city or "Non spécifié",
            "created_at": _iso(self.created_at),
            "force_password_change": self.force_password_change,
        }
        # ✅ Institution: Ajouter les champs institution si présents
        if self.institution_id:
            result["institution_id"] = self.institution_id
            result["institution_role"] = self.institution_role
            result["account_status"] = self.account_status or "active"
            result["job_title"] = self.job_title
        return result

    @property
    def full_name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip()

    # 📌 Représentation pour le debug
    @override
    def __repr__(self):
        return f"<User {self.username} ({self.email}) - Role: {self.role.value}>"

    def to_dict(self):
        return self.serialize
