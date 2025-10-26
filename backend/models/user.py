# models/user.py
"""Model User - Gestion des utilisateurs (tous rôles).
Extrait depuis models.py (lignes 249-418).
"""
from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Optional, cast

from sqlalchemy import Boolean, Column, Date, DateTime, Index, Integer, String, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override
from werkzeug.security import check_password_hash, generate_password_hash

from ext import db

from .base import _coerce_enum, _iso
from .enums import GenderEnum, UserRole

ADDRESS_MAX_LENGTH = 200


class User(db.Model):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id = Column(
        String(36),
        default=lambda: str(
            uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    first_name: Mapped[str] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str] = mapped_column(String(100), nullable=True)
    email = Column(String(255), nullable=True, unique=True, index=True)

    # ↓ Champs présents pour tous les rôles (client, driver, etc.)
    phone: Mapped[str] = mapped_column(String(255), nullable=True)
    address: Mapped[str] = mapped_column(String(200), nullable=True)
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    gender: Mapped[GenderEnum] = mapped_column(SAEnum(GenderEnum, name="gender"), nullable=True)
    profile_image: Mapped[str] = mapped_column(String(255), nullable=True)

    password: Mapped[str] = mapped_column(String(255), nullable=False)
    role = Column(
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

    # ✅ Ajout de l'index sur `public_id` pour optimiser les recherches
    __table_args__ = (
        Index("idx_public_id", "public_id"),
    )

    # ✅ Relations bidirectionnelles avec suppression en cascade
    clients = relationship(
        "Client",
        back_populates="user",
        cascade="all, delete-orphan")
    driver = relationship(
        "Driver",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True)
    company = relationship(
        "Company",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True)

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
            msg = "Numéro de téléphone invalide. Doit contenir 7 à 15 chiffres avec option '+'."
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
        coerced = _coerce_enum(gender_value, GenderEnum)
        if coerced is None:
            msg = "Genre invalide."
            raise ValueError(msg)
        return coerced

    @validates("role")
    def validate_role(self, _key, role_value):
        """Coerce str → UserRole, évite d'évaluer un Column en bool."""
        coerced = _coerce_enum(role_value, UserRole)
        if coerced is None:
            msg = "Invalid role value. Allowed values: admin, client, driver, company."
            raise ValueError(msg)
        return coerced

    @validates("email")
    def validate_email(self, _key, email):
        """Valide le format si fourni.
        ⚠️ La règle 'self-service => email requis' est déjà appliquée dans Client.validate_contact_email.
        On évite ici toute logique cross-model (et donc les tests sur self.clients / self.role).
        """
        if email is None or email.strip() == "":
            return None
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email.strip()):
            msg = "Format d'email invalide."
            raise ValueError(msg)
        return email.strip()

    # Propriété pour la sérialisation
    @property
    def serialize(self):
        role_val = getattr(self, "role", None)
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
            "birth_date": (self.birth_date.strftime("%Y-%m-%d") if self.birth_date else None),
            "gender": (self.gender.value if self.gender else "Non spécifié"),
            "profile_image": self.profile_image or None,
            "role": (role_val.value if role_val else str(role_val)),
            "zip_code": self.zip_code or "Non spécifié",
            "city": self.city or "Non spécifié",
            "created_at": _iso(self.created_at),
            "force_password_change": self.force_password_change
        }

    @property
    def full_name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip()

    # 📌 Représentation pour le debug
    @override
    def __repr__(self):
        return f"<User {self.username} ({self.email}) - Role: {self.role.value}>"

    def to_dict(self):
        return self.serialize
