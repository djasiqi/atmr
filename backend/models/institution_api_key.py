# models/institution_api_key.py
# pyright: reportGeneralTypeIssues=false, reportUnnecessaryComparison=false
"""Model InstitutionApiKey - Gestion des clés API pour DPI (logiciels cliniques).

Permet aux logiciels DPI de s'authentifier via X-API-Key sans interface web.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso

if TYPE_CHECKING:
    from .institution import Institution
    from .user import User

# Préfixe pour les clés API Lirie
API_KEY_PREFIX = "lir_"
# Longueur de la partie aléatoire de la clé (en bytes, sera encodé en hex)
API_KEY_RANDOM_BYTES = 32
# Secret HMAC pour le hashing (depuis env ou valeur par défaut dev)
API_KEY_HMAC_SECRET = os.getenv("API_KEY_HMAC_SECRET", "dev-secret-change-in-prod")

# Scopes autorisés pour les clés API
VALID_SCOPES = frozenset({
    "patients:read",
    "patients:write",
    "requests:read",
    "requests:write",
    "requests:cancel",
})


def generate_api_key() -> tuple[str, str, str]:
    """Génère une nouvelle clé API.

    Returns:
        Tuple (raw_key, key_prefix, key_hash):
        - raw_key: Clé brute à afficher une seule fois (ex: "lir_abc123...def456")
        - key_prefix: Préfixe pour identification (ex: "lir_abc123")
        - key_hash: Hash HMAC-SHA256 pour stockage sécurisé
    """
    # Générer la partie aléatoire
    random_part = secrets.token_hex(API_KEY_RANDOM_BYTES)

    # Construire la clé complète
    raw_key = f"{API_KEY_PREFIX}{random_part}"

    # Extraire le préfixe (8 premiers caractères après "lir_")
    key_prefix = f"{API_KEY_PREFIX}{random_part[:8]}"

    # Calculer le hash HMAC-SHA256
    key_hash = hash_api_key(raw_key)

    return raw_key, key_prefix, key_hash


def hash_api_key(raw_key: str) -> str:
    """Calcule le hash HMAC-SHA256 d'une clé API.

    Args:
        raw_key: Clé API brute

    Returns:
        Hash HMAC-SHA256 en hexadécimal
    """
    return hmac.new(
        API_KEY_HMAC_SECRET.encode(),
        raw_key.encode(),
        hashlib.sha256,
    ).hexdigest()


def validate_scopes(scopes: list[str]) -> tuple[bool, list[str]]:
    """Valide une liste de scopes.

    Args:
        scopes: Liste de scopes à valider

    Returns:
        Tuple (is_valid, invalid_scopes):
        - is_valid: True si tous les scopes sont valides
        - invalid_scopes: Liste des scopes invalides
    """
    invalid = [s for s in scopes if s not in VALID_SCOPES]
    return len(invalid) == 0, invalid


class InstitutionApiKey(db.Model):
    """Modèle pour les clés API des institutions (DPI).

    Permet aux logiciels DPI de s'authentifier via header X-API-Key.
    La clé brute n'est jamais stockée, uniquement son hash HMAC-SHA256.
    """

    __tablename__ = "institution_api_keys"
    __table_args__ = (
        Index("ix_institution_api_keys_institution_id", "institution_id"),
        Index("ix_institution_api_keys_key_prefix", "key_prefix"),
        Index("ix_institution_api_keys_key_hash", "key_hash", unique=True),
    )

    # Identifiant
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Institution propriétaire
    institution_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Nom descriptif de la clé (ex: "DPI Prod", "DPI Test")
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Préfixe de la clé pour identification (ex: "lir_abc123")
    key_prefix: Mapped[str] = mapped_column(String(20), nullable=False)

    # Hash HMAC-SHA256 de la clé (jamais la clé en clair)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Scopes autorisés (JSON array stocké en texte)
    # Ex: '["patients:read","requests:write"]'
    scopes: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    # Dernière utilisation
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    # Créateur de la clé
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relations
    institution: Mapped[Institution] = relationship(
        "Institution",
        backref="api_keys",
    )
    created_by: Mapped[User | None] = relationship(
        "User",
        foreign_keys=[created_by_user_id],
    )

    @override
    def __repr__(self) -> str:
        status = "revoked" if self.revoked_at else "active"
        return f"<InstitutionApiKey {self.id}: {self.name} ({status})>"

    @property
    def is_revoked(self) -> bool:
        """Retourne True si la clé est révoquée."""
        return self.revoked_at is not None

    @property
    def is_active(self) -> bool:
        """Retourne True si la clé est active (non révoquée)."""
        return self.revoked_at is None

    def get_scopes(self) -> list[str]:
        """Retourne la liste des scopes autorisés."""
        import json

        try:
            return json.loads(self.scopes)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_scopes(self, scopes: list[str]) -> None:
        """Définit les scopes autorisés."""
        import json

        self.scopes = json.dumps(scopes)

    def has_scope(self, scope: str) -> bool:
        """Vérifie si la clé a un scope donné."""
        return scope in self.get_scopes()

    def revoke(self) -> None:
        """Révoque la clé API."""
        self.revoked_at = datetime.now(UTC)

    def update_last_used(self) -> None:
        """Met à jour la date de dernière utilisation."""
        self.last_used_at = datetime.now(UTC)

    @property
    def serialize(self) -> dict[str, Any]:
        """Sérialise la clé API pour l'API (sans la clé brute)."""
        return {
            "id": self.id,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "scopes": self.get_scopes(),
            "last_used_at": _iso(self.last_used_at),
            "created_at": _iso(self.created_at),
            "revoked_at": _iso(self.revoked_at),
            "is_active": self.is_active,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias pour serialize."""
        return self.serialize

    @classmethod
    def find_by_raw_key(cls, raw_key: str) -> InstitutionApiKey | None:
        """Trouve une clé API par sa valeur brute.

        Args:
            raw_key: Clé API brute (ex: "lir_abc123...def456")

        Returns:
            InstitutionApiKey si trouvée et active, None sinon
        """
        key_hash = hash_api_key(raw_key)
        return cls.query.filter_by(key_hash=key_hash).first()

    @classmethod
    def find_active_by_raw_key(cls, raw_key: str) -> InstitutionApiKey | None:
        """Trouve une clé API active par sa valeur brute.

        Args:
            raw_key: Clé API brute

        Returns:
            InstitutionApiKey si trouvée et active, None sinon
        """
        api_key = cls.find_by_raw_key(raw_key)
        if api_key and api_key.is_active:
            return api_key
        return None
