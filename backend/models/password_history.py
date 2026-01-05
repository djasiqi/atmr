"""✅ S3: Modèle pour l'historique des mots de passe.

Stocke les hashs des mots de passe précédents pour empêcher la réutilisation.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String

from ext import db


class PasswordHistory(db.Model):
    """Stocke l'historique des mots de passe pour chaque utilisateur.

    Permet d'empêcher la réutilisation des N derniers mots de passe.
    """

    __tablename__ = "password_history"

    id = Column(Integer, primary_key=True)

    # ID de l'utilisateur
    user_id = Column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Hash du mot de passe (bcrypt)
    password_hash = Column(String(255), nullable=False)

    # Date de création (quand le mot de passe a été utilisé)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    # Index composite pour optimiser les requêtes
    __table_args__ = (
        Index("ix_password_history_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        return f"<PasswordHistory id={self.id} user_id={self.user_id} created_at={self.created_at}>"
