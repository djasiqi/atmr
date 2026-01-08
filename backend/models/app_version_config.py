# models/app_version_config.py
"""Modèle pour la configuration des versions minimales et recommandées de
l'application mobile.

Ce modèle stocke les versions minimales requises et les dernières versions
disponibles pour chaque plateforme (Android/iOS), permettant de forcer ou
recommander des mises à jour.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column
from typing_extensions import override

from ext import db


class AppVersionConfig(db.Model):
    """Configuration des versions de l'application mobile par plateforme.

    Une seule ligne par plateforme (singleton pattern via clé unique).
    """

    __tablename__ = "app_version_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    platform: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False, index=True
    )  # "android" ou "ios"

    # Version minimale requise (force la mise à jour si version < min_required)
    min_required_version: Mapped[str] = mapped_column(String(20), nullable=False)

    # Dernière version disponible (recommandation si version < latest)
    latest_version: Mapped[str] = mapped_column(String(20), nullable=False)

    # URL du store pour la mise à jour
    store_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Message personnalisé pour la mise à jour (optionnel)
    update_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    @override
    def __repr__(self) -> str:
        return (
            f"<AppVersionConfig(platform={self.platform}, "
            f"min={self.min_required_version}, latest={self.latest_version})>"
        )

    def to_dict(self) -> dict[str, str | None]:
        """Convertit le modèle en dictionnaire pour l'API."""
        return {
            "platform": self.platform,
            "min_required_version": self.min_required_version,
            "latest_version": self.latest_version,
            "store_url": self.store_url,
            "update_message": self.update_message,
        }
