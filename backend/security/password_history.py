"""✅ S3: Service pour gérer l'historique des mots de passe.

Empêche la réutilisation des N derniers mots de passe.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from werkzeug.security import (  # pyright: ignore[reportMissingImports]
    check_password_hash,
)

from ext import db
from models.password_history import PasswordHistory

logger = logging.getLogger(__name__)

# Configuration
PASSWORD_HISTORY_COUNT = int(
    os.getenv("PASSWORD_HISTORY_COUNT", "5")
)  # Nombre de mots de passe à conserver dans l'historique


class PasswordHistoryService:
    """Service pour gérer l'historique des mots de passe."""

    @staticmethod
    def add_password_to_history(user_id: int, password_hash: str) -> None:
        """✅ S3: Ajoute un mot de passe à l'historique.

        Args:
            user_id: ID de l'utilisateur
            password_hash: Hash du mot de passe (bcrypt)
        """
        try:
            # Créer une nouvelle entrée d'historique
            history_entry = PasswordHistory()
            history_entry.user_id = user_id
            history_entry.password_hash = password_hash
            history_entry.created_at = datetime.now(UTC)

            db.session.add(history_entry)
            db.session.commit()

            # Nettoyer l'historique (garder seulement les N derniers)
            PasswordHistoryService._cleanup_history(user_id)

            logger.debug(
                "[PasswordHistory] ✅ Mot de passe ajouté à l'historique pour user_id=%s",
                user_id,
            )
        except Exception as e:
            db.session.rollback()
            logger.error(
                "[PasswordHistory] ❌ Erreur lors de l'ajout à l'historique: %s", e
            )
            raise

    @staticmethod
    def check_password_history(user_id: int, password: str) -> tuple[bool, str | None]:
        """✅ S3: Vérifie si un mot de passe a déjà été utilisé récemment.

        Args:
            user_id: ID de l'utilisateur
            password: Mot de passe en clair à vérifier

        Returns:
            Tuple (is_not_reused, error_message)
            - is_not_reused: True si le mot de passe n'a pas été utilisé récemment
            - error_message: Message d'erreur si réutilisé, None sinon
        """
        try:
            # Récupérer l'historique des mots de passe (les N derniers)
            history_entries = (
                PasswordHistory.query.filter_by(user_id=user_id)
                .order_by(PasswordHistory.created_at.desc())
                .limit(PASSWORD_HISTORY_COUNT)
                .all()
            )

            # Vérifier si le mot de passe correspond à un des hashs de l'historique
            for entry in history_entries:
                if check_password_hash(entry.password_hash, password):
                    logger.warning(
                        "[PasswordHistory] ⚠️ Mot de passe réutilisé détecté pour user_id=%s",
                        user_id,
                    )
                    return (
                        False,
                        f"Ce mot de passe a déjà été utilisé récemment. Veuillez choisir un mot de passe différent parmi les {PASSWORD_HISTORY_COUNT} derniers.",
                    )

            # Le mot de passe n'a pas été trouvé dans l'historique
            logger.debug(
                "[PasswordHistory] ✅ Mot de passe non réutilisé pour user_id=%s",
                user_id,
            )
            return True, None

        except Exception as e:
            logger.error(
                "[PasswordHistory] ❌ Erreur lors de la vérification de l'historique: %s",
                e,
            )
            # En cas d'erreur, on accepte le mot de passe (fail-open)
            return True, None

    @staticmethod
    def _cleanup_history(user_id: int) -> None:
        """Nettoie l'historique en gardant seulement les N derniers mots de passe.

        Args:
            user_id: ID de l'utilisateur
        """
        try:
            # Compter le nombre d'entrées
            count = PasswordHistory.query.filter_by(user_id=user_id).count()

            if count > PASSWORD_HISTORY_COUNT:
                # Récupérer les IDs des entrées à supprimer (les plus anciennes)
                entries_to_delete = (
                    PasswordHistory.query.filter_by(user_id=user_id)
                    .order_by(PasswordHistory.created_at.asc())
                    .limit(count - PASSWORD_HISTORY_COUNT)
                    .all()
                )

                for entry in entries_to_delete:
                    db.session.delete(entry)

                db.session.commit()

                logger.debug(
                    "[PasswordHistory] ✅ Historique nettoyé pour user_id=%s (%d entrées supprimées)",
                    user_id,
                    len(entries_to_delete),
                )

        except Exception as e:
            db.session.rollback()
            logger.error(
                "[PasswordHistory] ❌ Erreur lors du nettoyage de l'historique: %s", e
            )
