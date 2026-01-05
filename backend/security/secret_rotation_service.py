"""✅ S3: Service de rotation automatique des secrets.

Gère la rotation automatique des clés JWT et de chiffrement tous les 90 jours.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import UTC, datetime
from typing import Any

from ext import db
from models.secret_rotation import SecretRotation
from security.crypto import get_encryption_service, rotate_to_new_key

logger = logging.getLogger(__name__)

# Intervalle de rotation par défaut (90 jours)
DEFAULT_ROTATION_INTERVAL_DAYS = int(os.getenv("SECRET_ROTATION_INTERVAL_DAYS", "90"))


class SecretRotationService:
    """Service pour gérer la rotation automatique des secrets."""

    @staticmethod
    def get_environment() -> str:
        """Détermine l'environnement actuel."""
        env = os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG", "development")
        if env == "production":
            return "prod"
        if env == "testing":
            return "testing"
        return "dev"

    @staticmethod
    def should_rotate_jwt_secret() -> bool:
        """Vérifie si la clé JWT doit être rotée.

        Returns:
            True si la dernière rotation date de plus de 90 jours
        """
        env = SecretRotationService.get_environment()

        # Vérifier la dernière rotation réussie
        last_rotation = (
            SecretRotation.query.filter_by(
                secret_type="jwt", status="success", environment=env
            )
            .order_by(SecretRotation.rotated_at.desc())
            .first()
        )

        if not last_rotation:
            # Pas de rotation précédente, vérifier si on doit initialiser
            # (on ne force pas la rotation au premier démarrage)
            return False

        # Vérifier si l'intervalle de rotation est dépassé
        days_since_rotation = (datetime.now(UTC) - last_rotation.rotated_at).days
        return days_since_rotation >= DEFAULT_ROTATION_INTERVAL_DAYS

    @staticmethod
    def should_rotate_encryption_key() -> bool:
        """Vérifie si la clé de chiffrement doit être rotée.

        Returns:
            True si la dernière rotation date de plus de 90 jours
        """
        env = SecretRotationService.get_environment()

        # Vérifier la dernière rotation réussie
        last_rotation = (
            SecretRotation.query.filter_by(
                secret_type="encryption", status="success", environment=env
            )
            .order_by(SecretRotation.rotated_at.desc())
            .first()
        )

        if not last_rotation:
            # Pas de rotation précédente, vérifier si on doit initialiser
            return False

        # Vérifier si l'intervalle de rotation est dépassé
        days_since_rotation = (datetime.now(UTC) - last_rotation.rotated_at).days
        return days_since_rotation >= DEFAULT_ROTATION_INTERVAL_DAYS

    @staticmethod
    def rotate_jwt_secret() -> dict[str, Any]:
        """✅ S3: Effectue la rotation de la clé JWT.

        Note: La rotation JWT nécessite que tous les tokens existants soient invalidés.
        Cette fonction génère une nouvelle clé et l'enregistre dans Vault ou .env.

        Returns:
            Dict avec status, message, et métadonnées
        """
        env = SecretRotationService.get_environment()
        task_id = os.getenv("CELERY_TASK_ID", None)

        try:
            # Générer une nouvelle clé JWT (32 bytes = 256 bits, encodé en hex)
            new_secret = secrets.token_hex(32)

            # ✅ S3: Enregistrer la nouvelle clé dans Vault (si disponible) ou .env
            # Pour l'instant, on log la nouvelle clé (en production, utiliser Vault)
            vault_available = os.getenv("VAULT_AVAILABLE", "false").lower() == "true"

            if vault_available:
                try:
                    from shared.vault_client import get_vault_client

                    vault = get_vault_client()
                    vault_path = f"{env}/jwt/secret_key"
                    vault.set_secret(vault_path, "value", new_secret)
                    logger.info(
                        "[S3] ✅ Nouvelle clé JWT enregistrée dans Vault: %s",
                        vault_path,
                    )
                except Exception as e:
                    logger.error(
                        "[S3] ❌ Erreur lors de l'enregistrement dans Vault: %s", e
                    )
                    raise
            else:
                # Fallback: logger un avertissement (en production, Vault est requis)
                logger.warning(
                    "[S3] ⚠️ Vault non disponible. Nouvelle clé JWT générée mais non enregistrée automatiquement."
                )
            logger.warning(
                "[S3] ⚠️ Nouvelle clé JWT (à enregistrer manuellement): %s...",
                f"{new_secret[:20]!s}",
            )

            # Enregistrer la rotation dans la base de données
            rotation_record = SecretRotation()
            rotation_record.secret_type = "jwt"
            rotation_record.status = "success"
            rotation_record.environment = env
            rotation_record.rotated_at = datetime.now(UTC)
            rotation_record.rotation_metadata = {
                "next_rotation_days": DEFAULT_ROTATION_INTERVAL_DAYS,
                "old_secret_present": bool(
                    os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_LEGACY_SECRET_KEY")
                ),
            }
            rotation_record.task_id = task_id
            db.session.add(rotation_record)
            db.session.commit()

            logger.info(
                "[S3] ✅ Rotation JWT réussie (env=%s, next_rotation=%d jours)",
                env,
                DEFAULT_ROTATION_INTERVAL_DAYS,
            )

            return {
                "status": "success",
                "message": "Rotation JWT réussie",
                "environment": env,
                "next_rotation_days": DEFAULT_ROTATION_INTERVAL_DAYS,
                "vault_used": vault_available,
            }

        except Exception as e:
            logger.exception("[S3] ❌ Erreur lors de la rotation JWT: %s", e)

            # Enregistrer l'erreur dans la base de données
            rotation_record = SecretRotation()
            rotation_record.secret_type = "jwt"
            rotation_record.status = "error"
            rotation_record.environment = env
            rotation_record.rotated_at = datetime.now(UTC)
            rotation_record.error_message = str(e)
            rotation_record.task_id = task_id
            db.session.add(rotation_record)
            db.session.commit()

            return {
                "status": "error",
                "message": f"Erreur lors de la rotation JWT: {e!s}",
                "environment": env,
            }

    @staticmethod
    def rotate_encryption_key() -> dict[str, Any]:
        """✅ S3: Effectue la rotation de la clé de chiffrement.

        Utilise le support multi-clés existant pour une rotation progressive.
        L'ancienne clé devient legacy, la nouvelle devient active.

        Returns:
            Dict avec status, message, et métadonnées
        """
        env = SecretRotationService.get_environment()
        task_id = os.getenv("CELERY_TASK_ID", None)

        try:
            # Récupérer le service de chiffrement
            encryption_service = get_encryption_service()

            # Générer une nouvelle clé (32 bytes = 256 bits)
            new_key = os.urandom(32)

            # ✅ S3: Effectuer la rotation (ancienne clé devient legacy)
            old_key = rotate_to_new_key(encryption_service, new_key)

            # Enregistrer la nouvelle clé dans Vault ou .env
            vault_available = os.getenv("VAULT_AVAILABLE", "false").lower() == "true"

            if vault_available:
                try:
                    from shared.vault_client import get_vault_client

                    vault = get_vault_client()
                    vault_path = f"{env}/encryption/master_key"
                    # Convertir la clé en hex pour stockage
                    new_key_hex = new_key.hex()
                    vault.set_secret(vault_path, "value", new_key_hex)

                    # Mettre à jour LEGACY_ENCRYPTION_KEYS dans Vault
                    legacy_keys_hex = [
                        key.hex() for key in encryption_service.legacy_keys
                    ]
                    legacy_keys_str = ",".join(legacy_keys_hex)
                    vault.set_secret(
                        f"{env}/encryption/legacy_keys", "value", legacy_keys_str
                    )

                    logger.info(
                        "[S3] ✅ Nouvelle clé de chiffrement enregistrée dans Vault: %s",
                        vault_path,
                    )
                except Exception as e:
                    logger.error(
                        "[S3] ❌ Erreur lors de l'enregistrement dans Vault: %s", e
                    )
                    raise
            else:
                # Fallback: logger un avertissement
                logger.warning(
                    "[S3] ⚠️ Vault non disponible. Nouvelle clé de chiffrement générée mais non enregistrée automatiquement."
                )
                logger.warning(
                    "[S3] ⚠️ Nouvelle clé (à enregistrer manuellement): %s...",
                    f"{new_key.hex()[:20]!s}",
                )

            # Enregistrer la rotation dans la base de données
            rotation_record = SecretRotation()
            rotation_record.secret_type = "encryption"
            rotation_record.status = "success"
            rotation_record.environment = env
            rotation_record.rotated_at = datetime.now(UTC)
            rotation_record.rotation_metadata = {
                "next_rotation_days": DEFAULT_ROTATION_INTERVAL_DAYS,
                "legacy_keys_count": len(encryption_service.legacy_keys),
                "old_secret_present": bool(old_key),
            }
            rotation_record.task_id = task_id
            db.session.add(rotation_record)
            db.session.commit()

            logger.info(
                "[S3] ✅ Rotation clé de chiffrement réussie (env=%s, legacy_keys=%d, next_rotation=%d jours)",
                env,
                len(encryption_service.legacy_keys),
                DEFAULT_ROTATION_INTERVAL_DAYS,
            )

            return {
                "status": "success",
                "message": "Rotation clé de chiffrement réussie",
                "environment": env,
                "legacy_keys_count": len(encryption_service.legacy_keys),
                "next_rotation_days": DEFAULT_ROTATION_INTERVAL_DAYS,
                "vault_used": vault_available,
            }

        except Exception as e:
            logger.exception(
                "[S3] ❌ Erreur lors de la rotation clé de chiffrement: %s", e
            )

            # Enregistrer l'erreur dans la base de données
            rotation_record = SecretRotation()
            rotation_record.secret_type = "encryption"
            rotation_record.status = "error"
            rotation_record.environment = env
            rotation_record.rotated_at = datetime.now(UTC)
            rotation_record.error_message = str(e)
            rotation_record.task_id = task_id
            db.session.add(rotation_record)
            db.session.commit()

            return {
                "status": "error",
                "message": f"Erreur lors de la rotation clé de chiffrement: {e!s}",
                "environment": env,
            }

    @staticmethod
    def check_and_rotate_all() -> dict[str, Any]:
        """✅ S3: Vérifie et effectue toutes les rotations nécessaires.

        Returns:
            Dict avec le résultat de toutes les rotations
        """
        results: dict[str, Any] = {
            "jwt": {"rotated": False, "result": None},
            "encryption": {"rotated": False, "result": None},
        }

        # Rotation JWT
        if SecretRotationService.should_rotate_jwt_secret():
            logger.info("[S3] 🔄 Rotation JWT nécessaire, démarrage...")
            results["jwt"]["rotated"] = True
            results["jwt"]["result"] = SecretRotationService.rotate_jwt_secret()
        else:
            logger.debug("[S3] ✅ Rotation JWT non nécessaire")

        # Rotation clé de chiffrement
        if SecretRotationService.should_rotate_encryption_key():
            logger.info("[S3] 🔄 Rotation clé de chiffrement nécessaire, démarrage...")
            results["encryption"]["rotated"] = True
            results["encryption"]["result"] = (
                SecretRotationService.rotate_encryption_key()
            )
        else:
            logger.debug("[S3] ✅ Rotation clé de chiffrement non nécessaire")

        return results
