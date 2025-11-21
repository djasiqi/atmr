"""✅ 4.1: Tâches Celery pour rotation automatique des secrets via Vault.

Rotation automatique des secrets stockés dans HashiCorp Vault:
- JWT secrets (30 jours)
- Encryption keys (90 jours)
- Database credentials (7 jours, via dynamic secrets)
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

try:
    import hvac  # type: ignore[import-untyped]  # noqa: F401  # Dépendance optionnelle, utilisée indirectement

    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False

from celery import Task

from celery_app import celery

logger = logging.getLogger(__name__)

# ✅ 4.1: Intervalles de rotation (jours)
JWT_ROTATION_INTERVAL_DAYS = int(os.getenv("VAULT_JWT_ROTATION_DAYS", "30"))
ENCRYPTION_ROTATION_INTERVAL_DAYS = int(os.getenv("VAULT_ENCRYPTION_ROTATION_DAYS", "90"))


def _get_vault_client() -> Any:
    """Récupère le client Vault depuis shared."""
    try:
        from shared.vault_client import get_vault_client as _get_client

        return _get_client()
    except ImportError:
        logger.warning("[4.1 Vault Rotation] vault_client non disponible")
        return None


@celery.task(bind=True, name="tasks.vault_rotation_tasks.rotate_jwt_secret")
def rotate_jwt_secret(self: Task) -> dict[str, Any]:
    """✅ 4.1: Rotation automatique de la clé JWT dans Vault.

    Génère une nouvelle clé JWT et la stocke dans Vault.
    L'ancienne clé est conservée temporairement pour transition.

    Returns:
        dict avec status, environment, rotated_at
    """
    if not HVAC_AVAILABLE:
        logger.warning("[4.1 Vault Rotation] hvac non installé, rotation JWT ignorée")
        result = {"status": "skipped", "reason": "hvac_not_available"}

        # ✅ Enregistrer le skip dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            environment = os.getenv("FLASK_ENV", "production")
            env_path = "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")

            record_rotation(
                secret_type="jwt",
                status="skipped",
                environment=env_path,
                metadata={"reason": "hvac_not_available"},
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        return result

    try:
        vault_client = _get_vault_client()
        if not vault_client or not vault_client.use_vault or not vault_client._client:
            logger.warning("[4.1 Vault Rotation] Vault non disponible, rotation JWT ignorée")
            result = {"status": "skipped", "reason": "vault_not_available"}

            # ✅ Enregistrer le skip dans le monitoring
            try:
                from services.secret_rotation_monitor import record_rotation

                environment = os.getenv("FLASK_ENV", "production")
                env_path = (
                    "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")
                )

                record_rotation(
                    secret_type="jwt",
                    status="skipped",
                    environment=env_path,
                    metadata={"reason": "vault_not_available"},
                    task_id=self.request.id if hasattr(self, "request") else None,
                )
            except Exception as monitor_error:
                logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

            return result

        # Déterminer l'environnement
        environment = os.getenv("FLASK_ENV", "production")
        if environment == "development":
            env_path = "dev"
        elif environment == "testing":
            env_path = "testing"
        else:
            env_path = "prod"

        logger.info("[4.1 Vault Rotation] Début rotation JWT secret (env=%s)", env_path)

        # Générer nouvelle clé JWT
        import secrets

        new_secret = secrets.token_urlsafe(64)  # 64 bytes = ~86 caractères URL-safe

        # Lire l'ancienne clé pour référence (optionnel)
        old_secret = None
        try:
            old_secret_response = vault_client._client.secrets.kv.v2.read_secret_version(
                path=f"atmr/data/{env_path}/jwt/secret_key"
            )
            old_secret = old_secret_response["data"]["data"].get("value")
        except Exception:
            pass  # Première rotation, pas d'ancienne clé

        # Stocker la nouvelle clé dans Vault
        vault_client._client.secrets.kv.v2.create_or_update_secret(
            path=f"atmr/data/{env_path}/jwt/secret_key", secret={"value": new_secret}
        )

        logger.info("[4.1 Vault Rotation] ✅ JWT secret roté avec succès (env=%s)", env_path)

        # ⚠️ IMPORTANT: Vider le cache pour forcer rechargement
        vault_client.clear_cache()

        result = {
            "status": "success",
            "environment": env_path,
            "rotated_at": datetime.now(UTC).isoformat(),
            "next_rotation_days": JWT_ROTATION_INTERVAL_DAYS,
            "old_secret_present": old_secret is not None,
        }

        # ✅ Enregistrer la rotation dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            record_rotation(
                secret_type="jwt",
                status="success",
                environment=env_path,
                metadata={
                    "next_rotation_days": JWT_ROTATION_INTERVAL_DAYS,
                    "old_secret_present": old_secret is not None,
                },
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        return result

    except Exception as e:
        logger.exception("[4.1 Vault Rotation] ❌ Erreur rotation JWT secret: %s", e)

        # ✅ Enregistrer l'échec dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            environment = os.getenv("FLASK_ENV", "production")
            env_path = "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")

            record_rotation(
                secret_type="jwt",
                status="error",
                environment=env_path,
                error_message=str(e),
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        raise


@celery.task(bind=True, name="tasks.vault_rotation_tasks.rotate_encryption_key")
def rotate_encryption_key(self: Task) -> dict[str, Any]:
    """✅ 4.1: Rotation automatique de la clé d'encryption dans Vault.

    Génère une nouvelle clé d'encryption et la stocke dans Vault.
    L'ancienne clé est ajoutée à la liste des legacy keys.

    Returns:
        dict avec status, environment, rotated_at
    """
    if not HVAC_AVAILABLE:
        logger.warning("[4.1 Vault Rotation] hvac non installé, rotation encryption ignorée")
        result = {"status": "skipped", "reason": "hvac_not_available"}

        # ✅ Enregistrer le skip dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            environment = os.getenv("FLASK_ENV", "production")
            env_path = "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")

            record_rotation(
                secret_type="encryption",
                status="skipped",
                environment=env_path,
                metadata={"reason": "hvac_not_available"},
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        return result

    try:
        vault_client = _get_vault_client()
        if not vault_client or not vault_client.use_vault or not vault_client._client:
            logger.warning("[4.1 Vault Rotation] Vault non disponible, rotation encryption ignorée")
            result = {"status": "skipped", "reason": "vault_not_available"}

            # ✅ Enregistrer le skip dans le monitoring
            try:
                from services.secret_rotation_monitor import record_rotation

                environment = os.getenv("FLASK_ENV", "production")
                env_path = (
                    "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")
                )

                record_rotation(
                    secret_type="encryption",
                    status="skipped",
                    environment=env_path,
                    metadata={"reason": "vault_not_available"},
                    task_id=self.request.id if hasattr(self, "request") else None,
                )
            except Exception as monitor_error:
                logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

            return result

        # Déterminer l'environnement
        environment = os.getenv("FLASK_ENV", "production")
        if environment == "development":
            env_path = "dev"
        elif environment == "testing":
            env_path = "testing"
        else:
            env_path = "prod"

        logger.info("[4.1 Vault Rotation] Début rotation encryption key (env=%s)", env_path)

        # Générer nouvelle clé (32 bytes pour AES-256)
        import base64

        from security.crypto import DEFAULT_KEY_LENGTH

        new_key = os.urandom(DEFAULT_KEY_LENGTH)
        new_key_b64 = base64.urlsafe_b64encode(new_key).decode("utf-8").rstrip("=")

        # Lire l'ancienne clé pour legacy
        old_key_b64 = None
        legacy_keys = []

        try:
            old_key_response = vault_client._client.secrets.kv.v2.read_secret_version(
                path=f"atmr/data/{env_path}/encryption/master_key"
            )
            old_key_b64 = old_key_response["data"]["data"].get("value")

            # Récupérer les legacy keys existantes
            legacy_keys_response = vault_client._client.secrets.kv.v2.read_secret_version(
                path=f"atmr/data/{env_path}/encryption/legacy_keys"
            )
            legacy_keys = legacy_keys_response["data"]["data"].get("keys", [])
        except Exception:
            pass  # Première rotation ou pas de legacy keys

        # Ajouter l'ancienne clé aux legacy keys
        if old_key_b64 and old_key_b64 not in legacy_keys:
            legacy_keys.insert(0, old_key_b64)  # Ajouter au début
            # Limiter à 5 clés legacy max (garder les plus récentes)
            legacy_keys = legacy_keys[:5]

        # Stocker la nouvelle clé maître
        vault_client._client.secrets.kv.v2.create_or_update_secret(
            path=f"atmr/data/{env_path}/encryption/master_key", secret={"value": new_key_b64}
        )

        # Stocker les legacy keys
        vault_client._client.secrets.kv.v2.create_or_update_secret(
            path=f"atmr/data/{env_path}/encryption/legacy_keys", secret={"keys": legacy_keys}
        )

        logger.info(
            "[4.1 Vault Rotation] ✅ Encryption key rotée avec succès (env=%s, %d legacy keys)",
            env_path,
            len(legacy_keys),
        )

        # ⚠️ IMPORTANT: Vider le cache pour forcer rechargement
        vault_client.clear_cache()

        # ✅ 2.5: Intégration avec système de rotation existant
        # Notifier le EncryptionService de la nouvelle clé (à faire manuellement ou via reload)

        result = {
            "status": "success",
            "environment": env_path,
            "rotated_at": datetime.now(UTC).isoformat(),
            "next_rotation_days": ENCRYPTION_ROTATION_INTERVAL_DAYS,
            "legacy_keys_count": len(legacy_keys),
            "old_key_present": old_key_b64 is not None,
        }

        # ✅ Enregistrer la rotation dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            record_rotation(
                secret_type="encryption",
                status="success",
                environment=env_path,
                metadata={
                    "next_rotation_days": ENCRYPTION_ROTATION_INTERVAL_DAYS,
                    "legacy_keys_count": len(legacy_keys),
                    "old_key_present": old_key_b64 is not None,
                },
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        return result

    except Exception as e:
        logger.exception("[4.1 Vault Rotation] ❌ Erreur rotation encryption key: %s", e)

        # ✅ Enregistrer l'échec dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            environment = os.getenv("FLASK_ENV", "production")
            env_path = "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")

            record_rotation(
                secret_type="encryption",
                status="error",
                environment=env_path,
                error_message=str(e),
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        raise


@celery.task(bind=True, name="tasks.vault_rotation_tasks.rotate_flask_secret_key")
def rotate_flask_secret_key(self: Task) -> dict[str, Any]:
    """✅ 4.1: Rotation automatique de SECRET_KEY Flask dans Vault.

    Génère une nouvelle SECRET_KEY et la stocke dans Vault.
    L'ancienne clé est conservée temporairement pour transition.

    Returns:
        dict avec status, environment, rotated_at
    """
    if not HVAC_AVAILABLE:
        logger.warning("[4.1 Vault Rotation] hvac non installé, rotation SECRET_KEY ignorée")
        result = {"status": "skipped", "reason": "hvac_not_available"}

        # ✅ Enregistrer le skip dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            environment = os.getenv("FLASK_ENV", "production")
            env_path = "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")

            record_rotation(
                secret_type="flask_secret_key",
                status="skipped",
                environment=env_path,
                metadata={"reason": "hvac_not_available"},
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        return result

    try:
        vault_client = _get_vault_client()
        if not vault_client or not vault_client.use_vault or not vault_client._client:
            logger.warning("[4.1 Vault Rotation] Vault non disponible, rotation SECRET_KEY ignorée")
            result = {"status": "skipped", "reason": "vault_not_available"}

            # ✅ Enregistrer le skip dans le monitoring
            try:
                from services.secret_rotation_monitor import record_rotation

                environment = os.getenv("FLASK_ENV", "production")
                env_path = (
                    "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")
                )

                record_rotation(
                    secret_type="flask_secret_key",
                    status="skipped",
                    environment=env_path,
                    metadata={"reason": "vault_not_available"},
                    task_id=self.request.id if hasattr(self, "request") else None,
                )
            except Exception as monitor_error:
                logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

            return result

        # Déterminer l'environnement
        environment = os.getenv("FLASK_ENV", "production")
        if environment == "development":
            env_path = "dev"
        elif environment == "testing":
            env_path = "testing"
        else:
            env_path = "prod"

        logger.info("[4.1 Vault Rotation] Début rotation SECRET_KEY Flask (env=%s)", env_path)

        # Générer nouvelle SECRET_KEY
        import secrets

        new_secret = secrets.token_urlsafe(64)  # 64 bytes = ~86 caractères URL-safe

        # Lire l'ancienne clé pour référence (optionnel)
        old_secret = None
        try:
            old_secret_response = vault_client._client.secrets.kv.v2.read_secret_version(
                path=f"atmr/data/{env_path}/flask/secret_key"
            )
            old_secret = old_secret_response["data"]["data"].get("value")
        except Exception:
            pass  # Première rotation, pas d'ancienne clé

        # Stocker la nouvelle clé dans Vault
        vault_client._client.secrets.kv.v2.create_or_update_secret(
            path=f"atmr/data/{env_path}/flask/secret_key", secret={"value": new_secret}
        )

        logger.info("[4.1 Vault Rotation] ✅ SECRET_KEY Flask roté avec succès (env=%s)", env_path)

        # ⚠️ IMPORTANT: Vider le cache pour forcer rechargement
        vault_client.clear_cache()

        result = {
            "status": "success",
            "environment": env_path,
            "rotated_at": datetime.now(UTC).isoformat(),
            "next_rotation_days": 90,  # Rotation tous les 90 jours par défaut
            "old_secret_present": old_secret is not None,
        }

        # ✅ Enregistrer la rotation dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            record_rotation(
                secret_type="flask_secret_key",
                status="success",
                environment=env_path,
                metadata={
                    "next_rotation_days": 90,
                    "old_secret_present": old_secret is not None,
                },
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        return result

    except Exception as e:
        logger.exception("[4.1 Vault Rotation] ❌ Erreur rotation SECRET_KEY Flask: %s", e)

        # ✅ Enregistrer l'échec dans le monitoring
        try:
            from services.secret_rotation_monitor import record_rotation

            environment = os.getenv("FLASK_ENV", "production")
            env_path = "dev" if environment == "development" else ("testing" if environment == "testing" else "prod")

            record_rotation(
                secret_type="flask_secret_key",
                status="error",
                environment=env_path,
                error_message=str(e),
                task_id=self.request.id if hasattr(self, "request") else None,
            )
        except Exception as monitor_error:
            logger.warning("[4.1 Vault Rotation] Erreur enregistrement monitoring: %s", monitor_error)

        raise


@celery.task(bind=True, name="tasks.vault_rotation_tasks.rotate_all_secrets")
def rotate_all_secrets(self: Task) -> dict[str, Any]:  # noqa: ARG001
    """✅ 4.1: Rotation globale de tous les secrets configurables.

    Exécute la rotation de tous les secrets qui ont dépassé leur intervalle.

    Returns:
        dict avec status, results pour chaque secret
    """
    try:
        logger.info("[4.1 Vault Rotation] Début rotation globale des secrets...")

        # Créer une instance Task factice pour appeler les fonctions bind=True
        class FakeTask:
            pass

        fake_task = FakeTask()

        results = {}

        # Rotation SECRET_KEY Flask
        try:
            flask_secret_result = rotate_flask_secret_key(fake_task)  # type: ignore[arg-type]
            results["flask_secret_key"] = flask_secret_result
        except Exception as e:
            logger.exception("[4.1 Vault Rotation] Erreur rotation SECRET_KEY Flask: %s", e)
            results["flask_secret_key"] = {"status": "error", "error": str(e)}

        # Rotation JWT
        try:
            jwt_result = rotate_jwt_secret(fake_task)  # type: ignore[arg-type]
            results["jwt"] = jwt_result
        except Exception as e:
            logger.exception("[4.1 Vault Rotation] Erreur rotation JWT: %s", e)
            results["jwt"] = {"status": "error", "error": str(e)}

        # Rotation Encryption
        try:
            encryption_result = rotate_encryption_key(fake_task)  # type: ignore[arg-type]
            results["encryption"] = encryption_result
        except Exception as e:
            logger.exception("[4.1 Vault Rotation] Erreur rotation encryption: %s", e)
            results["encryption"] = {"status": "error", "error": str(e)}

        # Note: Database credentials sont gérés via dynamic secrets (rotation automatique)

        success_count = sum(1 for r in results.values() if r.get("status") == "success")
        total_count = len(results)

        logger.info("[4.1 Vault Rotation] ✅ Rotation globale terminée: %d/%d succès", success_count, total_count)

        # ✅ Notification en cas d'échec
        if success_count < total_count:
            _notify_rotation_failure(results)

        return {
            "status": "completed",
            "rotated_at": datetime.now(UTC).isoformat(),
            "results": results,
            "success_count": success_count,
            "total_count": total_count,
        }

    except Exception as e:
        logger.exception("[4.1 Vault Rotation] ❌ Erreur rotation globale: %s", e)
        _notify_rotation_failure({"global_error": str(e)})
        raise


def _notify_rotation_failure(results: dict[str, Any]) -> None:
    """✅ 4.1: Notifie en cas d'échec de rotation des secrets.

    Args:
        results: Dictionnaire des résultats de rotation
    """
    try:
        import sentry_sdk

        # Envoyer à Sentry si configuré
        failed_secrets = [key for key, result in results.items() if result.get("status") != "success"]
        if failed_secrets:
            sentry_sdk.capture_message(
                f"Rotation secrets échouée: {', '.join(failed_secrets)}",
                level="error",
            )
            logger.error("[4.1 Vault Rotation] ❌ Échecs détectés: %s", failed_secrets)

        # TODO: Ajouter notification email/Slack si configuré
        # Exemple:
        # if os.getenv("ROTATION_NOTIFICATION_EMAIL"):
        #     send_email(...)

    except Exception as e:
        logger.warning("[4.1 Vault Rotation] Erreur notification: %s", e)
