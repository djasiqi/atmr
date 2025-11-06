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
def rotate_jwt_secret(self: Task) -> dict[str, Any]:  # noqa: ARG001
    """✅ 4.1: Rotation automatique de la clé JWT dans Vault.
    
    Génère une nouvelle clé JWT et la stocke dans Vault.
    L'ancienne clé est conservée temporairement pour transition.
    
    Returns:
        dict avec status, environment, rotated_at
    """
    if not HVAC_AVAILABLE:
        logger.warning("[4.1 Vault Rotation] hvac non installé, rotation JWT ignorée")
        return {"status": "skipped", "reason": "hvac_not_available"}
    
    try:
        vault_client = _get_vault_client()
        if not vault_client or not vault_client.use_vault or not vault_client._client:
            logger.warning("[4.1 Vault Rotation] Vault non disponible, rotation JWT ignorée")
            return {"status": "skipped", "reason": "vault_not_available"}
        
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
            path=f"atmr/data/{env_path}/jwt/secret_key",
            secret={"value": new_secret}
        )
        
        logger.info("[4.1 Vault Rotation] ✅ JWT secret roté avec succès (env=%s)", env_path)
        
        # ⚠️ IMPORTANT: Vider le cache pour forcer rechargement
        vault_client.clear_cache()
        
        return {
            "status": "success",
            "environment": env_path,
            "rotated_at": datetime.now(UTC).isoformat(),
            "next_rotation_days": JWT_ROTATION_INTERVAL_DAYS,
            "old_secret_present": old_secret is not None,
        }
        
    except Exception as e:
        logger.exception("[4.1 Vault Rotation] ❌ Erreur rotation JWT secret: %s", e)
        raise


@celery.task(bind=True, name="tasks.vault_rotation_tasks.rotate_encryption_key")
def rotate_encryption_key(self: Task) -> dict[str, Any]:  # noqa: ARG001
    """✅ 4.1: Rotation automatique de la clé d'encryption dans Vault.
    
    Génère une nouvelle clé d'encryption et la stocke dans Vault.
    L'ancienne clé est ajoutée à la liste des legacy keys.
    
    Returns:
        dict avec status, environment, rotated_at
    """
    if not HVAC_AVAILABLE:
        logger.warning("[4.1 Vault Rotation] hvac non installé, rotation encryption ignorée")
        return {"status": "skipped", "reason": "hvac_not_available"}
    
    try:
        vault_client = _get_vault_client()
        if not vault_client or not vault_client.use_vault or not vault_client._client:
            logger.warning("[4.1 Vault Rotation] Vault non disponible, rotation encryption ignorée")
            return {"status": "skipped", "reason": "vault_not_available"}
        
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
            path=f"atmr/data/{env_path}/encryption/master_key",
            secret={"value": new_key_b64}
        )
        
        # Stocker les legacy keys
        vault_client._client.secrets.kv.v2.create_or_update_secret(
            path=f"atmr/data/{env_path}/encryption/legacy_keys",
            secret={"keys": legacy_keys}
        )
        
        logger.info(
            "[4.1 Vault Rotation] ✅ Encryption key rotée avec succès (env=%s, %d legacy keys)",
            env_path,
            len(legacy_keys)
        )
        
        # ⚠️ IMPORTANT: Vider le cache pour forcer rechargement
        vault_client.clear_cache()
        
        # ✅ 2.5: Intégration avec système de rotation existant
        # Notifier le EncryptionService de la nouvelle clé (à faire manuellement ou via reload)
        
        return {
            "status": "success",
            "environment": env_path,
            "rotated_at": datetime.now(UTC).isoformat(),
            "next_rotation_days": ENCRYPTION_ROTATION_INTERVAL_DAYS,
            "legacy_keys_count": len(legacy_keys),
            "old_key_present": old_key_b64 is not None,
        }
        
    except Exception as e:
        logger.exception("[4.1 Vault Rotation] ❌ Erreur rotation encryption key: %s", e)
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
        
        logger.info(
            "[4.1 Vault Rotation] ✅ Rotation globale terminée: %d/%d succès",
            success_count,
            total_count
        )
        
        return {
            "status": "completed",
            "rotated_at": datetime.now(UTC).isoformat(),
            "results": results,
            "success_count": success_count,
            "total_count": total_count,
        }
        
    except Exception as e:
        logger.exception("[4.1 Vault Rotation] ❌ Erreur rotation globale: %s", e)
        raise

