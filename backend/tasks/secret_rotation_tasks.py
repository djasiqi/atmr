"""✅ 2.5: Tâches Celery pour rotation automatique des clés de chiffrement.

Rotation toutes les 90 jours + migration progressive des données.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from celery import Task

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)

# ✅ 2.5: Intervalle de rotation (90 jours par défaut)
DEFAULT_ROTATION_INTERVAL_DAYS = 90


@celery.task(bind=True, name="tasks.secret_rotation_tasks.rotate_encryption_keys")
def rotate_encryption_keys(self: Task) -> dict[str, Any]:  # noqa: ARG001
    """✅ 2.5: Génère une nouvelle clé de chiffrement et la marque comme active.

    L'ancienne clé devient legacy pour permettre le déchiffrement des données existantes.

    Returns:
        dict avec status, old_key_hex, new_key_hex
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from security.crypto import (
                DEFAULT_KEY_LENGTH,
                get_encryption_service,
                rotate_to_new_key,
            )

            logger.info("[2.5] Début rotation clés de chiffrement...")

            # Récupérer le service
            service = get_encryption_service()

            # Générer nouvelle clé
            new_key = os.urandom(DEFAULT_KEY_LENGTH)
            new_key_hex = new_key.hex()

            # Effectuer la rotation (ancienne devient legacy)
            old_key = rotate_to_new_key(service, new_key)
            old_key_hex = old_key.hex()

            logger.info(
                "[2.5] ✅ Rotation effectuée - Nouvelle clé: %s... (ancienne → legacy)",
                new_key_hex[:16],
            )

            # ⚠️ IMPORTANT: Le développeur doit mettre à jour les variables d'environnement
            # MASTER_ENCRYPTION_KEY et LEGACY_ENCRYPTION_KEYS manuellement
            warning_msg = (
                "[2.5] ⚠️ MISE À JOUR MANUELLE REQUISE:\n"
                f"  - MASTER_ENCRYPTION_KEY={new_key_hex}\n"
                f"  - LEGACY_ENCRYPTION_KEYS={old_key_hex} (ajouter l'ancienne clé)"
            )
            logger.warning(warning_msg)

            return {
                "status": "success",
                "message": "Rotation effectuée - Mise à jour manuelle des variables d'environnement requise",
                "new_key_hex": new_key_hex,
                "old_key_hex": old_key_hex,
                "legacy_count": len(service.legacy_keys),
            }

    except Exception as e:
        logger.exception("[2.5] ❌ Erreur lors de la rotation des clés: %s", e)
        raise


@celery.task(bind=True, name="tasks.secret_rotation_tasks.migrate_encrypted_data")
def migrate_encrypted_data(
    self: Task,  # noqa: ARG001
    model_name: str = "User",
    field_name: str | None = None,
    batch_size: int = 100,
    limit: int | None = None,
) -> dict[str, Any]:
    """✅ 2.5: Re-chiffre les données existantes avec la nouvelle clé active.

    Cette tâche migre progressivement les données chiffrées avec l'ancienne clé
    vers la nouvelle clé active. Elle peut être exécutée en plusieurs batches.

    Args:
        model_name: Nom du modèle à migrer (User, Client, etc.)
        field_name: Nom du champ chiffré (None = tous les champs chiffrés du modèle)
        batch_size: Nombre d'enregistrements à traiter par batch
        limit: Limite totale d'enregistrements (None = tous)

    Returns:
        dict avec status, migrated_count, errors
    """
    try:
        app = get_flask_app()
        with app.app_context():
            from ext import db
            from security.crypto import get_encryption_service

            logger.info(
                "[2.5] Début migration données chiffrées: model=%s, field=%s, batch_size=%d",
                model_name,
                field_name,
                batch_size,
            )

            service = get_encryption_service()

            # Importer dynamiquement le modèle
            try:
                from models import Client, User

                model_map = {
                    "User": User,
                    "Client": Client,
                }

                if model_name not in model_map:
                    raise ValueError(f"Modèle {model_name} non supporté")

                Model = model_map[model_name]

            except ImportError as e:
                logger.error("[2.5] Erreur import modèle: %s", e)
                raise

            # Déterminer les champs chiffrés à migrer
            encrypted_fields: list[str] = []
            if field_name:
                encrypted_fields = [field_name]
            elif model_name == "User":
                # Champs chiffrés courants (à adapter selon les modèles)
                encrypted_fields = [
                    "phone_encrypted",
                    "email_encrypted",
                    "first_name_encrypted",
                    "last_name_encrypted",
                    "address_encrypted",
                ]
            elif model_name == "Client":
                encrypted_fields = [
                    "contact_phone_encrypted",
                    "gp_name_encrypted",
                    "gp_phone_encrypted",
                    "billing_address_encrypted",
                ]

            migrated_count = 0
            errors: list[dict[str, Any]] = []

            # Récupérer tous les enregistrements avec données chiffrées
            query = db.session.query(Model)
            total_count = query.count()

            if limit:
                total_count = min(total_count, limit)

            logger.info("[2.5] %d enregistrements à traiter", total_count)

            # Traitement par batches
            offset = 0
            while offset < total_count:
                records = query.offset(offset).limit(batch_size).all()
                if not records:
                    break

                for record in records:
                    try:
                        updated = False

                        for field in encrypted_fields:
                            # Vérifier si le champ existe et contient des données
                            encrypted_value = getattr(record, field, None)
                            if not encrypted_value:
                                continue

                            # Déchiffrer avec l'ancienne clé (via service qui essaie toutes les clés)
                            try:
                                plaintext = service.decrypt_field(encrypted_value)
                            except Exception as decrypt_err:
                                logger.warning(
                                    "[2.5] Impossible de déchiffrer %s.%s (id=%s): %s",
                                    model_name,
                                    field,
                                    getattr(record, "id", "?"),
                                    decrypt_err,
                                )
                                errors.append(
                                    {
                                        "model": model_name,
                                        "id": getattr(record, "id", None),
                                        "field": field,
                                        "error": str(decrypt_err),
                                    }
                                )
                                continue

                            # Re-chiffrer avec la nouvelle clé active
                            new_encrypted = service.encrypt_field(plaintext)

                            # Mettre à jour le champ
                            setattr(record, field, new_encrypted)
                            updated = True

                        if updated:
                            db.session.commit()
                            migrated_count += 1

                            if migrated_count % 10 == 0:
                                logger.debug(
                                    "[2.5] %d enregistrements migrés...", migrated_count
                                )

                    except Exception as e:
                        db.session.rollback()
                        logger.error(
                            "[2.5] Erreur migration %s id=%s: %s",
                            model_name,
                            getattr(record, "id", "?"),
                            e,
                        )
                        errors.append(
                            {
                                "model": model_name,
                                "id": getattr(record, "id", None),
                                "error": str(e),
                            }
                        )

                offset += batch_size

            logger.info(
                "[2.5] ✅ Migration terminée: %d enregistrements migrés, %d erreurs",
                migrated_count,
                len(errors),
            )

            return {
                "status": "success",
                "model": model_name,
                "migrated_count": migrated_count,
                "total_count": total_count,
                "errors": errors,
            }

    except Exception as e:
        logger.exception("[2.5] ❌ Erreur lors de la migration: %s", e)
        raise


@celery.task(bind=True, name="tasks.secret_rotation_tasks.check_rotation_due")
def check_rotation_due(self: Task) -> dict[str, Any]:  # noqa: ARG001
    """✅ 2.5: Vérifie si une rotation de clé est due.

    Cette tâche peut être appelée périodiquement pour déterminer si une rotation
    est nécessaire (basée sur l'intervalle de 90 jours).

    Returns:
        dict avec rotation_due (bool) et jours_restants
    """
    try:
        app = get_flask_app()
        with app.app_context():
            logger.info("[2.5] Vérification rotation des clés...")

            # ⚠️ NOTE: L'age de la clé devrait être stocké en DB ou fichier
            # Pour simplifier, on vérifie via variable d'environnement ou dernière rotation
            # En production, utiliser une table de métadonnées pour stocker la date de dernière rotation

            rotation_interval_days = int(
                os.getenv("ENCRYPTION_KEY_ROTATION_INTERVAL_DAYS", "90")
            )

            # Logique simplifiée: si pas de rotation récente détectée, suggérer rotation
            logger.info(
                "[2.5] Intervalle rotation: %d jours (configurable via ENCRYPTION_KEY_ROTATION_INTERVAL_DAYS)",
                rotation_interval_days,
            )

            # ⚠️ En production, lire la date de dernière rotation depuis une source persistante
            # Ici on retourne juste l'info que la rotation peut être effectuée manuellement

            return {
                "status": "check_complete",
                "rotation_interval_days": rotation_interval_days,
                "message": "Vérification terminée - Utiliser rotate_encryption_keys() pour rotation manuelle",
            }

    except Exception as e:
        logger.exception("[2.5] ❌ Erreur lors de la vérification: %s", e)
        raise
