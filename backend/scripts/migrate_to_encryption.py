#!/usr/bin/env python3
"""✅ D2: Migration des données existantes vers chiffrement.

Usage: python -m scripts.migrate_to_encryption [--dry-run] [--batch-size N]
"""

import argparse
import logging
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from ext import db
from models.client import Client
from models.user import User
from security.crypto import get_encryption_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def migrate_users(dry_run: bool = False, batch_size: int = 100):
    """Migre les utilisateurs vers le chiffrement."""
    try:
        get_encryption_service()  # Vérifier que le service est disponible
    except Exception as e:
        logger.error("Impossible de charger le service de chiffrement: %s", e)
        return 0

    # Compter le total d'utilisateurs à migrer
    total_count = User.query.filter(User.encryption_migrated == False).count()  # noqa: E712
    logger.info("Migration de %d utilisateurs vers le chiffrement...", total_count)

    migrated = 0
    errors = 0
    batch_count = 0
    offset = 0
    chunk_size = 250  # Réduire à 250 pour éviter les timeouts

    # Traiter par chunks pour éviter la saturation mémoire
    while offset < total_count:
        # Charger un chunk d'utilisateurs
        users = User.query.filter(User.encryption_migrated == False).offset(offset).limit(chunk_size).all()  # noqa: E712

        if not users:
            break

        chunk_start = offset
        chunk_updated = 0

        for user in users:
            try:
                updated = False

                # Migrer phone
                if user.phone and not user.phone_encrypted:
                    user.phone_secure = user.phone
                    updated = True

                # Migrer email
                if user.email and not user.email_encrypted:
                    user.email_secure = user.email
                    updated = True

                # Migrer first_name
                if user.first_name and not user.first_name_encrypted:
                    user.first_name_secure = user.first_name
                    updated = True

                # Migrer last_name
                if user.last_name and not user.last_name_encrypted:
                    user.last_name_secure = user.last_name
                    updated = True

                # Migrer address
                if user.address and not user.address_encrypted:
                    user.address_secure = user.address
                    updated = True

                if updated:
                    chunk_updated += 1
                    batch_count += 1
                    # Commit par batch pour améliorer les performances
                    if batch_count >= batch_size and not dry_run:
                        try:
                            db.session.flush()  # Flush avant commit pour libérer mémoire
                            db.session.commit()
                            batch_count = 0
                        except Exception as commit_error:
                            logger.error("Erreur commit batch: %s", commit_error)
                            db.session.rollback()
                            errors += batch_count
                            batch_count = 0

                migrated += 1

                # Log progress tous les batch_size
                if migrated % batch_size == 0:
                    logger.info("Migré %d/%d utilisateurs...", migrated, total_count)

            except Exception as e:
                logger.exception("Erreur migration user %d: %s", user.id, e)
                errors += 1
                if not dry_run:
                    db.session.rollback()

        # Commit final du chunk si nécessaire (avant expunge_all)
        if batch_count > 0 and not dry_run:
            try:
                db.session.flush()
                db.session.commit()
                logger.debug("Commit chunk %d-%d: %d utilisateurs", chunk_start, offset + len(users), batch_count)
                batch_count = 0
            except Exception as commit_error:
                logger.error("Erreur commit chunk: %s", commit_error)
                db.session.rollback()
                errors += batch_count
                batch_count = 0

        # Réinitialiser la session après chaque chunk pour éviter les problèmes de cache
        try:
            db.session.expunge_all()
            # Réinitialiser la connexion périodiquement (tous les 3 chunks)
            if (offset // chunk_size) % 3 == 0:
                db.session.close()
                logger.debug("Session DB fermée/réinitialisée après chunk %d", offset + len(users))
        except Exception as expunge_error:
            logger.warning("Erreur expunge_all (ignorée): %s", expunge_error)

        logger.debug(
            "Chunk %d-%d traité: %d mis à jour, %d total migré",
            chunk_start,
            offset + len(users),
            chunk_updated,
            migrated,
        )
        offset += chunk_size

    # Commit final pour les utilisateurs restants dans le batch
    if batch_count > 0 and not dry_run:
        try:
            db.session.commit()
            logger.info("Commit final de %d utilisateurs...", batch_count)
        except Exception as commit_error:
            logger.error("Erreur commit final: %s", commit_error)
            db.session.rollback()
            errors += batch_count

    logger.info("✅ %d utilisateurs migrés (%d erreurs)", migrated, errors)
    return migrated


def migrate_clients(dry_run: bool = False, batch_size: int = 100):
    """Migre les clients vers le chiffrement."""
    try:
        get_encryption_service()  # Vérifier que le service est disponible
    except Exception as e:
        logger.error("Impossible de charger le service de chiffrement: %s", e)
        return 0

    # Compter le total de clients à migrer
    total_count = Client.query.filter(Client.encryption_migrated == False).count()  # noqa: E712
    logger.info("Migration de %d clients vers le chiffrement...", total_count)

    migrated = 0
    errors = 0
    batch_count = 0
    offset = 0
    chunk_size = 250  # Réduire à 250 pour éviter les timeouts

    # Traiter par chunks pour éviter la saturation mémoire
    while offset < total_count:
        # Charger un chunk de clients
        clients = Client.query.filter(Client.encryption_migrated == False).offset(offset).limit(chunk_size).all()  # noqa: E712

        if not clients:
            break

        chunk_start = offset
        chunk_updated = 0

        for client in clients:
            try:
                updated = False

                # Migrer contact_phone
                if client.contact_phone and not client.contact_phone_encrypted:
                    client.contact_phone_secure = client.contact_phone
                    updated = True

                # Migrer gp_name
                if client.gp_name and not client.gp_name_encrypted:
                    client.gp_name_secure = client.gp_name
                    updated = True

                # Migrer gp_phone
                if client.gp_phone and not client.gp_phone_encrypted:
                    client.gp_phone_secure = client.gp_phone
                    updated = True

                # Migrer billing_address
                if client.billing_address and not client.billing_address_encrypted:
                    client.billing_address_secure = client.billing_address
                    updated = True

                if updated:
                    chunk_updated += 1
                    batch_count += 1
                    # Commit par batch pour améliorer les performances
                    if batch_count >= batch_size and not dry_run:
                        try:
                            db.session.flush()  # Flush avant commit pour libérer mémoire
                            db.session.commit()
                            batch_count = 0
                        except Exception as commit_error:
                            logger.error("Erreur commit batch: %s", commit_error)
                            db.session.rollback()
                            errors += batch_count
                            batch_count = 0

                migrated += 1

                if migrated % batch_size == 0:
                    logger.info("Migré %d/%d clients...", migrated, total_count)

            except Exception as e:
                logger.exception("Erreur migration client %d: %s", client.id, e)
                errors += 1
                if not dry_run:
                    db.session.rollback()

        # Commit final du chunk si nécessaire (avant expunge_all)
        if batch_count > 0 and not dry_run:
            try:
                db.session.flush()
                db.session.commit()
                logger.debug("Commit chunk %d-%d: %d clients", chunk_start, offset + len(clients), batch_count)
                batch_count = 0
            except Exception as commit_error:
                logger.error("Erreur commit chunk: %s", commit_error)
                db.session.rollback()
                errors += batch_count
                batch_count = 0

        # Réinitialiser la session après chaque chunk pour éviter les problèmes de cache
        try:
            db.session.expunge_all()
            # Réinitialiser la connexion périodiquement (tous les 3 chunks)
            if (offset // chunk_size) % 3 == 0:
                db.session.close()
                logger.debug("Session DB fermée/réinitialisée après chunk %d", offset + len(clients))
        except Exception as expunge_error:
            logger.warning("Erreur expunge_all (ignorée): %s", expunge_error)

        logger.debug(
            "Chunk %d-%d traité: %d mis à jour, %d total migré",
            chunk_start,
            offset + len(clients),
            chunk_updated,
            migrated,
        )
        offset += chunk_size

    # Commit final pour les clients restants dans le batch
    if batch_count > 0 and not dry_run:
        try:
            db.session.commit()
            logger.info("Commit final de %d clients...", batch_count)
        except Exception as commit_error:
            logger.error("Erreur commit final: %s", commit_error)
            db.session.rollback()
            errors += batch_count

    logger.info("✅ %d clients migrés (%d erreurs)", migrated, errors)
    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migre les données existantes vers le chiffrement D2")
    parser.add_argument("--dry-run", action="store_true", help="Test sans modifications (compte seulement)")
    parser.add_argument("--batch-size", type=int, default=100, help="Taille du batch pour logging (défaut: 100)")
    parser.add_argument("--users-only", action="store_true", help="Migrer uniquement les utilisateurs")
    parser.add_argument("--clients-only", action="store_true", help="Migrer uniquement les clients")

    args = parser.parse_args()

    logger.info("🚀 Démarrage du script de migration (dry_run=%s, batch_size=%d)", args.dry_run, args.batch_size)

    if args.dry_run:
        logger.info("🔍 Mode DRY-RUN : aucune modification ne sera effectuée")

    # Désactiver Socket.IO, API legacy et routes pour les scripts (évite le blocage)
    import os

    os.environ["SKIP_SOCKETIO"] = "true"
    os.environ["SKIP_ROUTES_INIT"] = "true"  # Skip initialisation routes/handlers (évite blocage)
    os.environ["API_LEGACY_ENABLED"] = "false"  # Évite conflit /specs avec api_v1

    logger.info("🔧 Création de l'application Flask...")
    try:
        logger.debug("Appel de create_app()...")
        app = create_app()
        logger.debug("create_app() terminé avec succès")
        logger.info("✅ Application Flask créée, démarrage de la migration...")
    except SystemExit as e:
        logger.error("❌ SystemExit détecté lors de la création de l'application: %s", e)
        raise
    except KeyboardInterrupt:
        logger.info("⚠️ Interruption clavier lors de la création de l'application")
        raise
    except Exception as e:
        logger.exception("❌ Erreur lors de la création de l'application Flask: %s", e)
        import traceback

        logger.error("Traceback complet:\n%s", traceback.format_exc())
        raise

    try:
        with app.app_context():
            logger.info("✅ Contexte d'application activé")
            total_migrated = 0

            if not args.clients_only:
                logger.info("=" * 60)
                logger.info("MIGRATION DES UTILISATEURS")
                logger.info("=" * 60)
                users_migrated = migrate_users(args.dry_run, args.batch_size)
                total_migrated += users_migrated
                logger.info("✅ Phase utilisateurs terminée: %d migrés", users_migrated)

            if not args.users_only:
                logger.info("=" * 60)
                logger.info("MIGRATION DES CLIENTS")
                logger.info("=" * 60)
                clients_migrated = migrate_clients(args.dry_run, args.batch_size)
                total_migrated += clients_migrated
                logger.info("✅ Phase clients terminée: %d migrés", clients_migrated)

            logger.info("=" * 60)
            if args.dry_run:
                logger.info("📊 DRY-RUN terminé : %d enregistrements seraient migrés", total_migrated)
            else:
                logger.info("✅ Migration terminée : %d enregistrements migrés", total_migrated)
    except Exception as e:
        logger.exception("❌ Erreur lors de l'exécution de la migration: %s", e)
        raise


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("SCRIPT DE MIGRATION - DÉMARRAGE")
        logger.info("=" * 60)
        main()
        logger.info("=" * 60)
        logger.info("SCRIPT DE MIGRATION - TERMINÉ")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.info("⚠️ Migration interrompue par l'utilisateur")
        sys.exit(1)
    except SystemExit as e:
        logger.info("⚠️ SystemExit: code %s", e.code if hasattr(e, "code") else "inconnu")
        raise
    except Exception as e:
        logger.exception("❌ Erreur fatale lors de la migration: %s", e)
        import traceback

        logger.error("Traceback complet:\n%s", traceback.format_exc())
        sys.exit(1)
