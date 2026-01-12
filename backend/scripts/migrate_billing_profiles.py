#!/usr/bin/env python3
"""Script de migration des données vers CompanyBillingProfile.

Ce script crée un profil de facturation centralisé pour chaque entreprise
à partir des données existantes dans les tables `company` et `company_billing_settings`.

Usage:
    python backend/scripts/migrate_billing_profiles.py [--dry-run]
"""

import logging
import sys
from decimal import Decimal
from pathlib import Path

from sqlalchemy.exc import IntegrityError

# Ajouter le répertoire backend au PYTHONPATH
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# ruff: noqa: E402
from ext import db
from models import Company, CompanyBillingProfile, CompanyBillingSettings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_address_line(address_line: str) -> tuple[str, str]:
    """Parse une ligne d'adresse pour extraire rue et numéro.

    Args:
        address_line: Ligne d'adresse complète (ex: "Rue Verte 8")

    Returns:
        tuple[str, str]: (nom_rue, numéro_bâtiment)
    """
    if not address_line:
        return ("[Non configuré]", "0")

    # Essayer de séparer le numéro à la fin
    parts = address_line.strip().rsplit(" ", 1)
    if len(parts) == 2:
        street, number = parts
        # Vérifier si le dernier élément est un numéro (peut contenir des lettres: "8A")
        if any(char.isdigit() for char in number):
            return (street.strip(), number.strip())

    # Si pas de numéro trouvé, mettre toute l'adresse comme rue
    return (address_line.strip(), "0")


def migrate_company_billing_profiles(dry_run: bool = False) -> dict[str, int]:
    """Migre les données vers CompanyBillingProfile.

    Args:
        dry_run: Si True, n'effectue pas les modifications en base

    Returns:
        dict: Statistiques de migration
    """
    stats = {
        "total_companies": 0,
        "profiles_created": 0,
        "profiles_skipped": 0,
        "errors": 0,
    }

    try:
        # Récupérer toutes les entreprises
        companies = Company.query.all()
        stats["total_companies"] = len(companies)

        logger.info("=" * 80)
        logger.info("🚀 Démarrage migration profils de facturation")
        logger.info("   Mode: %s", "DRY-RUN (simulation)" if dry_run else "PRODUCTION")
        logger.info("   Entreprises à traiter: %s", stats["total_companies"])
        logger.info("=" * 80)

        for company in companies:
            logger.info("\n📋 Traitement: %s (ID: %s)", company.name, company.id)

            # Vérifier si un profil existe déjà
            existing_profile = CompanyBillingProfile.query.filter_by(
                company_id=company.id
            ).first()

            if existing_profile:
                logger.info(
                    "   ⏭️  Profil existant trouvé (ID: %s) - SKIP", existing_profile.id
                )
                stats["profiles_skipped"] += 1
                continue

            # Récupérer les paramètres de facturation existants
            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=company.id
            ).first()

            # === CONSTRUCTION DU PROFIL ===

            # 1. Identité légale
            legal_name = company.name or "[Non configuré]"
            uid_ide = company.uid_ide or "[Non configuré]"

            # 2. Adresse structurée (depuis company.domicile_*)
            street_name, building_number = "[Non configuré]", "0"
            postal_code = "[Non configuré]"
            city = "[Non configuré]"

            if company.domicile_address_line1:
                street_name, building_number = parse_address_line(
                    company.domicile_address_line1
                )

            if company.domicile_zip:
                postal_code = company.domicile_zip

            if company.domicile_city:
                city = company.domicile_city

            # 3. Contact facturation
            billing_email = company.billing_email or "[Non configuré]"
            billing_phone = company.contact_phone or "[Non configuré]"

            # 4. TVA
            vat_registered = False
            vat_number = None
            vat_rate = None

            if billing_settings:
                vat_registered = bool(billing_settings.vat_applicable)
                vat_number = billing_settings.vat_number
                vat_rate = billing_settings.vat_rate

            # 5. IBAN (en clair pour l'instant, à chiffrer si nécessaire)
            iban = company.iban or "[Non configuré]"

            # 6. Mode de paiement (par défaut SCOR)
            payment_reference_mode = "SCOR"

            # 7. Paramètres de facturation
            payment_terms_days = 30
            overdue_fee = Decimal("15.00")

            if billing_settings:
                if billing_settings.payment_terms_days is not None:
                    payment_terms_days = billing_settings.payment_terms_days
                if billing_settings.overdue_fee is not None:
                    overdue_fee = Decimal(str(billing_settings.overdue_fee))

            # === CRÉATION DU PROFIL ===

            profile_data = {
                "company_id": company.id,
                "legal_name": legal_name,
                "brand_name": None,  # À remplir manuellement si différent
                "uid_ide": uid_ide,
                "street_name": street_name,
                "building_number": building_number,
                "postal_code": postal_code,
                "city": city,
                "country_code": company.domicile_country or "CH",
                "billing_email": billing_email,
                "billing_phone": billing_phone,
                "vat_registered": vat_registered,
                "vat_number": vat_number,
                "vat_rate": vat_rate,
                "iban": iban,
                "qr_iban": None,  # À remplir si QRR utilisé
                "payment_reference_mode": payment_reference_mode,
                "creditor_reference_base": None,  # Pour QRR
                "payment_terms_days": payment_terms_days,
                "overdue_fee": overdue_fee,
                "legal_footer": None,
                "is_address_validated": False,  # À valider manuellement
            }

            logger.info("   📝 Données du profil:")
            logger.info("      - Nom légal: %s", legal_name)
            logger.info("      - UID: %s", uid_ide)
            logger.info("      - Adresse: %s %s", street_name, building_number)
            logger.info("      - Ville: %s %s", postal_code, city)
            logger.info("      - Email: %s", billing_email)
            logger.info("      - TVA: %s", "Oui" if vat_registered else "Non")
            if len(iban) > 20:
                logger.info("      - IBAN: %s...", iban[:20])
            else:
                logger.info("      - IBAN: %s", iban)

            if dry_run:
                logger.info("   🧪 DRY-RUN: Profil non créé (simulation)")
                stats["profiles_created"] += 1
            else:
                try:
                    new_profile = CompanyBillingProfile(**profile_data)
                    db.session.add(new_profile)
                    db.session.flush()  # Pour obtenir l'ID sans commit
                    logger.info("   ✅ Profil créé (ID: %s)", new_profile.id)
                    stats["profiles_created"] += 1
                except IntegrityError as e:
                    logger.error("   ❌ Erreur IntegrityError: %s", e)
                    db.session.rollback()
                    stats["errors"] += 1
                except Exception as e:
                    logger.error("   ❌ Erreur création profil: %s", e)
                    db.session.rollback()
                    stats["errors"] += 1

        # Commit final si pas en dry-run
        if not dry_run:
            db.session.commit()
            logger.info("\n💾 Changements commitées en base de données")
        else:
            logger.info("\n🧪 DRY-RUN: Aucune modification en base de données")

    except Exception as e:
        logger.error("\n❌ Erreur globale: %s", e)
        if not dry_run:
            db.session.rollback()
        stats["errors"] += 1

    finally:
        # Afficher les statistiques
        separator = "=" * 80
        logger.info("\n%s", separator)
        logger.info("📊 STATISTIQUES DE MIGRATION")
        logger.info("%s", separator)
        logger.info("   Entreprises totales:     %s", stats["total_companies"])
        logger.info("   Profils créés:           %s", stats["profiles_created"])
        logger.info("   Profils déjà existants:  %s", stats["profiles_skipped"])
        logger.info("   Erreurs:                 %s", stats["errors"])
        logger.info("%s", separator)

        if stats["errors"] > 0:
            logger.warning("⚠️  Des erreurs ont été rencontrées. Vérifiez les logs.")
        elif stats["profiles_created"] > 0:
            logger.info("✅ Migration terminée avec succès !")
        else:
            logger.info("ℹ️  Aucun profil à créer (tous déjà existants)")

    return stats


def main():
    """Point d'entrée principal du script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migration des données vers CompanyBillingProfile"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mode simulation (aucune modification en base)",
    )
    args = parser.parse_args()

    # Import de l'app Flask pour le contexte
    from app import create_app

    app = create_app()

    with app.app_context():
        stats = migrate_company_billing_profiles(dry_run=args.dry_run)

        # Code de sortie
        if stats["errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
