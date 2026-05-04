#!/usr/bin/env python3
"""
Script pour corriger la faute de frappe dans l'adresse email:
noreplay@lirie.ch → noreply@lirie.ch
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire backend au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Charger .env
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    with env_file.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from app import create_app  # noqa: E402
from ext import db  # noqa: E402
from models.invoice import CompanyBillingSettings  # noqa: E402

print("=" * 80)
print("CORRECTION ADRESSE EMAIL - LIRIE SA")
print("=" * 80)
print()

# Créer l'application Flask
app = create_app()

with app.app_context():
    print("Recherche de la configuration email pour LIRIE SA...")
    print()

    # Trouver tous les CompanyBillingSettings avec noreplay@lirie.ch
    settings = CompanyBillingSettings.query.filter(
        CompanyBillingSettings.smtp_username.like("%noreplay@lirie.ch%")
    ).all()

    if not settings:
        print("Aucune configuration trouvee avec 'noreplay@lirie.ch'")
        print()

        # Vérifier si noreply existe déjà
        correct_settings = CompanyBillingSettings.query.filter(
            CompanyBillingSettings.smtp_username.like("%noreply@lirie.ch%")
        ).all()

        if correct_settings:
            print("OK - L'adresse correcte 'noreply@lirie.ch' est deja configuree !")
            for s in correct_settings:
                print(f"   Company ID: {s.company_id}")
                print(f"   Email: {s.smtp_username}")
                print(f"   Nom: {s.from_name}")
                print(f"   Verifie: {s.domain_verified}")
        else:
            print("Aucune configuration email trouvee pour lirie.ch")

        sys.exit(0)

    print(f"Trouve {len(settings)} configuration(s) a corriger:")
    print()

    for setting in settings:
        print(f"Company ID: {setting.company_id}")
        print(f"   Ancien email: {setting.smtp_username}")
        print(f"   Nom: {setting.from_name}")
        print(f"   Verifie: {setting.domain_verified}")
        print()

    # Demander confirmation
    response = input("Corriger 'noreplay' -> 'noreply' ? (oui/non): ").strip().lower()

    if response not in ["oui", "yes", "y", "o"]:
        print("Operation annulee.")
        sys.exit(0)

    print()
    print("Correction en cours...")

    for setting in settings:
        old_email = setting.smtp_username
        new_email = old_email.replace("noreplay@", "noreply@")

        setting.smtp_username = new_email

        print(f"   Company ID {setting.company_id}:")
        print(f"      {old_email} -> {new_email}")

    # Sauvegarder les modifications
    try:
        db.session.commit()
        print()
        print("OK - Modifications sauvegardees avec succes !")
        print()

        print("=" * 80)
        print("IMPORTANT:")
        print("=" * 80)
        print()
        print("Vous devez maintenant RE-VERIFIER le domaine dans Brevo car")
        print("l'adresse d'expediteur a change !")
        print()
        print("Actions:")
        print("1. app.brevo.com -> Senders & Domains -> Domains")
        print("2. Cliquer sur lirie.ch -> [Verify Domain]")
        print()
        print("OU")
        print()
        print("Dans votre interface web:")
        print("Parametres -> Facturation -> Configuration Email")
        print("-> [Verifier la configuration DNS]")
        print()

    except Exception as e:
        db.session.rollback()
        print()
        print(f"ERREUR - Echec de la sauvegarde: {e}")
        sys.exit(1)

print("=" * 80)
print("TERMINE")
print("=" * 80)
