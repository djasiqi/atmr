#!/usr/bin/env python3
"""Script temporaire pour tester les migrations sans Flask-Migrate.

Ce script contourne le problème avec rich/flask_limiter en utilisant Alembic directement.
"""

import os
import sys

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alembic import command
from alembic.config import Config

# Importer l'application Flask
from app import create_app

# Créer l'application Flask
app = create_app("testing")

# Configurer Alembic
alembic_cfg = Config("migrations/alembic.ini")
alembic_cfg.set_main_option("script_location", "migrations")

if __name__ == "__main__":
    with app.app_context():
        print("=" * 60)
        print("📋 État actuel des migrations:")
        print("=" * 60)
        try:
            command.current(alembic_cfg, verbose=True)
        except Exception as e:
            print(f"⚠️ Erreur lors de la vérification de l'état: {e}")

        print("\n" + "=" * 60)
        print("🔄 Application des migrations (upgrade heads)...")
        print("=" * 60)
        try:
            command.upgrade(alembic_cfg, "heads")
            print("\n✅ Migrations appliquées avec succès!")
        except Exception as e:
            print(f"\n❌ Erreur lors de l'application des migrations: {e}")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("🔍 Vérification des migrations en attente...")
        print("=" * 60)
        print("Pour vérifier qu'il n'y a plus de migrations en attente,")
        print("exécutez: python -m alembic revision --autogenerate -m 'test'")
        print("(ou utilisez Flask-Migrate si rich est corrigé)")
