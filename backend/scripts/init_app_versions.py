#!/usr/bin/env python3
"""Script d'initialisation/mise à jour des configurations de version d'application.

Ce script permet de:
- Initialiser les configurations de version pour Android et iOS
- Mettre à jour les versions minimales/recommandées
- Configurer les URLs des stores

Usage:
    python scripts/init_app_versions.py
    python scripts/init_app_versions.py --android-min 1.2.0 --android-latest 1.3.0
    python scripts/init_app_versions.py --ios-min 1.1.0 --ios-latest 1.2.0
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from ext import db
from models.app_version_config import AppVersionConfig


def init_or_update_config(
    platform: str,
    min_required: str,
    latest: str,
    store_url: str | None = None,
    message: str | None = None,
) -> AppVersionConfig:
    """Initialise ou met à jour une configuration de version."""
    config = AppVersionConfig.query.filter_by(platform=platform).first()

    if config:
        # Mise à jour
        config.min_required_version = min_required
        config.latest_version = latest
        if store_url:
            config.store_url = store_url
        if message:
            config.update_message = message
        print(f"✅ Configuration {platform} mise à jour")
    else:
        # Création
        config = AppVersionConfig()
        config.platform = platform  # type: ignore[assignment]
        config.min_required_version = min_required  # type: ignore[assignment]
        config.latest_version = latest  # type: ignore[assignment]
        if store_url:
            config.store_url = store_url  # type: ignore[assignment]
        if message:
            config.update_message = message  # type: ignore[assignment]
        db.session.add(config)
        print(f"✅ Configuration {platform} créée")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Initialise ou met à jour les configurations de version d'application"
    )
    parser.add_argument(
        "--android-min",
        type=str,
        help="Version minimale requise pour Android (ex: 1.2.0)",
    )
    parser.add_argument(
        "--android-latest",
        type=str,
        help="Dernière version disponible pour Android (ex: 1.3.0)",
    )
    parser.add_argument(
        "--android-store-url",
        type=str,
        help="URL du Play Store pour Android",
        default="https://play.google.com/store/apps/details?id=com.drinjasiqi.atmr",
    )
    parser.add_argument(
        "--ios-min",
        type=str,
        help="Version minimale requise pour iOS (ex: 1.1.0)",
    )
    parser.add_argument(
        "--ios-latest",
        type=str,
        help="Dernière version disponible pour iOS (ex: 1.2.0)",
    )
    parser.add_argument(
        "--ios-store-url",
        type=str,
        help="URL de l'App Store pour iOS",
    )

    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        # Valeurs par défaut (basées sur package.json: 1.0.3)
        android_min = args.android_min or "1.0.0"
        android_latest = args.android_latest or "1.0.3"
        ios_min = args.ios_min or "1.0.0"
        ios_latest = args.ios_latest or "1.0.3"

        print("📱 Initialisation des configurations de version...")
        print(f"   Android: min={android_min}, latest={android_latest}")
        print(f"   iOS: min={ios_min}, latest={ios_latest}")

        # Android
        init_or_update_config(
            platform="android",
            min_required=android_min,
            latest=android_latest,
            store_url=args.android_store_url,
        )

        # iOS
        init_or_update_config(
            platform="ios",
            min_required=ios_min,
            latest=ios_latest,
            store_url=args.ios_store_url,
        )

        # Commit
        try:
            db.session.commit()
            print("\n✅ Configurations sauvegardées avec succès!")
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Erreur lors de la sauvegarde: {e}")
            sys.exit(1)

        # Afficher les configurations finales
        print("\n📋 Configurations actuelles:")
        for platform in ["android", "ios"]:
            config = AppVersionConfig.query.filter_by(platform=platform).first()
            if config:
                print(f"\n{platform.upper()}:")
                print(f"  Min requise: {config.min_required_version}")
                print(f"  Dernière: {config.latest_version}")
                print(f"  Store URL: {config.store_url or '(non configuré)'}")
            else:
                print(f"\n{platform.upper()}: (non configuré)")


if __name__ == "__main__":
    main()
