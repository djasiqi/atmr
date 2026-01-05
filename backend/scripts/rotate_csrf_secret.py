#!/usr/bin/env python3
"""✅ S3: Script pour rotation manuelle de la clé CSRF.

Usage:
    python backend/scripts/rotate_csrf_secret.py

Génère une nouvelle clé CSRF et affiche les instructions pour la mettre à jour.
"""

import secrets
import sys
from pathlib import Path

# Ajouter le répertoire backend au PYTHONPATH
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def generate_csrf_secret() -> str:
    """Génère une nouvelle clé CSRF sécurisée.

    Returns:
        Clé CSRF en format URL-safe (base64, 64 bytes)
    """
    return secrets.token_urlsafe(64)


def main() -> None:
    """Point d'entrée principal."""
    print("=" * 80)
    print("ROTATION DE LA CLÉ CSRF")
    print("=" * 80)
    print()

    # Générer nouvelle clé
    new_secret = generate_csrf_secret()

    print("✅ Nouvelle clé CSRF générée :")
    print()
    print(f"CSRF_SECRET_KEY={new_secret}")
    print()
    print("=" * 80)
    print("INSTRUCTIONS :")
    print("=" * 80)
    print()
    print("⚠️  NOTE IMPORTANTE :")
    print("   Le système CSRF utilise actuellement JWT_SECRET_KEY ou SECRET_KEY")
    print("   (pas de CSRF_SECRET_KEY dédiée).")
    print()
    print("1. Mettre à jour la variable d'environnement JWT_SECRET_KEY ou SECRET_KEY")
    print("   - Dans votre fichier .env ou gestionnaire de secrets")
    print("   - Ou via Vault si configuré")
    print()
    print("2. Redémarrer l'application")
    print()
    print("3. Les clients devront récupérer un nouveau token CSRF")
    print("   - Endpoint : GET /api/v1/auth/csrf-token")
    print()
    print("⚠️  IMPORTANT :")
    print("   - Tous les tokens CSRF existants deviendront invalides")
    print("   - La rotation de JWT_SECRET_KEY affectera aussi tous les tokens JWT")
    print("   - Les utilisateurs devront se reconnecter")
    print("   - Planifier la rotation pendant une période de faible trafic")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
