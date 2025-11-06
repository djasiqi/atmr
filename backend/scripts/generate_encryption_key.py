#!/usr/bin/env python3
"""✅ D2: Génère une clé d'encryption maître pour MASTER_ENCRYPTION_KEY.

Usage: python -m scripts.generate_encryption_key
"""
import os
import secrets
import sys
from pathlib import Path

# Clé d'encryption doit faire 32 bytes (AES-256) = 64 caractères hex
KEY_LENGTH = 32


def generate_master_key() -> str:
    """Génère une clé maître aléatoire sécurisée en hexadécimal."""
    return secrets.token_hex(KEY_LENGTH)


def main():
    """Génère et affiche la clé d'encryption."""
    print("🔐 Génération d'une clé d'encryption maître (AES-256)...")
    print()
    
    master_key = generate_master_key()
    
    print("✅ Clé générée avec succès:")
    print(f"MASTER_ENCRYPTION_KEY={master_key}")
    print()
    print("📝 Pour l'ajouter à votre configuration:")
    print()
    print("1. Dans backend/.env:")
    print(f"   MASTER_ENCRYPTION_KEY={master_key}")
    print()
    print("2. Dans docker-compose.yml (section api -> environment):")
    print(f"   - MASTER_ENCRYPTION_KEY={master_key}")
    print()
    print("⚠️  IMPORTANT:")
    print("   - Conservez cette clé en sécurité (elle chiffre toutes les données)")
    print("   - Ne la commitez PAS dans Git")
    print("   - Utilisez un gestionnaire de secrets pour la production")
    print()
    
    # Optionnel: ajouter automatiquement au .env si disponible et mode non-interactif
    env_file = Path(__file__).parent.parent / ".env"
    is_interactive = sys.stdin.isatty()
    
    if env_file.exists():
        with env_file.open("r", encoding="utf-8") as f:
            content = f.read()
            if "MASTER_ENCRYPTION_KEY" in content:
                print(f"⚠️  MASTER_ENCRYPTION_KEY existe déjà dans {env_file}")
                print("   Mettez à jour manuellement si nécessaire.")
            elif is_interactive:
                response = input(f"Voulez-vous ajouter cette clé à {env_file}? (o/N): ")
                if response.lower() == "o":
                    with env_file.open("a", encoding="utf-8") as env_file_write:
                        env_file_write.write("\n# ✅ D2: Clé d'encryption maître (AES-256)\n")
                        env_file_write.write(f"MASTER_ENCRYPTION_KEY={master_key}\n")
                    print(f"✅ Clé ajoutée à {env_file}")
    else:
        print(f"⚠️  Fichier {env_file} non trouvé")
        print("   Créez-le et ajoutez la clé manuellement.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

