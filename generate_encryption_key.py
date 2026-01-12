#!/usr/bin/env python3
"""
Script de génération de clé d'encryption pour ATMR.

Ce script génère une clé AES-256 sécurisée en base64 et la configure
dans le fichier backend/.env si elle n'existe pas déjà.

Usage:
    python generate_encryption_key.py

    # OU avec force pour régénérer une nouvelle clé
    python generate_encryption_key.py --force
"""

import base64
import secrets
import sys
from pathlib import Path


def generate_encryption_key() -> str:
    """Génère une clé AES-256 (32 bytes) sécurisée en base64.

    Returns:
        Clé en format base64
    """
    key_bytes = secrets.token_bytes(32)  # 256 bits
    key_b64 = base64.b64encode(key_bytes).decode("utf-8")
    return key_b64


def check_env_file_exists(env_path: Path) -> bool:
    """Vérifie si le fichier .env existe.

    Args:
        env_path: Chemin vers le fichier .env

    Returns:
        True si le fichier existe, False sinon
    """
    return env_path.exists()


def check_key_in_env(env_path: Path) -> bool:
    """Vérifie si APP_ENCRYPTION_KEY_B64 est déjà définie dans .env.

    Args:
        env_path: Chemin vers le fichier .env

    Returns:
        True si la clé existe et est non vide, False sinon
    """
    if not env_path.exists():
        return False

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("APP_ENCRYPTION_KEY_B64="):
                value = line.split("=", 1)[1].strip()
                # Vérifier que la valeur n'est pas vide et ne contient pas de placeholder
                if (
                    value
                    and not value.startswith("CHANGE_ME")
                    and not value.startswith("TODO")
                ):
                    return True
    return False


def add_key_to_env(env_path: Path, key: str, force: bool = False) -> None:
    """Ajoute APP_ENCRYPTION_KEY_B64 au fichier .env.

    Args:
        env_path: Chemin vers le fichier .env
        key: Clé d'encryption en base64
        force: Si True, remplace la clé existante
    """
    if not env_path.exists():
        # Créer le fichier .env avec un header
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# Configuration ATMR - Backend\n")
            f.write("# Généré automatiquement\n\n")
            f.write(f"APP_ENCRYPTION_KEY_B64={key}\n")
        print(f"✅ Fichier {env_path} créé avec la clé d'encryption")
        return

    # Lire le contenu existant
    with open(env_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Vérifier si la clé existe déjà
    key_found = False
    new_lines = []

    for line in lines:
        if line.strip().startswith("APP_ENCRYPTION_KEY_B64="):
            if force:
                new_lines.append(f"APP_ENCRYPTION_KEY_B64={key}\n")
                print("⚠️  Clé d'encryption existante remplacée (--force)")
            else:
                new_lines.append(line)
            key_found = True
        else:
            new_lines.append(line)

    # Si la clé n'existe pas, l'ajouter
    if not key_found:
        # Chercher la section sécurité ou l'ajouter à la fin
        inserted = False
        for i, line in enumerate(new_lines):
            if "# Sécurité" in line or "# Security" in line:
                # Insérer après cette ligne
                new_lines.insert(i + 1, f"APP_ENCRYPTION_KEY_B64={key}\n")
                inserted = True
                break

        if not inserted:
            # Ajouter à la fin avec une section
            if not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append("\n# Sécurité\n")
            new_lines.append(f"APP_ENCRYPTION_KEY_B64={key}\n")

    # Écrire le nouveau contenu
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"✅ Clé d'encryption ajoutée à {env_path}")


def main() -> int:
    """Point d'entrée principal.

    Returns:
        Code de sortie (0 = succès, 1 = erreur)
    """
    force = "--force" in sys.argv or "-f" in sys.argv

    print("🔐 Générateur de Clé d'Encryption ATMR")
    print("=" * 50)

    # Trouver le répertoire backend
    script_dir = Path(__file__).parent
    backend_dir = (
        script_dir / "backend" if (script_dir / "backend").exists() else script_dir
    )
    env_path = backend_dir / ".env"

    print(f"📂 Répertoire backend: {backend_dir}")
    print(f"📄 Fichier .env: {env_path}")
    print()

    # Vérifier si la clé existe déjà
    if not force and check_key_in_env(env_path):
        print("✅ APP_ENCRYPTION_KEY_B64 existe déjà dans .env")
        print("ℹ️  Utilisez --force pour générer une nouvelle clé")
        return 0

    # Générer une nouvelle clé
    print("🔑 Génération d'une nouvelle clé AES-256...")
    key = generate_encryption_key()
    print(f"✅ Clé générée: {key[:16]}...{key[-16:]}")
    print()

    # Ajouter la clé au fichier .env
    try:
        add_key_to_env(env_path, key, force=force)
        print()
        print("=" * 50)
        print("✅ Configuration terminée avec succès!")
        print()
        print("⚠️  IMPORTANT:")
        print("   1. Ne partagez JAMAIS cette clé")
        print("   2. Sauvegardez-la dans un gestionnaire de secrets")
        print("   3. En production, utilisez Vault ou AWS Secrets Manager")
        print()
        print("🚀 Vous pouvez maintenant démarrer les services:")
        print("   docker-compose up -d")
        print()
        return 0
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
