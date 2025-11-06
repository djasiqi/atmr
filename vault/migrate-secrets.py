#!/usr/bin/env python3
"""✅ 4.1: Script de migration des secrets depuis .env vers Vault.

Usage:
    python vault/migrate-secrets.py --env-file backend/.env --vault-addr http://localhost:8200 --vault-token <token>
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import hvac  # type: ignore[import-untyped]  # Dépendance optionnelle
except ImportError:
    print("❌ hvac non installé. Installer avec: pip install hvac")
    sys.exit(1)


# ✅ 4.1: Mapping des secrets .env vers paths Vault
SECRETS_MAPPING = {
    # Flask
    "SECRET_KEY": {
        "path": "dev/flask/secret_key",
        "key": "value",
        "description": "Flask secret key",
    },
    # JWT
    "JWT_SECRET_KEY": {
        "path": "dev/jwt/secret_key",
        "key": "value",
        "description": "JWT secret key",
    },
    # Encryption
    "APP_ENCRYPTION_KEY_B64": {
        "path": "dev/encryption/master_key",
        "key": "value",
        "description": "Encryption master key (Base64)",
    },
    "ENCRYPTION_KEY_HEX": {
        "path": "dev/encryption/legacy_key_hex",
        "key": "value",
        "description": "Legacy encryption key (hex)",
    },
    # Database
    "DATABASE_URL": {
        "path": "dev/database/url",
        "key": "value",
        "description": "Database connection URL",
    },
    # Mail
    "MAIL_PASSWORD": {
        "path": "dev/mail/password",
        "key": "value",
        "description": "Mail password",
    },
    # Redis
    "REDIS_URL": {
        "path": "dev/redis/url",
        "key": "value",
        "description": "Redis connection URL",
    },
    # API Externes
    "GOOGLE_PLACES_API_KEY": {
        "path": "dev/api/google_places",
        "key": "value",
        "description": "Google Places API key",
    },
    "OPENWEATHER_API_KEY": {
        "path": "dev/api/openweather",
        "key": "value",
        "description": "OpenWeather API key",
    },
    # Monitoring
    "SENTRY_DSN": {
        "path": "dev/monitoring/sentry",
        "key": "value",
        "description": "Sentry DSN",
    },
}


def load_env_file(env_file: Path) -> dict[str, str]:
    """Charge un fichier .env et retourne un dict."""
    secrets = {}
    if not env_file.exists():
        print(f"⚠️  Fichier .env non trouvé: {env_file}")
        return secrets
    
    with env_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"\'')  # Enlever quotes
            
            if key in SECRETS_MAPPING:
                secrets[key] = value
    
    return secrets


def migrate_secrets(
    client: hvac.Client,
    secrets: dict[str, str],
    environment: str = "dev",
    dry_run: bool = False,
) -> None:
    """Migre les secrets vers Vault.
    
    Args:
        client: Client Vault
        secrets: Dict des secrets à migrer
        environment: Environnement (dev, staging, prod)
        dry_run: Si True, affiche seulement sans écrire
    """
    migrated = 0
    errors = 0
    
    print(f"\n🔐 Migration des secrets vers Vault (environnement: {environment})")
    print("=" * 80)
    
    for env_key, value in secrets.items():
        if env_key not in SECRETS_MAPPING:
            continue
        
        mapping = SECRETS_MAPPING[env_key]
        vault_path = f"atmr/{environment}/{mapping['path']}"
        key = mapping["key"]
        description = mapping.get("description", "")
        
        print(f"\n📝 {env_key}")
        print(f"   Path Vault: {vault_path}")
        print(f"   Description: {description}")
        print(f"   Valeur: {'*' * min(len(value), 20)}...")
        
        if not dry_run:
            try:
                # KV v2: utiliser data/ dans le path
                vault_data_path = f"atmr/data/{environment}/{mapping['path']}"
                
                client.secrets.kv.v2.create_or_update_secret(
                    path=vault_data_path,
                    secret={key: value}
                )
                
                print("   ✅ Migré avec succès")
                migrated += 1
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                errors += 1
        else:
            print("   🔍 [DRY RUN] Serait migré")
            migrated += 1
    
    print("\n" + "=" * 80)
    print(f"✅ {migrated} secret(s) migré(s)")
    if errors > 0:
        print(f"❌ {errors} erreur(s)")
    
    if dry_run:
        print("\n💡 Exécuter sans --dry-run pour effectuer la migration")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Migre les secrets depuis .env vers Vault"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("backend/.env"),
        help="Fichier .env à migrer (défaut: backend/.env)",
    )
    parser.add_argument(
        "--vault-addr",
        default=os.getenv("VAULT_ADDR", "http://localhost:8200"),
        help="Adresse Vault (défaut: VAULT_ADDR ou http://localhost:8200)",
    )
    parser.add_argument(
        "--vault-token",
        default=os.getenv("VAULT_TOKEN"),
        help="Token Vault (défaut: VAULT_TOKEN)",
    )
    parser.add_argument(
        "--environment",
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environnement cible (défaut: dev)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche seulement sans écrire dans Vault",
    )
    
    args = parser.parse_args()
    
    # Validation
    if not args.vault_token:
        print("❌ VAULT_TOKEN requis (--vault-token ou variable d'environnement)")
        sys.exit(1)
    
    # Charger secrets depuis .env
    print(f"📂 Chargement secrets depuis: {args.env_file}")
    secrets = load_env_file(args.env_file)
    
    if not secrets:
        print("⚠️  Aucun secret trouvé à migrer")
        sys.exit(0)
    
    print(f"✅ {len(secrets)} secret(s) trouvé(s)")
    
    # Initialiser client Vault
    try:
        client = hvac.Client(url=args.vault_addr)
        client.token = args.vault_token
        
        if not client.is_authenticated():
            print("❌ Authentification Vault échouée")
            sys.exit(1)
        
        print(f"✅ Connecté à Vault: {args.vault_addr}")
    except Exception as e:
        print(f"❌ Erreur connexion Vault: {e}")
        sys.exit(1)
    
    # Migrer
    migrate_secrets(client, secrets, environment=args.environment, dry_run=args.dry_run)
    
    print("\n✅ Migration terminée")


if __name__ == "__main__":
    main()

