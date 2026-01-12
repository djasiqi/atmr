#!/usr/bin/env python3
"""Script pour générer la spécification OpenAPI depuis Flask-RESTX.

Usage:
    # Depuis l'hôte (avec Docker):
    docker-compose exec api python scripts/generate_openapi.py --output /app/../docs/openapi.json

    # Ou depuis le conteneur:
    docker-compose exec api python scripts/generate_openapi.py --output /app/../docs/openapi.json

    # Depuis l'hôte (sans Docker):
    python backend/scripts/generate_openapi.py [--output openapi.json] [--format json|yaml]
"""

import argparse
import base64
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# Ajouter le répertoire backend au path pour les imports
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Changer le répertoire de travail vers backend pour les imports relatifs
original_cwd = Path.cwd()
with suppress(OSError):
    # Si on est déjà dans backend, ne rien faire
    os.chdir(backend_dir)

# ✅ Définir les variables d'environnement AVANT d'importer create_app
# Note: APP_ENCRYPTION_KEY_B64 et MASTER_ENCRYPTION_KEY doivent être fournis
# par le workflow GitHub Actions (depuis les secrets) ou manuellement

# Fallback: générer une clé AES-256 valide (32 octets) pour tests locaux uniquement
if not os.getenv("APP_ENCRYPTION_KEY_B64"):
    test_encryption_key = b"test-encryption-key-for-test!"  # Exactement 32 octets
    test_encryption_key_b64 = base64.b64encode(test_encryption_key).decode()
    os.environ.setdefault("APP_ENCRYPTION_KEY_B64", test_encryption_key_b64)

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-openapi-generation")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key-for-openapi-generation")
os.environ.setdefault(
    "MASTER_ENCRYPTION_KEY",
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
)
os.environ.setdefault("DATABASE_URL", "sqlite:///test.db")
os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("SKIP_ROUTES_INIT", "0")

# Import après changement de répertoire et définition des variables d'environnement
from app import create_app  # noqa: E402


def generate_openapi(output_path: Path, format_type: str = "json") -> None:
    """Génère la spécification OpenAPI depuis Flask-RESTX.

    Args:
        output_path: Chemin du fichier de sortie
        format_type: Format de sortie ('json' ou 'yaml')
    """
    # Les variables d'environnement sont déjà définies au niveau du module
    app = create_app("testing")

    # ✅ Configurer SERVER_NAME pour permettre la génération d'URLs sans contexte de requête
    app.config["SERVER_NAME"] = "localhost:5000"
    app.config["APPLICATION_ROOT"] = "/"
    app.config["PREFERRED_URL_SCHEME"] = "http"

    with app.app_context():
        # Récupérer la spec OpenAPI depuis Flask-RESTX
        api_v1 = app.extensions.get("restx_api_v1")
        if not api_v1:
            # Essayer d'importer directement
            from routes_api import api_v1

        # Générer la spec
        spec = api_v1.__schema__

        # Écrire le fichier
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type.lower() == "yaml":
            if yaml is None:
                print(
                    "❌ Erreur: PyYAML n'est pas installé. Installez-le avec: pip install pyyaml"
                )
                sys.exit(1)
            with output_path.open("w", encoding="utf-8") as f:
                yaml.dump(
                    spec,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        else:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2, ensure_ascii=False)

        print(f"✅ Spécification OpenAPI générée: {output_path}")
        print(f"   Format: {format_type.upper()}")
        print(f"   Endpoints documentés: {len(spec.get('paths', {}))}")


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Génère la spécification OpenAPI depuis Flask-RESTX"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/openapi.json"),
        help="Chemin du fichier de sortie (défaut: docs/openapi.json)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Format de sortie (défaut: json)",
    )

    args = parser.parse_args()

    # Ajuster l'extension du fichier selon le format
    if args.format == "yaml" and args.output.suffix == ".json":
        args.output = args.output.with_suffix(".yaml")
    elif args.format == "json" and args.output.suffix == ".yaml":
        args.output = args.output.with_suffix(".json")

    # ✅ Résoudre le chemin absolu pour éviter les problèmes de permissions
    # Si le chemin commence par /app/../ (hors volume monté), utiliser /app/docs à la place
    output_path = args.output.resolve()
    if str(output_path).startswith("/app/../"):
        # Le répertoire docs n'est pas monté, utiliser /app/docs à la place
        output_path = Path("/app/docs") / output_path.name
        print("⚠️  Le répertoire docs n'est pas monté dans le conteneur.")
        print(
            f"   Écriture dans {output_path} (accessible via ./backend/docs sur l'hôte)"
        )

    try:
        generate_openapi(output_path, args.format)
    except Exception as e:
        print(f"❌ Erreur lors de la génération: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
