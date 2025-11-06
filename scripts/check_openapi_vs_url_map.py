#!/usr/bin/env python3
"""
Script pour comparer les routes réelles de l'application Flask avec l'OpenAPI/Swagger.

Usage:
    python scripts/check_openapi_vs_url_map.py > artifacts/routes_diff.txt
"""

import sys
from pathlib import Path

# Ajouter le backend au path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from flask import Flask  # noqa: E402

try:
    from app import create_app
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    print("Assurez-vous d'être dans le répertoire racine du projet.")
    sys.exit(1)


def main():
    """Compare les routes Flask avec OpenAPI."""
    app: Flask = create_app("testing")
    
    with app.app_context():
        # 1) Routes réelles de Flask
        real_routes = set()
        for rule in app.url_map.iter_rules():
            # Ignorer routes statiques, uploads, socket.io
            if rule.rule.startswith(("/static", "/uploads", "/socket.io")):
                continue
            methods = sorted(rule.methods - {"HEAD", "OPTIONS"})
            if methods:
                real_routes.add((rule.rule, ",".join(methods)))
        
        # 2) Routes OpenAPI/Swagger (parser swagger.json si serveur lancé)
        doc_routes = set()
        try:
            import requests
            resp = requests.get("http://localhost:5000/api/swagger.json", timeout=2)
            if resp.status_code == 200:
                spec = resp.json()
                for path, methods in spec.get("paths", {}).items():
                    for method in methods.keys():
                        if method.upper() not in ("HEAD", "OPTIONS"):
                            doc_routes.add((path, method.upper()))
        except Exception as e:
            print(f"⚠️ Impossible de récupérer OpenAPI (serveur non lancé?): {e}")
            print("   Pour comparer, lancez: python backend/app.py")
        
        # 3) Différences
        missing_in_doc = sorted(real_routes - doc_routes)
        missing_in_real = sorted(doc_routes - real_routes)
        
        print("=" * 80)
        print("COMPARAISON ROUTES FLASK ↔ OPENAPI/SWAGGER")
        print("=" * 80)
        print()
        
        print("📊 STATISTIQUES")
        print(f"  Routes dans app Flask: {len(real_routes)}")
        print(f"  Routes dans OpenAPI/Swagger: {len(doc_routes)}")
        print(f"  Routes app sans documentation: {len(missing_in_doc)}")
        print(f"  Routes doc sans implémentation: {len(missing_in_real)}")
        print()
        
        if missing_in_doc:
            print("=" * 80)
            print("⚠️  ROUTES DANS APP MAIS PAS DANS OPENAPI")
            print("=" * 80)
            for path, methods in missing_in_doc[:20]:  # Limiter à 20 pour lisibilité
                print(f"  {path:50s} [{methods}]")
            if len(missing_in_doc) > 20:
                remaining = len(missing_in_doc) - 20
                print(f"  ... et {remaining} autres")
            print()
        
        if missing_in_real:
            print("=" * 80)
            print("⚠️  ROUTES DANS OPENAPI MAIS PAS MONTÉES DANS APP")
            print("=" * 80)
            for path, methods in missing_in_real[:20]:
                print(f"  {path:50s} [{methods}]")
            if len(missing_in_real) > 20:
                remaining = len(missing_in_real) - 20
                print(f"  ... et {remaining} autres")
            print()
        
        if not missing_in_doc and not missing_in_real:
            print("✅ Parfait: Toutes les routes sont documentées et montées!")
        elif len(missing_in_doc) + len(missing_in_real) < 5:
            print("⚠️  Quelques écarts mineurs détectés")
        else:
            print("❌ Écarts importants détectés - action recommandée")


if __name__ == "__main__":
    main()

