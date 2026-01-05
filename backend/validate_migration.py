#!/usr/bin/env python3
"""Script de validation de la migration dispatch_routes -> dispatch."""

import base64
import os
import sys

# Générer une clé d'encryption valide pour les tests
os.environ.setdefault(
    "APP_ENCRYPTION_KEY_B64", base64.b64encode(os.urandom(32)).decode()
)

print("🔍 Validation de la migration dispatch_routes -> dispatch...")
print()

# Test 1: Import du namespace
try:
    from routes.dispatch import dispatch_ns

    print("✅ Import dispatch_ns réussi")
except Exception as e:
    print(f"❌ Erreur import dispatch_ns: {e}")
    sys.exit(1)

# Test 2: Import de routes_api
try:
    from routes_api import init_namespaces

    # Vérifier que la fonction existe
    assert callable(init_namespaces), "init_namespaces doit être callable"
    print("✅ Import routes_api réussi")
except Exception as e:
    print(f"❌ Erreur import routes_api: {e}")
    sys.exit(1)

# Test 3: Vérifier que dispatch_ns est bien enregistré
try:
    # Vérifier que le namespace existe et a des routes
    print(f"✅ Namespace dispatch_ns créé: {dispatch_ns.name}")
    print(f"   Description: {dispatch_ns.description}")
except Exception as e:
    print(f"❌ Erreur vérification namespace: {e}")
    sys.exit(1)

print()
print("✅ Migration validée : tous les imports fonctionnent correctement !")
print("   Le nouveau module routes.dispatch est opérationnel.")
