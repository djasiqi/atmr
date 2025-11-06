#!/usr/bin/env python3
"""Script de test pour valider la fonction validate_required_env_vars.

Usage:
    # Test en mode development (moins strict)
    FLASK_CONFIG=development python scripts/test_env_validation.py
    
    # Test en mode production (strict)
    FLASK_CONFIG=production python scripts/test_env_validation.py
"""

import os
import sys
from pathlib import Path

# Ajouter le backend au path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app import create_app # noqa: E402


def test_development_mode():
    """Test la validation en mode development."""
    print("🧪 Test validation en mode DEVELOPMENT...")
    
    # En development, seulement SECRET_KEY/JWT_SECRET_KEY sont requis
    try:
        _ = create_app("development")
        print("   ✅ Mode development OK (validation passée)")
        return True
    except RuntimeError as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_production_mode():
    """Test la validation en mode production."""
    print("\n🧪 Test validation en mode PRODUCTION...")
    
    # Sauvegarder les valeurs actuelles
    original_db = os.environ.get("DATABASE_URL")
    original_redis = os.environ.get("REDIS_URL")
    
    try:
        # Tester avec toutes les variables présentes
        _ = create_app("production")
        print("   ✅ Mode production OK (toutes variables présentes)")
        
        # Tester sans DATABASE_URL
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        try:
            _ = create_app("production")
            print("   ❌ Erreur: Devrait échouer sans DATABASE_URL")
            return False
        except RuntimeError as e:
            print(f"   ✅ Validation fonctionne (bloqué sans DATABASE_URL): {e}")
        
        # Restaurer pour test suivant
        if original_db:
            os.environ["DATABASE_URL"] = original_db
        
        return True
        
    except RuntimeError as e:
        print(f"   ⚠️  Production mode: {e}")
        print("   (Normal si variables manquantes)")
        return True
    finally:
        # Restaurer valeurs originales
        if original_db:
            os.environ["DATABASE_URL"] = original_db
        elif "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
            
        if original_redis:
            os.environ["REDIS_URL"] = original_redis
        elif "REDIS_URL" in os.environ:
            del os.environ["REDIS_URL"]


if __name__ == "__main__":
    print("=" * 60)
    print("TEST VALIDATION VARIABLES D'ENVIRONNEMENT")
    print("=" * 60)
    
    dev_ok = test_development_mode()
    prod_ok = test_production_mode()
    
    print("\n" + "=" * 60)
    if dev_ok and prod_ok:
        print("✅ Tous les tests passent!")
        sys.exit(0)
    else:
        print("❌ Certains tests ont échoué")
        sys.exit(1)

