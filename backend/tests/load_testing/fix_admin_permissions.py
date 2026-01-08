"""
Script pour donner les permissions COMPANY a l'utilisateur admin@test.com
pour permettre les tests de charge dispatch.
"""

import sys
from pathlib import Path

# Ajouter le repertoire backend au PYTHONPATH
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app
from ext import db
from models.enums import UserRole
from models.user import User


def fix_admin_permissions():
    print("=" * 80)
    print("Fix Admin Permissions pour Tests de Charge")
    print("=" * 80)
    print()

    app = create_app(config_name="development")

    with app.app_context():
        # Trouver l'utilisateur admin@test.com
        user = User.query.filter_by(email="admin@test.com").first()

        if not user:
            print("[ERROR] Utilisateur admin@test.com introuvable!")
            print("   Executez d'abord: python tests/load_testing/seed_test_user.py")
            return False

        print(f"[1/2] Utilisateur trouve: {user.email}")
        print(f"   Role actuel: {user.role.value}")
        print(f"   ID: {user.id}")
        print()

        # Changer le role en COMPANY (temporairement pour tests)
        old_role = user.role
        user.role = UserRole.COMPANY

        try:
            db.session.commit()
            print("[2/2] Role modifie avec succes!")
            print(f"   Ancien role: {old_role.value}")
            print(f"   Nouveau role: {user.role.value}")
            print()
            print("[OK] Utilisateur admin@test.com peut maintenant executer dispatch!")
            print()
            print("ATTENTION: Ce changement est TEMPORAIRE pour les tests de charge.")
            print("           Restaurez le role ADMIN apres les tests si necessaire.")
            return True
        except Exception as e:
            db.session.rollback()
            print(f"[ERROR] Echec de la modification: {e}")
            return False

    print()
    print("=" * 80)


if __name__ == "__main__":
    success = fix_admin_permissions()
    sys.exit(0 if success else 1)
