"""Script pour créer l'utilisateur de test admin@test.com."""

import sys
from pathlib import Path

# Ajouter le répertoire backend au PYTHONPATH
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

def seed_test_user():
    """Créer l'utilisateur admin@test.com pour les tests."""
    from app import create_app
    from ext import db
    from models.user import User, UserRole
    
    app = create_app()
    
    with app.app_context():
        # Vérifier si l'utilisateur existe déjà
        existing_user = User.query.filter_by(email="admin@test.com").first()
        
        if existing_user:
            print(f"✅ Utilisateur admin@test.com existe déjà (ID: {existing_user.id})")
            print(f"   Username: {existing_user.username}")
            print(f"   Role: {existing_user.role.value}")
            return
        
        # Créer l'utilisateur
        print("Creating user admin@test.com...")
        user = User(
            username="admin",
            email="admin@test.com",
            role=UserRole.ADMIN,
        )
        user.set_password("test123")
        
        db.session.add(user)
        db.session.commit()
        
        print(f"✅ Utilisateur créé avec succès!")
        print(f"   Email: {user.email}")
        print(f"   Username: {user.username}")
        print(f"   Role: {user.role.value}")
        print(f"   ID: {user.id}")

if __name__ == "__main__":
    print("=" * 80)
    print("Seed Test User (admin@test.com)")
    print("=" * 80)
    print()
    try:
        seed_test_user()
        print()
        print("=" * 80)
        print("✅ Seed complété")
        print("=" * 80)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

