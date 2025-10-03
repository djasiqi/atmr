from __future__ import annotations

import os
from typing import Any, cast

from dotenv import load_dotenv
from app import create_app
from ext import db  # instance SQLAlchemy
from models import User, UserRole

# Charger l'environnement APRÈS les imports pour éviter Ruff E402
load_dotenv()

# 1. On utilise la factory pour créer une instance de l'application
config_name = os.getenv('FLASK_ENV') or 'development'
app = create_app(config_name)

def add_or_update_admin() -> None:
    email = "jasiqi.drin@gmail.com"
    existing_admin = User.query.filter_by(email=email).first()
    
    if existing_admin:
        # Mettre à jour les informations de l'administrateur existant
        # Utilise setattr pour éviter les avertissements de typage Pylance
        setattr(existing_admin, "username", "drinjasiqi")
        setattr(existing_admin, "first_name", "Drin")
        setattr(existing_admin, "last_name", "Jasiqi")
        setattr(existing_admin, "role", UserRole.admin)
        existing_admin.set_password("Palidhje@24_07!!")
        print("Administrateur mis à jour avec succès !")
    else:
        # Créer un nouvel administrateur
        UserCtor = cast(Any, User)
        admin = UserCtor(
            username="drinjasiqi",
            first_name="Drin",
            last_name="Jasiqi",
            email=email,
            role=UserRole.admin
        )
        admin.set_password("Palidhje@24_07!!")
        db.session.add(admin)
        print("Administrateur ajouté avec succès !")
    
    # On commit une seule fois à la fin
    db.session.commit()

if __name__ == "__main__":
    # 2. On utilise le contexte de l'application que nous venons de créer
    with app.app_context():
        add_or_update_admin()