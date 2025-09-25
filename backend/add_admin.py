from dotenv import load_dotenv
load_dotenv()

import os
from app import create_app
from ext import db  # On importe 'db' depuis 'ext.py', comme dans le reste du projet
from models import User, UserRole

# 1. On utilise la factory pour créer une instance de l'application
config_name = os.getenv('FLASK_ENV') or 'development'
app = create_app(config_name)

def add_or_update_admin():
    email = "jasiqi.drin@gmail.com"
    existing_admin = User.query.filter_by(email=email).first()
    
    if existing_admin:
        # Mettre à jour les informations de l'administrateur existant
        existing_admin.username = "drinjasiqi"
        existing_admin.first_name = "Drin"
        existing_admin.last_name = "Jasiqi"
        existing_admin.role = UserRole.admin
        existing_admin.set_password("Palidhje@24_07!!")
        print("Administrateur mis à jour avec succès !")
    else:
        # Créer un nouvel administrateur
        admin = User(
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