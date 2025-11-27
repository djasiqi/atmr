#!/usr/bin/env python3
"""
Script pour gérer les utilisateurs ATMR (créer, lister, réinitialiser mot de passe).

Usage:
    # Lister tous les utilisateurs
    python scripts/manage_users.py list

    # Créer un utilisateur admin
    python scripts/manage_users.py create-admin --username admin --email admin@example.com --password MonMotDePasse123

    # Réinitialiser le mot de passe d'un utilisateur
    python scripts/manage_users.py reset-password --username admin --new-password NouveauMotDePasse123

    # Réinitialiser le mot de passe d'un utilisateur par ID
    python scripts/manage_users.py reset-password --user-id 1 --new-password NouveauMotDePasse123
"""

import os
import sys
import uuid
from pathlib import Path

# Ajouter le répertoire backend au PYTHONPATH
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

import click
from werkzeug.security import generate_password_hash

from app import create_app
from ext import db
from models.user import User
from models.enums import UserRole


@click.group()
def cli():
    """Gestion des utilisateurs ATMR."""
    pass


@cli.command()
def list():
    """Liste tous les utilisateurs de la base de données."""
    app = create_app()
    with app.app_context():
        users = User.query.all()
        if not users:
            click.echo("❌ Aucun utilisateur trouvé dans la base de données.")
            return

        click.echo(f"\n📋 {len(users)} utilisateur(s) trouvé(s):\n")
        click.echo(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Role':<10}")
        click.echo("-" * 70)
        for user in users:
            click.echo(
                f"{user.id:<5} {user.username:<20} {user.email or 'N/A':<30} {user.role.value if user.role else 'N/A':<10}"
            )
        click.echo()


@cli.command()
@click.option("--username", required=True, help="Nom d'utilisateur")
@click.option("--email", required=True, help="Email de l'utilisateur")
@click.option("--password", required=True, help="Mot de passe")
@click.option("--role", default="ADMIN", help="Rôle (ADMIN, CLIENT, DRIVER, COMPANY)")
def create_admin(username, email, password, role):
    """Crée un utilisateur admin."""
    app = create_app()
    with app.app_context():
        # Vérifier si l'utilisateur existe déjà
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing_user:
            click.echo(
                f"❌ Un utilisateur avec ce username ou email existe déjà (ID: {existing_user.id})"
            )
            return

        # Valider le rôle
        try:
            user_role = UserRole[role.upper()]
        except KeyError:
            click.echo(f"❌ Rôle invalide: {role}. Rôles valides: ADMIN, CLIENT, DRIVER, COMPANY")
            return

        # Créer l'utilisateur
        user = User()
        user.username = username
        user.email = email
        user.role = user_role
        user.public_id = str(uuid.uuid4())
        user.set_password(password, force_change=False)

        db.session.add(user)
        db.session.commit()

        click.echo(f"✅ Utilisateur créé avec succès!")
        click.echo(f"   ID: {user.id}")
        click.echo(f"   Username: {user.username}")
        click.echo(f"   Email: {user.email}")
        click.echo(f"   Role: {user.role.value}")
        click.echo(f"   Public ID: {user.public_id}")


@cli.command()
@click.option("--username", help="Nom d'utilisateur")
@click.option("--user-id", type=int, help="ID de l'utilisateur")
@click.option("--new-password", required=True, help="Nouveau mot de passe")
def reset_password(username, user_id, new_password):
    """Réinitialise le mot de passe d'un utilisateur."""
    if not username and not user_id:
        click.echo("❌ Vous devez fournir soit --username soit --user-id")
        return

    app = create_app()
    with app.app_context():
        if user_id:
            user = User.query.filter_by(id=user_id).first()
        else:
            user = User.query.filter_by(username=username).first()

        if not user:
            click.echo("❌ Utilisateur non trouvé")
            return

        user.set_password(new_password, force_change=False)
        db.session.commit()

        click.echo(f"✅ Mot de passe réinitialisé avec succès pour l'utilisateur:")
        click.echo(f"   ID: {user.id}")
        click.echo(f"   Username: {user.username}")
        click.echo(f"   Email: {user.email}")
        click.echo(f"   Nouveau mot de passe: {new_password}")


if __name__ == "__main__":
    cli()

