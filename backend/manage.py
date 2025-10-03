# backend/manage.py
# ruff: noqa: E402

import eventlet
eventlet.monkey_patch()

import os
import click
from app import create_app

# Importation des fonctions de migration nécessaires
from flask_migrate import upgrade as _upgrade
from flask_migrate import migrate as _migrate
from flask_migrate import stamp as _stamp
from flask_migrate import init as _init  # initialisation du dossier migrations

# Crée une instance de l'application pour le contexte
config_name = os.getenv('FLASK_ENV') or 'development'
app = create_app(config_name)


# --- Création de l'interface en ligne de commande avec Click ---

@click.group()
def cli():
    """Point d'entrée principal pour les commandes de gestion."""
    pass

@cli.group(name="db")
def dbcli():
    """Commandes pour les migrations de base de données."""
    pass

# ...
@dbcli.command()
def init():
    """Initialise le dossier des migrations."""
    with app.app_context():
        _init()
    click.echo("Dossier des migrations initialisé.")

@dbcli.command()
@click.option('-m', '--message', required=True, help="Message de description pour la migration.")
def migrate(message):
    """Génère une nouvelle migration."""
    with app.app_context():
        _migrate(message=message)
    click.echo("Script de migration généré.")

@dbcli.command()
def upgrade():
    """Applique les migrations à la base de données."""
    with app.app_context():
        _upgrade()
    click.echo("Migrations appliquées à la base de données.")

@dbcli.command()
@click.argument('revision', default='head')
def stamp(revision):
    """'Tamponne' la base de données avec une révision, sans exécuter la migration."""
    with app.app_context():
        _stamp(revision=revision)
    click.echo(f"Base de données tamponnée avec la révision : {revision}.")


if __name__ == "__main__":
    cli()