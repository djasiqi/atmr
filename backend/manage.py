# backend/manage.py

import eventlet
eventlet.monkey_patch()

import click
import os
from flask.cli import with_appcontext
from app import create_app
from ext import db

# Importation de TOUTES les fonctions de migration nécessaires
from flask_migrate import upgrade as _upgrade
from flask_migrate import migrate as _migrate
from flask_migrate import stamp as _stamp
from flask_migrate import init as _init # La fonction qui manquait

# Crée une instance de l'application pour le contexte
config_name = os.getenv('FLASK_ENV') or 'development'
app = create_app(config_name)


# --- Création de l'interface en ligne de commande avec Click ---

@click.group()
def cli():
    """Point d'entrée principal pour les commandes de gestion."""
    pass

@cli.group()
def db():
    """Commandes pour les migrations de base de données."""
    pass

# ---- LA COMMANDE QUI MANQUAIT ----
@db.command()
@with_appcontext
def init():
    """Initialise le dossier des migrations."""
    _init()
    click.echo("Dossier des migrations initialisé.")

# Définition de la commande "migrate"
@db.command()
@with_appcontext
@click.option('-m', '--message', required=True, help="Message de description pour la migration.")
def migrate(message):
    """Génère une nouvelle migration."""
    with app.app_context():
        _migrate(message=message)
    click.echo("Script de migration généré.")

# Définition de la commande "upgrade"
@db.command()
@with_appcontext
def upgrade():
    """Applique les migrations à la base de données."""
    with app.app_context():
        _upgrade()
    click.echo("Migrations appliquées à la base de données.")

# Définition de la commande "stamp"
@db.command()
@with_appcontext
@click.argument('revision', default='head')
def stamp(revision):
    """'Tamponne' la base de données avec une révision, sans exécuter la migration."""
    with app.app_context():
        _stamp(revision=revision)
    click.echo(f"Base de données tamponnée avec la révision : {revision}.")

if __name__ == "__main__":
    cli()