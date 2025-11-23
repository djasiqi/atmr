# backend/manage.py
# ruff: noqa: E402
# Note: E402 est une règle Flake8, pas Ruff
# Les directives # noqa: E402 sont nécessaires pour Flake8 dans le CI

import eventlet

eventlet.monkey_patch()

import os  # noqa: E402

import click  # noqa: E402
from flask_migrate import init as _init  # noqa: E402
from flask_migrate import migrate as _migrate  # noqa: E402
from flask_migrate import stamp as _stamp  # noqa: E402

# Importation des fonctions de migration nécessaires
from flask_migrate import upgrade as _upgrade  # noqa: E402

from app import create_app  # noqa: E402

# Crée une instance de l'application pour le contexte
config_name = os.getenv("FLASK_ENV") or "development"
app = create_app(config_name)


# --- Création de l'interface en ligne de commande avec Click ---


@click.group()
def cli():
    """Point d'entrée principal pour les commandes de gestion."""
    pass


@cli.group(name="db")  # type: ignore[reportFunctionMemberAccess]
def dbcli():
    """Commandes pour les migrations de base de données."""


# ...
@dbcli.command()
def init():
    """Initialise le dossier des migrations."""
    with app.app_context():
        _init()
    click.echo("Dossier des migrations initialisé.")


@dbcli.command()
@click.option(
    "-m", "--message", required=True, help="Message de description pour la migration."
)
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
@click.argument("revision", default="head")
def stamp(revision):
    """'Tamponne' la base de données avec une révision, sans exécuter la migration."""
    with app.app_context():
        _stamp(revision=revision)
    click.echo(f"Base de données tamponnée avec la révision : {revision}.")


if __name__ == "__main__":
    cli()
