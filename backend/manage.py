# backend/manage.py
# ruff: noqa: E402
# Note: E402 est une règle Flake8, pas Ruff
# Les directives # noqa: E402 sont nécessaires pour Flake8 dans le CI

import os
import sys

# ✅ FIX: Permettre de désactiver eventlet pour les migrations
# eventlet.monkey_patch() interfère avec les transactions Alembic/psycopg
# Utiliser DISABLE_EVENTLET=1 pour les commandes de migration
_disable_eventlet = os.getenv("DISABLE_EVENTLET", "0") == "1"

# ✅ RECOMMANDATION A: Auto-détection des commandes de migration
# Si la commande contient 'db' ou 'alembic', désactiver eventlet automatiquement
_is_migration_command = any(
    arg
    in ("db", "alembic", "migrate", "upgrade", "downgrade", "stamp", "heads", "current")
    for arg in sys.argv
)

if _is_migration_command and not _disable_eventlet:
    # ✅ RECOMMANDATION B: Warning si migrations sans DISABLE_EVENTLET
    print(
        "⚠️  [manage.py] Commande de migration détectée sans DISABLE_EVENTLET=1",
        flush=True,
    )
    print(
        "    → Désactivation automatique d'eventlet pour éviter les problèmes de transaction.",
        flush=True,
    )
    print(
        "    → Pour supprimer ce warning, utilisez: DISABLE_EVENTLET=1 flask db ...",
        flush=True,
    )
    _disable_eventlet = True

if not _disable_eventlet:
    import eventlet

    eventlet.monkey_patch()
elif os.getenv("DISABLE_EVENTLET", "0") == "1":
    # Explicitement désactivé par l'utilisateur
    print("✅ [manage.py] eventlet désactivé (DISABLE_EVENTLET=1)", flush=True)
# else: auto-désactivé pour migration (message déjà affiché)

import click  # noqa: E402
from flask_migrate import init as _init  # noqa: E402
from flask_migrate import migrate as _migrate  # noqa: E402
from flask_migrate import stamp as _stamp  # noqa: E402

# Importation des fonctions de migration nécessaires
from flask_migrate import upgrade as _upgrade  # noqa: E402

from app import create_app  # noqa: E402
from services.demo.seed_service import (  # noqa: E402
    PROFILES,
    reset_and_seed_demo_dataset,
)

# Crée une instance de l'application pour le contexte
config_name = os.getenv("FLASK_ENV") or "development"
app = create_app(config_name)


# --- Création de l'interface en ligne de commande avec Click ---


@click.group()
def cli():
    """Point d'entrée principal pour les commandes de gestion."""
    pass


@cli.group(name="db")
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


@cli.group(name="seed")
def seedcli():
    """Commandes de seed de données."""


@seedcli.command(name="demo")
@click.option("--reset/--no-reset", default=False, show_default=True)
@click.option(
    "--profile",
    "profile_name",
    default="sales",
    type=click.Choice(sorted(PROFILES.keys())),
    show_default=True,
)
def seed_demo(reset: bool, profile_name: str):
    """Seed dataset démo déterministe (tiny|sales)."""
    with app.app_context():
        summary = reset_and_seed_demo_dataset(
            profile_name=profile_name,
            reset=reset,
        )
    click.echo(
        f"✅ Seed demo terminé (profile={profile_name}, reset={reset}) - {summary}"
    )


if __name__ == "__main__":
    cli()
