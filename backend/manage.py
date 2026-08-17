# backend/manage.py
# ruff: noqa: E402
# Note: E402 est une règle Flake8, pas Ruff
# Les directives # noqa: E402 sont nécessaires pour Flake8 dans le CI

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

MIGRATION_CLI_ARGS = (
    "db",
    "alembic",
    "migrate",
    "upgrade",
    "downgrade",
    "stamp",
    "heads",
    "current",
)


def is_migration_command(argv: Sequence[str]) -> bool:
    """True si la ligne de commande vise une migration Alembic/Flask-Migrate."""
    return any(arg in MIGRATION_CLI_ARGS for arg in argv)


def env_disables_eventlet(value: str | None = None) -> bool:
    """True si DISABLE_EVENTLET=1 (valeur explicite ou environnement)."""
    raw = os.getenv("DISABLE_EVENTLET", "0") if value is None else value
    return raw == "1"


def apply_eventlet_monkey_patch() -> None:
    """Applique ``eventlet.monkey_patch()``."""
    import eventlet

    eventlet.monkey_patch()


def running_under_pytest() -> bool:
    """Évite le monkey_patch eventlet pendant la collecte pytest."""
    return "pytest" in sys.modules


def bootstrap_eventlet(
    argv: Sequence[str] | None = None,
    *,
    disable_env: str | None = None,
    apply_patch: bool | None = None,
) -> bool:
    """Décide si eventlet doit être désactivé. Retourne True si désactivé."""
    argv = sys.argv if argv is None else argv
    explicit = env_disables_eventlet(disable_env)
    disable = explicit
    should_patch = (not running_under_pytest()) if apply_patch is None else apply_patch

    if is_migration_command(argv) and not explicit:
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
        disable = True

    if not disable:
        if should_patch:
            apply_eventlet_monkey_patch()
    elif explicit:
        print("✅ [manage.py] eventlet désactivé (DISABLE_EVENTLET=1)", flush=True)

    return disable


def resolve_config_name(flask_env: str | None = None) -> str:
    """Nom de config Flask (FLASK_ENV ou development)."""
    env = os.getenv("FLASK_ENV") if flask_env is None else flask_env
    return env or "development"


# ✅ FIX: Permettre de désactiver eventlet pour les migrations
# eventlet.monkey_patch() interfère avec les transactions Alembic/psycopg
_disable_eventlet = bootstrap_eventlet()

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

_app = None


def get_app():
    """Instance Flask (création paresseuse pour les tests et ``manage.app``)."""
    global _app
    if _app is None:
        _app = create_app(resolve_config_name())
    return _app


def __getattr__(name: str):
    """Compat : ``import manage ; manage.app``."""
    if name == "app":
        return get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    with get_app().app_context():
        _init()
    click.echo("Dossier des migrations initialisé.")


@dbcli.command()
@click.option(
    "-m", "--message", required=True, help="Message de description pour la migration."
)
def migrate(message):
    """Génère une nouvelle migration."""
    with get_app().app_context():
        _migrate(message=message)
    click.echo("Script de migration généré.")


@dbcli.command()
def upgrade():
    """Applique les migrations à la base de données."""
    with get_app().app_context():
        _upgrade()
    click.echo("Migrations appliquées à la base de données.")


@dbcli.command()
@click.argument("revision", default="head")
def stamp(revision):
    """'Tamponne' la base de données avec une révision, sans exécuter la migration."""
    with get_app().app_context():
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
    with get_app().app_context():
        summary = reset_and_seed_demo_dataset(
            profile_name=profile_name,
            reset=reset,
        )
    click.echo(
        f"✅ Seed demo terminé (profile={profile_name}, reset={reset}) - {summary}"
    )


if __name__ == "__main__":
    cli()
