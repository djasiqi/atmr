"""Barrière du dossier de simulation. Fermée par défaut.

Une variable du téléphone ne l'ouvre pas. La production, la base
partagée et les noms de base réels sont refusés. L'ouverture exige
une base jetable et un interrupteur serveur : SETTLEMENT_SIMULATION_LEDGER
pour ce seul dossier, ou SETTLEMENT_LOCAL_RECIPE pour la recette locale
complète (volet chauffeur, grand livre simulé, préparation financière).
"""

from __future__ import annotations

import os

from services.recipe_environment import (
    database_host,
    database_name,
    production_environment,
)

_FORBIDDEN = frozenset({"", "atmr", "atmr_test", "atmr_mig_ci", "atmrstg", "postgres"})
_PREFIXES = ("atmr_etape", "lirie_sim_", "lirie_e2e_")
_SHARED_LOCAL_NAME = "atmr"
_SHARED_LOCAL_HOSTS = frozenset({"pgbouncer", "localhost", "127.0.0.1", "postgres"})
_SWITCH = "SETTLEMENT_SIMULATION_LEDGER"
LOCAL_RECIPE_SWITCH = "SETTLEMENT_LOCAL_RECIPE"


def disposable_database(name: str | None) -> bool:
    if not name or name in _FORBIDDEN:
        return False
    return name.startswith(_PREFIXES)


def shared_local_database(name: str | None, host: str | None) -> bool:
    """Base de développement locale, seulement si l'hôte est celui du compose local.

    Le nom `atmr` seul ne suffit pas : une base de même nom sur un hôte distant
    reste fermée. `atmr_test`, `atmrstg` et la production ne passent pas ici.
    """
    return name == _SHARED_LOCAL_NAME and (host or "") in _SHARED_LOCAL_HOSTS


def recipe_database_allowed(
    name: str | None, *, host: str | None = None, configured: bool = False
) -> bool:
    if disposable_database(name):
        return True
    if host is None and not configured:
        return False
    return shared_local_database(name, host)


def local_recipe_switch_on(switch: str | None = None) -> bool:
    """Interrupteur serveur de la recette locale. Jamais déduit d'un client."""
    flag = os.environ.get(LOCAL_RECIPE_SWITCH, "") if switch is None else switch
    return str(flag).strip() == "1"


def ledger_open(
    *,
    database: str | None = None,
    switch: str | None = None,
    production: bool | None = None,
    host: str | None = None,
) -> bool:
    if production if production is not None else production_environment():
        return False
    resolved = database_name() if database is None else database
    resolved_host = database_host() if host is None and database is None else host
    if not recipe_database_allowed(
        resolved, host=resolved_host, configured=database is None
    ):
        return False
    if switch is not None:
        return str(switch).strip() == "1"
    if os.environ.get(_SWITCH, "").strip() == "1":
        return True
    return local_recipe_switch_on()
