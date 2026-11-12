"""Registre financier V1 sur la base de développement locale.

Distinct de ``END_OF_TRIP_COLLECTION_ENABLED`` et de la simulation
historique. Le téléphone ne décide pas. La production, ``atmr_test``,
``atmrstg`` et une base ``atmr`` servie par un hôte distant restent
fermées. L'interrupteur serveur ``SETTLEMENT_LOCAL_RECIPE`` est requis
en plus de l'identité de la base.
"""

from __future__ import annotations

from domain.end_of_trip_settlement.simulation_barrier import (
    local_recipe_switch_on,
    shared_local_database,
)
from services.recipe_environment import (
    database_host,
    database_name,
    production_environment,
)


def v1_local_ledger_enabled(
    *,
    database: str | None = None,
    switch: str | None = None,
    production: bool | None = None,
    host: str | None = None,
) -> bool:
    """Vrai seulement pour la base locale ``atmr`` du compose, hors production."""
    if production if production is not None else production_environment():
        return False
    resolved = database_name() if database is None else database
    resolved_host = database_host() if host is None and database is None else host
    if not shared_local_database(resolved, resolved_host):
        return False
    return local_recipe_switch_on(switch)
