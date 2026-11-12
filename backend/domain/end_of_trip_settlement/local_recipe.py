"""Environnement de recette locale explicitement identifié.

Conditions cumulatives, aucune déduite du téléphone : hors production,
base jetable (lirie_e2e_, lirie_sim_, atmr_etape) ou base locale `atmr`
servie par un hôte du compose local, interrupteur serveur
SETTLEMENT_LOCAL_RECIPE à 1.

Cet environnement ouvre ensemble le volet chauffeur, le grand livre
simulé et la préparation financière. Il n'ouvre ni atmr_test, ni
atmrstg, ni un hôte distant, ni la production, et ne rend aucun
paiement exécutable.
"""

from __future__ import annotations

from domain.end_of_trip_settlement.simulation_barrier import (
    local_recipe_switch_on,
    recipe_database_allowed,
)
from services.recipe_environment import (
    database_host,
    database_name,
    production_environment,
)


def local_recipe_environment(
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
    return local_recipe_switch_on(switch)
