"""Barrière de la préparation financière. Fermée par défaut.

Le téléphone ne l'ouvre pas. La production, atmr_test et les hôtes
distants sont refusés. Même ouverte, elle n'autorise ni l'encaissement
réel ni l'émission d'une facture. La recette locale explicite
(SETTLEMENT_LOCAL_RECIPE) l'ouvre sur une base jetable ou sur la base
locale atmr servie par le compose local.
"""

from __future__ import annotations

import os

from domain.end_of_trip_settlement.simulation_barrier import (
    local_recipe_switch_on,
    recipe_database_allowed,
)
from services.recipe_environment import (
    database_host,
    database_name,
    production_environment,
)

_SWITCH = "SETTLEMENT_PRODUCTION_PREP"


class PrepClosed(Exception):
    """La préparation financière n'est pas ouverte."""


class PrepRefusal(Exception):
    """Refus métier. Aucune écriture n'est engagée si le statut est 4xx avant flush."""

    def __init__(self, code: str, status: int = 409) -> None:
        self.code = code
        self.status = status
        super().__init__(code)


def prep_open(
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
