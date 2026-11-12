"""Activation produit de l'encaissement en fin de transport.

La constante reste fausse : aucune route, aucun paiement et aucune
facture ne doivent s'en servir tant que la politique financière n'est
pas approuvée.
"""

from __future__ import annotations

import os
from pathlib import Path

END_OF_TRIP_COLLECTION_ENABLED = False

# Rubrique de préférences. Elle n'autorise aucun moyen réel.
SETTLEMENT_COMPANY_SETTINGS_ENABLED = False

# Préparation financière à l'acceptation. Le parcours réel reste inchangé.
SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED = False

# Lecture du panneau chauffeur. Elle n'autorise aucun encaissement.
SETTLEMENT_DRIVER_PANEL_ENABLED = False

_RECIPE_DIR = Path("/tmp/lirie-settlement-recipe")


def _recipe_switch(name: str) -> bool:
    """Interrupteur local au conteneur de recette. Inerte hors atmr_test."""
    if "atmr_test" not in os.environ.get("DATABASE_URL", ""):
        return False
    try:
        value = (_RECIPE_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return False
    return value.strip() == "1"


def company_settings_enabled() -> bool:
    if bool(SETTLEMENT_COMPANY_SETTINGS_ENABLED) or _recipe_switch("company-settings"):
        return True
    from domain.end_of_trip_settlement.local_recipe import local_recipe_environment

    return local_recipe_environment()


def acceptance_snapshot_enabled() -> bool:
    if bool(SETTLEMENT_ACCEPTANCE_SNAPSHOT_ENABLED) or _recipe_switch(
        "acceptance-snapshot"
    ):
        return True
    from domain.end_of_trip_settlement.local_recipe import local_recipe_environment

    return local_recipe_environment()


def driver_panel_enabled() -> bool:
    """Constante produit, interrupteur atmr_test, ou recette locale jetable."""
    if bool(SETTLEMENT_DRIVER_PANEL_ENABLED) or _recipe_switch("driver-panel"):
        return True
    from domain.end_of_trip_settlement.local_recipe import local_recipe_environment

    return local_recipe_environment()


def simulation_preview_allowed(
    *, panel_enabled: bool, production: bool, recipe_database: bool
) -> bool:
    """La prévisualisation exige la recette, pas seulement un bouton masqué."""
    return bool(panel_enabled) and not production and recipe_database


def simulation_preview_enabled() -> bool:
    from domain.end_of_trip_settlement.local_recipe import local_recipe_environment
    from services.recipe_environment import (
        production_environment,
        recipe_writes_allowed,
    )

    return simulation_preview_allowed(
        panel_enabled=driver_panel_enabled(),
        production=production_environment(),
        recipe_database=recipe_writes_allowed() or local_recipe_environment(),
    )
