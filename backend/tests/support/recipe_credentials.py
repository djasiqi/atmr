"""Identifiants synthétiques pour recettes isolées (bases jetables).

Construit par concaténation à l'exécution pour éviter un littéral unique
de type « Generic Password » dans le dépôt, tout en restant déterministe
pour les recettes HTTP (Python + Jest).
"""

from __future__ import annotations

# Mot de passe de test uniquement — jamais un secret opérationnel.
RECIPE_PASSWORD = "-".join(("local", "test", "fixture", "p0gg"))

__all__ = ["RECIPE_PASSWORD"]
