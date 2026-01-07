"""
unified_dispatch.validation - Validation des contraintes et assignments

Ce module contient :
- constraints.py : Validation des contraintes métier (ancien validation.py)
- assignment.py : Validation des assignments (ancien assignment_validator.py)
- analysis/ : Analyse des unassigned (déplacé depuis analysis/)

Créé lors du refactoring B1 - 7 janvier 2025
"""

# Import de compatibilité pour tools.py
from .constraints import (
    check_existing_assignment_conflict,
    is_groupable,
    validate_assignments,
    validate_driver_capacity,
    validate_no_duplicate_times,
    validate_no_temporal_conflicts,
)

# Exports publics (à compléter lors de la migration)
__all__ = [
    "check_existing_assignment_conflict",
    "is_groupable",
    "validate_assignments",
    "validate_driver_capacity",
    "validate_no_duplicate_times",
    "validate_no_temporal_conflicts",
]
