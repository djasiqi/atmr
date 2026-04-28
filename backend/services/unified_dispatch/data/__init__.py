"""
unified_dispatch.data - Récupération et préparation des données

Ce module contient :
- loader.py : Récupération des données depuis la DB (ancien data.py)
- clustering.py : Clustering géographique des bookings/drivers
- warm_start.py : Initialisation warm start pour l'optimisation

Créé lors du refactoring B1 - 7 janvier 2025
"""

# ✅ Imports depuis loader.py pour compatibilité avec les anciens imports
from .loader import (
    FALLBACK_COORD_DEFAULT,
    _company_latlon_optional,
    _configured_fallback_coords,
    acquire_dispatch_lock,
    build_problem_data,
    build_time_matrix,
    build_vrptw_problem,
    calculate_eta,
    detect_delay,
    enrich_booking_coords,
    enrich_driver_coords,
    get_available_drivers,
    get_available_drivers_split,
    get_bookings_for_day,
    get_bookings_for_dispatch,
    get_next_free_at,
    pick_urgent_returns,
    release_dispatch_lock,
)

# Exports publics
__all__ = [
    "FALLBACK_COORD_DEFAULT",
    "_company_latlon_optional",
    "_configured_fallback_coords",
    "acquire_dispatch_lock",
    "build_problem_data",
    "build_time_matrix",
    "build_vrptw_problem",
    "calculate_eta",
    "detect_delay",
    "enrich_booking_coords",
    "enrich_driver_coords",
    "get_available_drivers",
    "get_available_drivers_split",
    "get_bookings_for_day",
    "get_bookings_for_dispatch",
    "get_next_free_at",
    "pick_urgent_returns",
    "release_dispatch_lock",
]
