"""Alias de compatibilité vers services.geolocation.osrm.

Ce module ré-exporte les symboles utilisés par les tests qui patchent
``services.osrm_client._table`` / ``services.osrm_client._route``
(test_osrm_fallback, test_dispatch). Le code réel est dans
services.geolocation.osrm.
"""
from services.geolocation import osrm as _osrm

# Symboles utilisés par les patches (tests/integration/test_osrm_fallback.py,
# tests/test_dispatch.py)
_table = _osrm._table
_route = _osrm._route

# Ré-export de l'API publique pour compatibilité
build_distance_matrix_osrm = _osrm.build_distance_matrix_osrm
route_info = _osrm.route_info
get_distance_time = _osrm.get_distance_time
get_matrix = _osrm.get_matrix
eta_seconds = _osrm.eta_seconds
build_distance_matrix_osrm_with_cb = _osrm.build_distance_matrix_osrm_with_cb
get_distance_time_cached = _osrm.get_distance_time_cached
get_matrix_cached = _osrm.get_matrix_cached

__all__ = [
    "_route",
    "_table",
    "build_distance_matrix_osrm",
    "build_distance_matrix_osrm_with_cb",
    "eta_seconds",
    "get_distance_time",
    "get_distance_time_cached",
    "get_matrix",
    "get_matrix_cached",
    "route_info",
]
