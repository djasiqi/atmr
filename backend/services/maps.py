"""Alias de compatibilité vers ``services.geolocation.maps``.

Les tests et l'infrastructure historique importent encore ``services.maps``.
"""

from __future__ import annotations

from services.geolocation.maps import (
    geocode_address,
    geocode_address_nominatim,
    geocode_addresses_batch,
    get_distance_duration,
)

__all__ = [
    "geocode_address",
    "geocode_address_nominatim",
    "geocode_addresses_batch",
    "get_distance_duration",
]
