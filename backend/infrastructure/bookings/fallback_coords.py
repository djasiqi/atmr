from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from models import Company


def get_booking_fallback_coords(company: Company | None) -> tuple[float, float]:
    """Adapter infrastructure pour choisir des coordonnées de fallback (pickup/dropoff).

    Implémentation déléguée à `services.unified_dispatch.data` pour réutiliser la
    logique existante (company coords / config / default).
    """
    from services.unified_dispatch.data import (
        FALLBACK_COORD_DEFAULT,
        _company_latlon_optional,
        _configured_fallback_coords,
    )

    if company:
        coords = _company_latlon_optional(company)
        if coords:
            return coords

        coords = _configured_fallback_coords(company)
        if coords:
            return coords

    return FALLBACK_COORD_DEFAULT
