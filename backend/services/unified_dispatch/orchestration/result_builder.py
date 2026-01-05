# backend/services/unified_dispatch/orchestration/result_builder.py
"""Constructeur de résultat pour le dispatch.

Ce module gère la construction du résultat final du dispatch. Il est responsable de :
- La sérialisation des assignations, bookings et drivers
- La construction du DispatchResult avec toutes les métadonnées
- La conversion en format dict pour l'API

Side-effects:
    Aucun (module purement fonctionnel de sérialisation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, cast

from services.unified_dispatch.types import DispatchResult

logger = logging.getLogger(__name__)


class ResultBuilder:
    """Constructeur de résultat pour le dispatch.

    Cette classe centralise la logique de construction du résultat final :
    - Sérialisation des entités (assignations, bookings, drivers)
    - Construction du DispatchResult
    - Conversion en format dict pour l'API

    Exemple:
        >>> builder = ResultBuilder()
        >>> result = builder.build(
        ...     dispatch_run_id=42,
        ...     assignments=assignments,
        ...     unassigned_ids=[5, 10],
        ...     bookings=bookings,
        ...     drivers=drivers,
        ...     meta={"assignments_count": 20},
        ...     debug={"used_heuristic": True}
        ... )
        >>> result["assignments_count"]  # doctest: +SKIP
        20
    """

    def build(
        self,
        dispatch_run_id: int | None,
        assignments: list[Any],
        unassigned_ids: list[int],
        bookings: list[Any],
        drivers: list[Any],
        meta: dict[str, Any],
        debug: dict[str, Any],
    ) -> dict[str, Any]:
        """Construit le résultat final du dispatch.

        Sérialise toutes les entités et construit un DispatchResult avec
        toutes les métadonnées, puis le convertit en dict pour l'API.

        Args:
            dispatch_run_id: ID du DispatchRun (peut être None)
            assignments: List des assignations finales
            unassigned_ids: List des IDs de bookings non assignés
            bookings: List des bookings
            drivers: List des drivers
            meta: Dict avec métriques agrégées
            debug: Dict avec informations de debug

        Returns:
            Dict contenant le résultat final avec clés :
            - dispatch_run_id: ID du DispatchRun
            - assignments: List des assignations sérialisées
            - unassigned: List des IDs non assignés
            - bookings: List des bookings sérialisés
            - drivers: List des drivers sérialisés
            - meta: Dict avec métriques agrégées
            - debug: Dict avec informations de debug
        """
        # Sérialiser toutes les entités
        ser_assignments = [self._serialize_assignment(a) for a in assignments]
        ser_bookings = [self._serialize_booking(b) for b in bookings]
        ser_drivers = [self._serialize_driver(d) for d in drivers]

        # Construire le résultat final
        result = DispatchResult(
            dispatch_run_id=dispatch_run_id,
            assignments=ser_assignments,
            unassigned=unassigned_ids,
            bookings=ser_bookings,
            drivers=ser_drivers,
            meta=meta,
            debug=debug,
        )

        return result.to_dict()

    def _serialize_assignment(self, assignment: Any) -> Dict[str, Any]:
        """Sérialise une assignation en dict API.

        Convertit une assignation (objet ou dict) en format dict pour l'API.
        Utilise to_dict() si disponible, sinon extrait les attributs manuellement.

        Args:
            assignment: Objet assignation ou dict

        Returns:
            Dict avec clés : booking_id, driver_id, dispatch_run_id
        """
        if hasattr(assignment, "to_dict"):
            return cast(Dict[str, Any], assignment.to_dict())
        return {
            "booking_id": getattr(assignment, "booking_id", None),
            "driver_id": getattr(assignment, "driver_id", None),
            "dispatch_run_id": getattr(assignment, "dispatch_run_id", None),
        }

    def _serialize_booking(self, booking: Any) -> Dict[str, Any]:
        """Sérialise un booking en dict API.

        Convertit un booking (objet ou dict) en format dict pour l'API.
        Utilise to_dict() si disponible, sinon extrait les attributs manuellement.

        Args:
            booking: Objet booking ou dict

        Returns:
            Dict avec clés : id, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
        """
        if hasattr(booking, "to_dict"):
            return cast(Dict[str, Any], booking.to_dict())
        return {
            "id": getattr(booking, "id", None),
            "pickup_lat": getattr(booking, "pickup_lat", None),
            "pickup_lon": getattr(booking, "pickup_lon", None),
            "dropoff_lat": getattr(booking, "dropoff_lat", None),
            "dropoff_lon": getattr(booking, "dropoff_lon", None),
        }

    def _serialize_driver(self, driver: Any) -> Dict[str, Any]:
        """Sérialise un driver en dict API.

        Convertit un driver (objet ou dict) en format dict pour l'API.
        Utilise to_dict() si disponible, sinon extrait les attributs manuellement.

        Args:
            driver: Objet driver ou dict

        Returns:
            Dict avec clés : id, current_lat, current_lon
        """
        if hasattr(driver, "to_dict"):
            return cast(Dict[str, Any], driver.to_dict())
        return {
            "id": getattr(driver, "id", None),
            "current_lat": getattr(driver, "current_lat", None),
            "current_lon": getattr(driver, "current_lon", None),
        }
