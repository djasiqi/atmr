"""Analyseur des raisons de non-assignation des bookings."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Constantes pour éviter les valeurs magiques
DISTANCE_THRESHOLD_KM = 0.1  # ~1km en degrés


class UnassignedAnalyzer:
    """Analyse les raisons détaillées pour lesquelles certaines courses n'ont pas pu être assignées."""

    def analyze(
        self,
        problem: dict[str, Any],
        assignments: list[Any],  # noqa: ARG002 - Argument conservé pour compatibilité API
        unassigned_ids: list[int],
    ) -> dict[int, list[str]]:
        """Analyse les raisons de non-assignation.

        Args:
            problem: Dictionnaire contenant les bookings et drivers
            assignments: Liste des assignations (non utilisé mais conservé pour compatibilité)
            unassigned_ids: Liste des IDs de bookings non assignés

        Returns:
            Dictionnaire {booking_id: [raison1, raison2, ...]}
        """
        reasons: dict[int, list[str]] = {}
        bookings = problem.get("bookings", [])
        drivers = problem.get("drivers", [])

        # Créer un dictionnaire pour un accès rapide
        bookings_dict = {b.id: b for b in bookings}

        for booking_id in unassigned_ids:
            booking = bookings_dict.get(booking_id)
            if not booking:
                reasons[booking_id] = ["booking_not_found"]
                continue

            booking_reasons: list[str] = []

            # Vérifier la disponibilité des chauffeurs
            available_drivers = [d for d in drivers if getattr(d, "is_available", True)]
            if not available_drivers:
                booking_reasons.append("no_driver_available")

            # Vérifier la capacité
            if hasattr(booking, "capacity_required") and booking.capacity_required:
                suitable_drivers = [
                    d
                    for d in available_drivers
                    if hasattr(d, "capacity")
                    and d.capacity >= booking.capacity_required
                ]
                if not suitable_drivers:
                    booking_reasons.append("capacity_exceeded")

            # Vérifier les fenêtres horaires
            if hasattr(booking, "scheduled_time") and booking.scheduled_time:
                # Vérifier si l'heure est dans une fenêtre de travail
                booking_time = booking.scheduled_time
                working_drivers = []
                for driver in available_drivers:
                    if hasattr(driver, "work_windows") and driver.work_windows:
                        for window in driver.work_windows:
                            if window.start <= booking_time <= window.end:
                                working_drivers.append(driver)
                                break

                if not working_drivers:
                    booking_reasons.append("time_window_infeasible")

            # Vérifier les contraintes géographiques
            if hasattr(booking, "pickup_lat") and hasattr(booking, "pickup_lon"):
                # Vérifier si des chauffeurs sont dans la zone
                nearby_drivers = []
                for driver in available_drivers:
                    if hasattr(driver, "current_lat") and hasattr(
                        driver, "current_lon"
                    ):
                        # Calculer la distance (simplifié)
                        distance = (
                            (booking.pickup_lat - driver.current_lat) ** 2
                            + (booking.pickup_lon - driver.current_lon) ** 2
                        ) ** 0.5
                        if distance < DISTANCE_THRESHOLD_KM:  # ~1km
                            nearby_drivers.append(driver)

                if not nearby_drivers:
                    booking_reasons.append("no_nearby_drivers")

            # Vérifier les contraintes d'urgence
            if hasattr(booking, "is_emergency") and booking.is_emergency:
                emergency_drivers = [
                    d
                    for d in available_drivers
                    if hasattr(d, "can_handle_emergency") and d.can_handle_emergency
                ]
                if not emergency_drivers:
                    booking_reasons.append("no_emergency_drivers")

            # Si aucune raison spécifique n'a été trouvée
            if not booking_reasons:
                booking_reasons.append("unknown_constraint")

            reasons[booking_id] = booking_reasons

        return reasons
