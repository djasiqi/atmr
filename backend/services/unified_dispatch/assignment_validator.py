# backend/services/unified_dispatch/assignment_validator.py
"""Validateur pour vérifier les contraintes après RL."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, List, Tuple, cast

from models import Booking
from models.dispatch import Assignment

logger = logging.getLogger(__name__)


class AssignmentValidator:
    """Valide les assignations après fusion RL."""

    def __init__(self, settings: Any):
        """Initialise le validateur.

        Args:
            settings: Configuration settings avec RLSettings
        """
        super().__init__()
        self.settings = settings
        self.min_minutes_before_pickup = getattr(settings.rl, "min_minutes_before_pickup", 10)

    def validate_assignments(
        self, assignments: List[Assignment], bookings: List[Booking], drivers: List[Any]
    ) -> Tuple[bool, List[str]]:
        """Valide une liste d'assignations.

        Args:
            assignments: Liste d'assignations
            bookings: Liste de bookings
            drivers: Liste de drivers

        Returns:
            Tuple (is_valid, violations)
        """
        violations = []

        # Créer des dictionnaires pour accès rapide
        bookings_dict = {b.id: b for b in bookings}
        drivers_dict = {d.id: d for d in drivers}

        # Vérifier chaque assignation
        for assignment in assignments:
            try:
                booking_id = int(cast("Any", assignment.booking_id))
                driver_id = int(cast("Any", assignment.driver_id))
            except (ValueError, TypeError):
                continue
            booking = bookings_dict.get(booking_id)
            driver = drivers_dict.get(driver_id)

            if not booking or not driver:
                violations.append(f"Assignment {assignment.id}: booking or driver not found")
                continue

            # Vérifier min_minutes_before_pickup
            if not self._check_min_minutes_before_pickup(booking, driver):
                violations.append(
                    f"Assignment {assignment.id}: violates min_minutes_before_pickup ({self.min_minutes_before_pickup} min)"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def _check_min_minutes_before_pickup(self, booking: Booking, _driver: Any) -> bool:
        """Vérifie que le pickup n'est pas trop proche dans le temps.

        Args:
            booking: Booking à vérifier
            driver: Driver assigné

        Returns:
            True si la contrainte est respectée
        """
        if not hasattr(booking, "scheduled_time"):
            return True

        scheduled_time = booking.scheduled_time
        if not scheduled_time:
            return True

        now = datetime.now(UTC)
        time_diff = (scheduled_time - now).total_seconds() / 60  # minutes

        # Si le pickup est dans moins de min_minutes_before_pickup minutes,
        # on ne doit pas réassigner
        if 0 < time_diff < self.min_minutes_before_pickup:
            logger.warning(
                "[Validator] Booking %s pickup in %.1f min (< %d min threshold)",
                booking.id,
                time_diff,
                self.min_minutes_before_pickup,
            )
            return False

        return True
