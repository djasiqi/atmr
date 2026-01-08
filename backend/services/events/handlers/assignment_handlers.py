"""Event handlers pour les événements d'assignments."""

from __future__ import annotations

import logging
from typing import Any

from services.realtime.socketio import emit_assignment_cancelled

logger = logging.getLogger(__name__)


def handle_assignment_cancelled(event: dict[str, Any]) -> None:
    """Gère l'annulation d'un assignment.

    Notifie via Socket.IO la company et le driver que l'assignment a été annulé.
    """
    company_id = event.get("company_id")
    assignment_id = event.get("assignment_id")
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")

    if not all([company_id, assignment_id, booking_id, driver_id]):
        logger.warning(
            "[EventBus] AssignmentCancelledEvent missing required fields: %s",
            event,
        )
        return

    try:
        # Conversion sécurisée avec validation
        company_id_int = int(company_id) if company_id is not None else 0
        booking_id_int = int(booking_id) if booking_id is not None else 0
        driver_id_int = int(driver_id) if driver_id is not None else 0
        assignment_id_str = str(assignment_id) if assignment_id is not None else ""

        if not all([company_id_int, booking_id_int, driver_id_int, assignment_id_str]):
            logger.warning(
                "[EventBus] AssignmentCancelledEvent invalid values: %s",
                event,
            )
            return

        emit_assignment_cancelled(
            company_id=company_id_int,
            assignment_id=assignment_id_str,
            booking_id=booking_id_int,
            driver_id=driver_id_int,
        )
        logger.info(
            "✅ Assignment cancelled notification sent: assignment_id=%s, booking_id=%s, driver_id=%s",
            assignment_id,
            booking_id,
            driver_id,
        )
    except (ValueError, TypeError, AttributeError) as e:
        # Erreurs de validation attendues : conversion de types, attributs manquants
        logger.warning(
            "❌ Failed to send assignment cancelled notification (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        logger.warning(
            "❌ Failed to send assignment cancelled notification (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        # Erreur inattendue : logger avec trace complète
        logger.exception("❌ Failed to send assignment cancelled notification")
