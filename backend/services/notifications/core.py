# backend/services/notification_service.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ext import app_logger
from repositories.dispatch_run_repository import DispatchRunRepository
from services.events.fanout import (
    fanout_booking_assigned_to_company,
    fanout_booking_assigned_to_driver,
    fanout_booking_cancelled,
    fanout_booking_updated,
    fanout_dispatch_run_completed,
    fanout_urgent_alert,
)
from services.realtime.socketio import emit_company_event

if TYPE_CHECKING:
    from models import Booking


# Note: _emit_room() a été supprimée car elle n'est plus utilisée.
# Le code utilise maintenant les fonctions de event_fanout.py qui appellent
# directement emit_driver_event() et emit_company_event() depuis socketio_service.py


# ---------- 1) Notification PUSH Expo ----------
# ✅ send_push_message est maintenant importé depuis push_service (ligne 11)
# pour éviter les cycles d'import avec socketio_service


# ---------- 2) WebSocket - nouvelle course ----------
def notify_driver_new_booking(
    driver_id: int,
    booking: Booking,
    *,
    event_id: str | None = None,
    correlation_id: str | None = None,
) -> None:
    """Notifie un chauffeur d'une nouvelle mission assignée.

    Fan-out hybride:
    1. Socket.IO (foreground)
    2. Push notification (background)
    """
    booking_id = getattr(booking, "id", None)
    booking_data: Dict[str, Any] | None = None
    try:
        booking_data = (
            booking.to_dict() if hasattr(booking, "to_dict") else {"id": booking_id}
        )
    except (ValueError, TypeError, AttributeError):
        booking_data = {"id": booking_id}

    from services.notifications.push_pipeline_log import log_driver_push_stage

    log_driver_push_stage(
        "driver_push.notify",
        event_id=event_id,
        correlation_id=correlation_id,
        booking_id=booking_id,
        driver_id=driver_id,
        notification_type="booking_assigned",
    )

    app_logger.warning(
        "[notify_driver_new_booking] Calling fanout_booking_assigned_to_driver: driver_id=%s booking_id=%s",
        driver_id,
        booking_id,
    )

    # ✅ Utiliser le service centralisé de fan-out
    fanout_booking_assigned_to_driver(
        driver_id=driver_id,
        booking_id=booking_id or 0,
        booking_data=booking_data,
        event_id=event_id,
        correlation_id=correlation_id,
    )

    app_logger.warning(
        "[notify_driver_new_booking] fanout_booking_assigned_to_driver completed for driver_id=%s booking_id=%s",
        driver_id,
        booking_id,
    )


# ---------- 3) WebSocket - mise à jour de mission ----------
def notify_booking_update(driver_id: int, booking: Booking) -> None:
    """Notifie un chauffeur d'une mise à jour de mission.

    Fan-out hybride:
    1. Socket.IO (foreground)
    2. Push notification (si changement critique)
    """
    booking_id = getattr(booking, "id", None)
    booking_data: Dict[str, Any] | None = None
    try:
        booking_data = (
            booking.to_dict() if hasattr(booking, "to_dict") else {"id": booking_id}
        )
    except (ValueError, TypeError, AttributeError):
        booking_data = {"id": booking_id}

    # Envoyer push seulement pour certains statuts importants
    status = getattr(booking, "status", None)
    critical_statuses = ["cancelled", "urgent", "delayed"]
    should_send_push = status in critical_statuses

    # ✅ Utiliser le service centralisé de fan-out
    fanout_booking_updated(
        driver_id=driver_id,
        booking_id=booking_id or 0,
        booking_data=booking_data,
        send_push=should_send_push,
    )


# ---------- 4) WebSocket - annulation de mission ----------
def notify_booking_cancelled(
    driver_id: int,
    booking_id: int,
    *,
    booking_data: Dict[str, Any] | None = None,
) -> None:
    """Notifie un chauffeur de l'annulation d'une mission.

    Fan-out hybride:
    1. Socket.IO (foreground)
    2. Push notification (background) - template "Course • Annulée" + Client • heure
    """
    fanout_booking_cancelled(
        driver_id=driver_id,
        booking_id=booking_id,
        booking_data=booking_data,
    )


def notify_booking_assigned(booking: Booking) -> None:
    """Notification unifiée quand une réservation est assignée.

    Fan-out hybride:
    1. Socket.IO (foreground)
    2. Push notification (background) - pour les dispatchers de l'entreprise
    """
    company_id = int(getattr(booking, "company_id", 0) or 0)
    booking_id = getattr(booking, "id", None)
    driver_id = getattr(booking, "driver_id", None)
    booking_data: Dict[str, Any] | None = None
    try:
        booking_data = (
            booking.to_dict() if hasattr(booking, "to_dict") else {"id": booking_id}
        )
    except (ValueError, TypeError, AttributeError):
        booking_data = {"id": booking_id}

    # ✅ Utiliser le service centralisé de fan-out (booking_data → message métier avec client_name)
    fanout_booking_assigned_to_company(
        company_id=company_id,
        booking_id=booking_id or 0,
        driver_id=driver_id,
        booking_data=booking_data,
    )


def notify_dispatch_run_completed(
    company_id: int,
    dispatch_run_id: int | str,
    assignments_count: int,
    date_str: str | None = None,
) -> None:
    """Notifie qu'un dispatch run est terminé.

    Fan-out hybride:
    1. Socket.IO (foreground)
    2. Push notification (background) - conditionnelle si urgent
    """
    # Si la date n'est pas fournie, on tente de la récupérer depuis la DB
    if not date_str:
        try:
            # ✅ Utilisation du repository pour découpler de SQLAlchemy
            dispatch_run_repo = DispatchRunRepository()
            # Convertir en int si nécessaire
            run_id = (
                int(dispatch_run_id)
                if isinstance(dispatch_run_id, str)
                else dispatch_run_id
            )
            dispatch_run_dto = dispatch_run_repo.find_by_id(run_id)
            if dispatch_run_dto and dispatch_run_dto.day:
                d = dispatch_run_dto.day
                date_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
                app_logger.info(
                    "[notify_dispatch_run_completed] Retrieved date_str=%s from dispatch_run_id=%s",
                    date_str,
                    dispatch_run_id,
                )
        except (ValueError, TypeError, AttributeError) as e:
            # Erreurs de validation attendues : conversion de types, attributs manquants
            app_logger.warning(
                "[notify_dispatch_run_completed] Failed to get day_str from DispatchRun (validation error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue lors de la récupération de date_str
            app_logger.exception(
                "[notify_dispatch_run_completed] Failed to get day_str from DispatchRun"
            )

    # ✅ Utiliser le service centralisé de fan-out
    fanout_dispatch_run_completed(
        company_id=company_id,
        dispatch_run_id=dispatch_run_id,
        assignments_count=assignments_count,
        date_str=date_str,
        send_push_if_urgent=True,
        urgent_threshold=10,
    )


def notify_dispatcher_optimization_opportunity(
    opportunity_data: Dict[str, Any],
) -> None:
    """Notifie le dispatcher d'une opportunité d'optimisation détectée.

    Args:
        opportunity_data: Dict contenant les détails de l'opportunité
            {
                "company_id": int,
                "assignment_id": int,
                "booking_id": int,
                "driver_id": int,
                "current_delay": int,
                "severity": str,
                "suggestions": List[Dict],
                "auto_apply": bool
            }

    """
    try:
        company_id = opportunity_data.get("company_id")
        if not company_id:
            app_logger.warning(
                "[notify_dispatcher_optimization_opportunity] No company_id in data"
            )
            return

        payload: Dict[str, Any] = {
            "type": "optimization_opportunity",
            "assignment_id": opportunity_data.get("assignment_id"),
            "booking_id": opportunity_data.get("booking_id"),
            "driver_id": opportunity_data.get("driver_id"),
            "current_delay": opportunity_data.get("current_delay"),
            "severity": opportunity_data.get("severity"),
            "suggestions": opportunity_data.get("suggestions", []),
            "auto_apply": opportunity_data.get("auto_apply", False),
        }

        app_logger.info(
            "[notify_dispatcher_optimization_opportunity] Emitting to company %s: severity=%s delay=%d",
            company_id,
            payload.get("severity"),
            payload.get("current_delay"),
        )

        emit_company_event(company_id, "optimization_opportunity", payload)
    except (ValueError, TypeError, AttributeError) as e:
        # Erreurs de validation attendues : conversion de types, attributs manquants
        app_logger.error(
            "[notify_dispatcher_optimization_opportunity] emit failed (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        app_logger.error(
            "[notify_dispatcher_optimization_opportunity] emit failed (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        # Erreur inattendue : logger avec trace complète
        app_logger.exception("[notify_dispatcher_optimization_opportunity] emit failed")


def notify_urgent_alert(
    company_id: int,
    alert_id: int | str,
    alert_type: str,
    message: str,
    booking_id: int | None = None,
    driver_id: int | None = None,
    severity: str = "high",
) -> None:
    """Notifie une alerte urgente avec fan-out hybride.

    Fan-out hybride:
    1. Socket.IO (foreground)
    2. Push notification (background)
    3. Deep link pour navigation

    Args:
        company_id: ID de l'entreprise
        alert_id: ID unique de l'alerte
        alert_type: Type d'alerte (ex: "delay", "driver_unavailable",
            "booking_conflict")
        message: Message de l'alerte
        booking_id: ID du booking concerné (optionnel)
        driver_id: ID du chauffeur concerné (optionnel)
        severity: Niveau de sévérité ("low", "medium", "high", "critical")
    """
    # ✅ Utiliser le service centralisé de fan-out
    fanout_urgent_alert(
        company_id=company_id,
        alert_id=alert_id,
        alert_type=alert_type,
        message=message,
        severity=severity,
        booking_id=booking_id,
        driver_id=driver_id,
    )
