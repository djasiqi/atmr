# backend/services/socketio_service.py
from __future__ import annotations

from typing import Any, Optional
import json

from ext import socketio, app_logger
from models import Booking

# ---------------------------------------------------------------------------
# Constantes simples
# ---------------------------------------------------------------------------
DEFAULT_NAMESPACE = "/"


# ---------------------------------------------------------------------------
# Helpers de rooms (source de vérité unique)
# ---------------------------------------------------------------------------
def get_company_room(company_id: int) -> str:
    """Room d'entreprise (ex: company_42)."""
    return f"company_{company_id}"


def get_driver_room(driver_id: int) -> str:
    """Room personnelle d'un chauffeur (ex: driver_101)."""
    return f"driver_{driver_id}"


def get_date_room(date_str: str) -> str:
    """Room par date locale 'YYYY-MM-DD' (ex: date_2025-09-20)."""
    return f"date_{date_str}"


# ---------------------------------------------------------------------------
# Garde-fous utilitaires
# ---------------------------------------------------------------------------
def _is_jsonable(x: Any) -> bool:
    """Vérifie qu'un payload est sérialisable en JSON (évite les plantages silencieux)."""
    try:
        json.dumps(x)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Émission thread-safe (depuis handlers HTTP, workers, threads…)
# - Flask-SocketIO >= 5: 'to='
# - Compat anciennes versions: 'room='
# ---------------------------------------------------------------------------
def _safe_emit(
    event: str,
    payload: dict[str, Any],
    *,
    room: Optional[str] = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """
    Émet un événement Socket.IO de façon sûre:
      - exige une room (sinon log d'erreur), 
      - vérifie la sérialisabilité JSON,
      - gère la compatibilité Flask-SocketIO v4/v5,
      - ne remonte pas d'exception aux appelants.
    """
    if room is None:
        app_logger.error("[socketio] _safe_emit sans room: event=%s", event)
        return

    if not _is_jsonable(payload):
        app_logger.error("[socketio] payload non-JSON pour event=%s room=%s", event, room)
        return

    try:
        # Flask-SocketIO >= 5
        socketio.emit(event, payload, to=room, namespace=namespace)
    except TypeError:
        # Compat < 5.x (param 'room')
        try:
            socketio.emit(event, payload, room=room, namespace=namespace)
        except Exception as e:
            app_logger.error(
                "[socketio] emit failed (compat) event=%s room=%s err=%s", event, room, e
            )
    except Exception as e:
        app_logger.error("[socketio] emit failed event=%s room=%s err=%s", event, room, e)


# ---------------------------------------------------------------------------
# Helpers "métier" d'émission
# ---------------------------------------------------------------------------
def notify_driver_new_booking(
    driver_id: int, booking: Booking, *, namespace: str = DEFAULT_NAMESPACE
) -> None:
    """
    Émet 'new_booking' vers la room du chauffeur correspondant.
    """
    try:
        data = booking.to_dict() if hasattr(booking, "to_dict") else {
            "id": getattr(booking, "id", None)
        }
    except Exception:
        # fallback minimal en cas de serialization tricky
        data = {"id": getattr(booking, "id", None)}

    _safe_emit("new_booking", data, room=get_driver_room(driver_id), namespace=namespace)


def emit_driver_event(
    driver_id: int,
    event: str,
    payload: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Émet un événement générique vers un chauffeur (room driver_...)."""
    _safe_emit(event, payload, room=get_driver_room(driver_id), namespace=namespace)


def emit_company_event(
    company_id: int,
    event: str,
    payload: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """
    Émet un événement SocketIO dans la room de l’entreprise (thread-safe).
    Utilise 'to=' si dispo, sinon 'room=' (compat v4/v5).
    Ne lève pas d’exception : log l’erreur si l’envoi échoue.
    """
    _safe_emit(event, payload, room=get_company_room(company_id), namespace=namespace)


def emit_date_event(
    date_str: str,
    event: str,
    payload: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """Émet un événement vers la room d'une date (utile pour vues par journée)."""
    _safe_emit(event, payload, room=get_date_room(date_str), namespace=namespace)


# Évènements typés du moteur/dispatch
def emit_dispatch_run_started(
    company_id: int,
    dispatch_run_id: str,
    date_str: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    emit_company_event(
        company_id,
        "dispatch:run:started",
        {"dispatch_run_id": dispatch_run_id, "date": date_str},
        namespace=namespace,
    )
    # Optionnel: cibler aussi la room date_YYYY-MM-DD
    emit_date_event(
        date_str,
        "dispatch:run:started",
        {"dispatch_run_id": dispatch_run_id, "date": date_str},
        namespace=namespace,
    )


def emit_dispatch_run_completed(
    company_id: int,
    dispatch_run_id: str,
    date_str: str,
    assignments_count: int,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "dispatch_run_id": dispatch_run_id,
        "date": date_str,
        "assignments_count": int(assignments_count),
    }
    # Change these event names to match what the frontend is expecting
    emit_company_event(company_id, "dispatch_run_completed", payload, namespace=namespace)
    emit_date_event(date_str, "dispatch_run_completed", payload, namespace=namespace)


def emit_dispatch_run_failed(
    company_id: int,
    dispatch_run_id: str,
    date_str: str,
    error: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "dispatch_run_id": dispatch_run_id,
        "date": date_str,
        "error": str(error),
    }
    emit_company_event(company_id, "dispatch:run:failed", payload, namespace=namespace)
    emit_date_event(date_str, "dispatch:run:failed", payload, namespace=namespace)


def emit_assignment_created(
    company_id: int,
    booking_id: int,
    driver_id: int,
    assignment_id: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """
    Notifie la création d'une assignation :
      - room entreprise (tableau de bord),
      - room chauffeur (réception tâche),
      - (optionnel) room booking si vous la gérez côté front.
    """
    company_payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
    }
    emit_company_event(
        company_id, "dispatch:assignment:created", company_payload, namespace=namespace
    )

    driver_payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
    }
    emit_driver_event(
        driver_id, "driver:assignment:received", driver_payload, namespace=namespace
    )


def emit_assignment_updated(
    company_id: int,
    assignment_id: str,
    booking_id: int,
    driver_id: int,
    fields: dict[str, Any],
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
        "fields": fields,
    }
    emit_company_event(
        company_id, "dispatch:assignment:updated", payload, namespace=namespace
    )
    emit_driver_event(
        driver_id, "driver:assignment:updated", payload, namespace=namespace
    )


def emit_assignment_cancelled(
    company_id: int,
    assignment_id: str,
    booking_id: int,
    driver_id: int,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
    }
    emit_company_event(
        company_id, "dispatch:assignment:cancelled", payload, namespace=namespace
    )
    emit_driver_event(
        driver_id, "driver:assignment:cancelled", payload, namespace=namespace
    )


def emit_delay_detected(
    company_id: int,
    booking_id: int,
    assignment_id: str,
    driver_id: int,
    delay_minutes: float,
    *,
    has_alternative: bool = False,
    alternative_driver_id: Optional[int] = None,
    alternative_delay_minutes: Optional[float] = None,
    is_dropoff: bool = False,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    payload: dict[str, Any] = {
        "assignment_id": assignment_id,
        "booking_id": booking_id,
        "driver_id": driver_id,
        "delay_minutes": float(delay_minutes),
        "has_alternative": bool(has_alternative),
        "is_dropoff": bool(is_dropoff),
    }
    if has_alternative and alternative_driver_id is not None:
        payload["alternative_driver_id"] = int(alternative_driver_id)
    if alternative_delay_minutes is not None:
        payload["alternative_delay_minutes"] = float(alternative_delay_minutes)

    emit_company_event(company_id, "dispatch:delay:detected", payload, namespace=namespace)
    emit_driver_event(driver_id, "driver:delay:detected", payload, namespace=namespace)


# ---------------------------------------------------------------------------
# Helpers pour joindre/quitter des rooms côté serveur (utilisable hors handler)
# ---------------------------------------------------------------------------
def join_company_room(sid: str, company_id: int, namespace: str = DEFAULT_NAMESPACE) -> None:
    """Ajoute un client (sid) à la room d’entreprise — utilisable hors contexte handler."""
    try:
        socketio.enter_room(sid, get_company_room(company_id), namespace=namespace)
    except Exception as e:
        app_logger.error(
            "[socketio] enter_room failed sid=%s company=%s err=%s", sid, company_id, e
        )


def leave_company_room(sid: str, company_id: int, namespace: str = DEFAULT_NAMESPACE) -> None:
    """Retire un client (sid) de la room d’entreprise — utilisable hors contexte handler."""
    try:
        socketio.leave_room(sid, get_company_room(company_id), namespace=namespace)
    except Exception as e:
        app_logger.error(
            "[socketio] leave_room failed sid=%s company=%s err=%s", sid, company_id, e
        )


def join_date_room(sid: str, date_str: str, namespace: str = DEFAULT_NAMESPACE) -> None:
    """Ajoute un client (sid) à la room de date (YYYY-MM-DD)."""
    try:
        socketio.enter_room(sid, get_date_room(date_str), namespace=namespace)
    except Exception as e:
        app_logger.error(
            "[socketio] enter_room(date) failed sid=%s date=%s err=%s", sid, date_str, e
        )


def leave_date_room(sid: str, date_str: str, namespace: str = DEFAULT_NAMESPACE) -> None:
    """Retire un client (sid) de la room de date (YYYY-MM-DD)."""
    try:
        socketio.leave_room(sid, get_date_room(date_str), namespace=namespace)
    except Exception as e:
        app_logger.error(
            "[socketio] leave_room(date) failed sid=%s date=%s err=%s", sid, date_str, e
        )


# ---------------------------------------------------------------------------
# Exports publics explicites (facultatif)
# ---------------------------------------------------------------------------
__all__ = [
    "DEFAULT_NAMESPACE",
    "get_company_room",
    "get_driver_room",
    "get_date_room",
    "_safe_emit",
    "emit_company_event",
    "emit_driver_event",
    "emit_date_event",
    "notify_driver_new_booking",
    "emit_dispatch_run_started",
    "emit_dispatch_run_completed",
    "emit_dispatch_run_failed",
    "emit_assignment_created",
    "emit_assignment_updated",
    "emit_assignment_cancelled",
    "emit_delay_detected",
    "join_company_room",
    "leave_company_room",
    "join_date_room",
    "leave_date_room",
]
