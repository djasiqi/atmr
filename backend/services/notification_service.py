# backend/services/notification_service.py
from __future__ import annotations

from typing import Any, Dict, Optional, cast  # <-- ajout de cast

import requests

from ext import socketio, app_logger
from models import Booking

# ---------- Helper d'émission compatible v4/v5 ----------
def _emit_room(event: str, payload: Dict[str, Any], room: str, *, namespace: str = "/") -> None:
    """Émet un event vers une room en essayant d’abord `to=`, puis fallback `room=`."""
    try:
        socketio.emit(event, payload, to=room, namespace=namespace)  # Flask-SocketIO >= 5
    except TypeError:
        try:
            sio_any = cast(Any, socketio)  # <-- cast pour calmer Pylance
            sio_any.emit(event, payload, room=room, namespace=namespace)  # compat v4
            # (alternativement: socketio.emit(..., room=..., namespace=...)  # type: ignore[call-arg])
        except Exception as e:
            app_logger.error("[notify] emit (compat) failed event=%s room=%s err=%s", event, room, e)
    except Exception as e:
        app_logger.error("[notify] emit failed event=%s room=%s err=%s", event, room, e)


# ---------- 1) Notification PUSH Expo ----------
def send_push_message(token: str, title: str, body: str, *, timeout: int = 5) -> Dict[str, Any]:
    message = {
        "to": token,
        "sound": "default",
        "title": title,
        "body": body,
    }
    try:
        resp = requests.post("https://exp.host/--/api/v2/push/send", json=message, timeout=timeout)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]
    except Exception as e:
        app_logger.warning("[notify] Expo push failed: %s", e)
        return {"ok": False, "error": str(e)}


# ---------- 2) WebSocket – nouvelle course ----------
def notify_driver_new_booking(driver_id: int, booking: Booking) -> None:
    room = f"driver_{driver_id}"
    try:
        payload: Dict[str, Any]
        if hasattr(booking, "to_dict"):
            payload = booking.to_dict()  # type: ignore[assignment]
        else:
            payload = {"id": getattr(booking, "id", None)}
        _emit_room("new_booking", payload, room)
    except Exception as e:
        app_logger.error("[notify_driver_new_booking] failed: %s", e)


# ---------- 3) WebSocket – mise à jour de mission ----------
def notify_booking_update(driver_id: int, booking: Booking) -> None:
    room = f"driver_{driver_id}"
    try:
        payload: Dict[str, Any]
        if hasattr(booking, "to_dict"):
            payload = booking.to_dict()  # type: ignore[assignment]
        else:
            payload = {"id": getattr(booking, "id", None)}
        _emit_room("booking_updated", payload, room)
    except Exception as e:
        app_logger.error("[notify_booking_update] failed: %s", e)


# ---------- 4) WebSocket – annulation de mission ----------
def notify_booking_cancelled(driver_id: int, booking_id: int) -> None:
    room = f"driver_{driver_id}"
    try:
        _emit_room("booking_cancelled", {"id": booking_id}, room)
    except Exception as e:
        app_logger.error("[notify_booking_cancelled] failed: %s", e)


def notify_booking_assigned(booking: Booking) -> None:
    """
    Notification unifiée quand une réservation est assignée.
    - Émet un événement SocketIO côté entreprise
    """
    try:
        from services.socketio_service import emit_company_event
        payload = {
            "booking_id": getattr(booking, "id", None),
            "driver_id": getattr(booking, "driver_id", None),
            "status": str(getattr(booking, "status", "")) if hasattr(booking, "status") else None,
        }
        emit_company_event(int(getattr(booking, "company_id", 0) or 0), "booking_assigned", payload)
    except Exception as e:
        app_logger.error("[notify_booking_assigned] emit failed: %s", e)


def notify_dispatch_run_completed(
    company_id: int,
    dispatch_run_id: int | str,
    assignments_count: int,
    date_str: Optional[str] = None,
) -> None:
    """
    Notification unifiée quand un run de dispatch est terminé.
    """
    try:
        from services.socketio_service import emit_company_event, emit_date_event

        # Si la date n’est pas fournie, on tente de la récupérer depuis la DB
        if not date_str:
            try:
                from models import DispatchRun
                dr = DispatchRun.query.get(dispatch_run_id)  # type: ignore[arg-type]
                if dr and getattr(dr, "day", None):
                    d = getattr(dr, "day")
                    date_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
                    app_logger.info(
                        "[notify_dispatch_run_completed] Retrieved date_str=%s from dispatch_run_id=%s",
                        date_str, dispatch_run_id
                    )
            except Exception as e:
                app_logger.warning(
                    "[notify_dispatch_run_completed] Failed to get day_str from DispatchRun: %s", e
                )

        payload: Dict[str, Any] = {
            "dispatch_run_id": dispatch_run_id,
            "assignments_count": int(assignments_count),
            "date": date_str,
        }
        app_logger.info("[notify_dispatch_run_completed] Emitting payload: %s", payload)

        emit_company_event(company_id, "dispatch_run_completed", payload)
        if date_str:
            emit_date_event(date_str, "dispatch_run_completed", payload)
    except Exception as e:
        app_logger.error("[notify_dispatch_run_completed] emit failed: %s", e)


def notify_dispatcher_optimization_opportunity(opportunity_data: Dict[str, Any]) -> None:
    """
    Notifie le dispatcher d'une opportunité d'optimisation détectée.
    
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
        from services.socketio_service import emit_company_event
        
        company_id = opportunity_data.get("company_id")
        if not company_id:
            app_logger.warning("[notify_dispatcher_optimization_opportunity] No company_id in data")
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
            payload.get("current_delay")
        )
        
        emit_company_event(company_id, "optimization_opportunity", payload)
    except Exception as e:
        app_logger.error("[notify_dispatcher_optimization_opportunity] emit failed: %s", e)