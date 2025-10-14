from __future__ import annotations

from flask_socketio import SocketIO

from services.socketio_service import emit_company_event


def emit_shift_created(company_id: int, payload: dict) -> None:
    emit_company_event(company_id, "planning:shift_created", payload)


def emit_shift_updated(company_id: int, payload: dict) -> None:
    emit_company_event(company_id, "planning:shift_updated", payload)


def emit_shift_deleted(company_id: int, payload: dict) -> None:
    emit_company_event(company_id, "planning:shift_deleted", payload)


def init_planning_socket(socketio: SocketIO) -> None:
    # Pas d'events à écouter côté serveur pour l'instant (uniquement émission côté HTTP)
    return None


