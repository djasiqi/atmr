from __future__ import annotations

from typing import TYPE_CHECKING, Any

from services.socketio_service import emit_company_event

if TYPE_CHECKING:
    from flask_socketio import SocketIO


def emit_shift_created(company_id: int, payload: dict[str, Any]) -> None:
    emit_company_event(company_id, "planning:shift_created", payload)


def emit_shift_updated(company_id: int, payload: dict[str, Any]) -> None:
    emit_company_event(company_id, "planning:shift_updated", payload)


def emit_shift_deleted(company_id: int, payload: dict[str, Any]) -> None:
    emit_company_event(company_id, "planning:shift_deleted", payload)


def init_planning_socket(socketio: SocketIO) -> None:  # noqa: ARG001
    # Pas d'events à écouter côté serveur pour l'instant (uniquement émission côté HTTP)
    return None
