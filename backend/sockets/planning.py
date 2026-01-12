from __future__ import annotations

from typing import TYPE_CHECKING, Any

from services.realtime.socketio import emit_company_event

if TYPE_CHECKING:
    from flask_socketio import SocketIO  # pyright: ignore[reportMissingModuleSource]


# ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
def emit_shift_created(company_id: int, payload: dict[str, Any]) -> None:
    emit_company_event(company_id, "planning_shift_created", payload)


def emit_shift_updated(company_id: int, payload: dict[str, Any]) -> None:
    emit_company_event(company_id, "planning_shift_updated", payload)


def emit_shift_deleted(company_id: int, payload: dict[str, Any]) -> None:
    emit_company_event(company_id, "planning_shift_deleted", payload)


def init_planning_socket(socketio: SocketIO) -> None:  # noqa: ARG001
    # Pas d'events à écouter côté serveur pour l'instant (uniquement émission côté HTTP)
    return None
