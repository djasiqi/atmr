"""Catalogue des services plateforme (aligné runtime / status)."""

from __future__ import annotations

from typing import Any


def list_platform_services() -> list[dict[str, Any]]:
    """Liste indicative — enrichissement depuis métriques LATER."""
    return [
        {
            "id": "api",
            "kind": "control_plane",
            "display_name": "API ATMR",
            "notes": "Processus WSGI / workers",
        },
        {
            "id": "redis",
            "kind": "infrastructure",
            "display_name": "Redis",
        },
        {
            "id": "celery",
            "kind": "async",
            "display_name": "Celery",
        },
        {
            "id": "websocket",
            "kind": "realtime",
            "display_name": "Socket.IO",
        },
        {
            "id": "dispatch",
            "kind": "domain",
            "display_name": "Dispatch",
        },
        {
            "id": "gps_pipeline",
            "kind": "domain",
            "display_name": "Pipeline GPS",
        },
    ]
