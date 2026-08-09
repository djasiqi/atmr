"""Flags d'environnement partagés (boot WSGI, Socket.IO, relay, mobile startup)."""

from __future__ import annotations

import os
from typing import Any

IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV = "IOS_STARTUP_FATAL_RECOVERY_DISABLED"
# Kill-switch runtime : désactive l'ingestion GPS Socket.IO (chat + fanout cartes inchangés).
# Défaut true (opt-out) pour ne pas casser les builds anciens encore en socket-batch.
SOCKET_GPS_INGEST_ENABLED_ENV = "SOCKET_GPS_INGEST_ENABLED"


def env_truthy(name: str, default: str = "false") -> bool:
    """True si la variable vaut 1, true, yes (insensible à la casse)."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def is_skip_socketio() -> bool:
    return env_truthy("SKIP_SOCKETIO")


def is_ws_relay_publish_enabled() -> bool:
    return env_truthy("WS_RELAY_PUBLISH_ENABLED")


def is_socket_gps_ingest_enabled() -> bool:
    """True si l'ingestion GPS via Socket.IO (driver_location / batch) est autorisée.

    Ops urgence : SOCKET_GPS_INGEST_ENABLED=false — le client doit retenter via HTTP.
    """
    return env_truthy(SOCKET_GPS_INGEST_ENABLED_ENV, "true")


def is_ios_startup_fatal_recovery_disabled() -> bool:
    """Désarme le chemin fatal startup iOS côté client (builds qui consomment le flag).

    Ops: IOS_STARTUP_FATAL_RECOVERY_DISABLED=true
    Ne modifie pas les builds anciens qui n'implémentent pas la lecture du flag.
    """
    return env_truthy(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV)


def get_mobile_startup_runtime_flags() -> dict[str, bool | int]:
    """Flags startup mobile exposés via bootstrap/version-check."""
    flags: dict[str, bool | int] = {
        "ios_startup_fatal_recovery_disabled": is_ios_startup_fatal_recovery_disabled(),
    }
    flush_ms = os.getenv("COMPANY_MAP_REALTIME_FLUSH_MS", "").strip()
    if flush_ms.isdigit():
        flags["company_map_realtime_flush_ms"] = int(flush_ms)
    if os.getenv("COMPANY_MAP_DYNAMIC_FILTER_ENABLED") is not None:
        flags["company_map_dynamic_filter_enabled"] = env_truthy(
            "COMPANY_MAP_DYNAMIC_FILTER_ENABLED", "true"
        )
    if os.getenv("COMPANY_MAP_AUTOFIT_STRUCTURAL_ONLY") is not None:
        flags["company_map_autofit_structural_only"] = env_truthy(
            "COMPANY_MAP_AUTOFIT_STRUCTURAL_ONLY", "true"
        )
    if os.getenv("DRIVER_CAPTURE_AGGRESSIVE_ENABLED") is not None:
        flags["driver_capture_aggressive_enabled"] = env_truthy(
            "DRIVER_CAPTURE_AGGRESSIVE_ENABLED", "false"
        )
    if os.getenv("MOBILE_MAP_PARITY_MODE") is not None:
        flags["mobile_map_parity_mode"] = env_truthy("MOBILE_MAP_PARITY_MODE", "false")
    return flags


def get_runtime_flags_status() -> dict[str, Any]:
    """Statut ops des flags runtime (vérification sans redéploiement mobile)."""
    return {
        "skip_socketio": is_skip_socketio(),
        "ws_relay_publish_enabled": is_ws_relay_publish_enabled(),
        "socket_gps_ingest_enabled": is_socket_gps_ingest_enabled(),
        "mobile_startup": get_mobile_startup_runtime_flags(),
        "notes": {
            "ios_startup_fatal_recovery_disabled": (
                "Protection builds futurs uniquement; les builds anciens "
                "ignorent ce flag tant que le hotfix mobile n'est pas déployé."
            ),
            "socket_gps_ingest_enabled": (
                "Kill-switch ingestion GPS Socket.IO uniquement. "
                "false → ACK ingest_disabled + retry_event_ids, sans preuve durable. "
                "Chat et fanout cartes restent actifs."
            ),
        },
    }
