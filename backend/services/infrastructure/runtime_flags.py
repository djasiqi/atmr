"""Flags d'environnement partagés (boot WSGI, Socket.IO, relay, mobile startup)."""

from __future__ import annotations

import os
from typing import Any

IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV = "IOS_STARTUP_FATAL_RECOVERY_DISABLED"


def env_truthy(name: str, default: str = "false") -> bool:
    """True si la variable vaut 1, true, yes (insensible à la casse)."""
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def is_skip_socketio() -> bool:
    return env_truthy("SKIP_SOCKETIO")


def is_ws_relay_publish_enabled() -> bool:
    return env_truthy("WS_RELAY_PUBLISH_ENABLED")


def is_ios_startup_fatal_recovery_disabled() -> bool:
    """Désarme le chemin fatal startup iOS côté client (builds qui consomment le flag).

    Ops: IOS_STARTUP_FATAL_RECOVERY_DISABLED=true
    Ne modifie pas les builds anciens qui n'implémentent pas la lecture du flag.
    """
    return env_truthy(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV)


def get_mobile_startup_runtime_flags() -> dict[str, bool]:
    """Flags startup mobile exposés via bootstrap/version-check."""
    return {
        "ios_startup_fatal_recovery_disabled": is_ios_startup_fatal_recovery_disabled(),
    }


def get_runtime_flags_status() -> dict[str, Any]:
    """Statut ops des flags runtime (vérification sans redéploiement mobile)."""
    return {
        "skip_socketio": is_skip_socketio(),
        "ws_relay_publish_enabled": is_ws_relay_publish_enabled(),
        "mobile_startup": get_mobile_startup_runtime_flags(),
        "notes": {
            "ios_startup_fatal_recovery_disabled": (
                "Protection builds futurs uniquement; les builds anciens "
                "ignorent ce flag tant que le hotfix mobile n'est pas déployé."
            ),
        },
    }
