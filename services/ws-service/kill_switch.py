"""Kill switch connexions ws-service + drain / force disconnect."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

logger = logging.getLogger("ws-service.kill_switch")

DRAIN_SEC = float(os.getenv("WS_KILL_SWITCH_DRAIN_SEC", "30"))
_force_disconnect_total = 0
_kill_switch_active = False


def connections_accepted() -> bool:
    return os.getenv("WS_SERVICE_ACCEPT_CONNECTIONS", "true").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def is_kill_switch_engaged() -> bool:
    return not connections_accepted() or _kill_switch_active


def engage_kill_switch() -> None:
    global _kill_switch_active
    _kill_switch_active = True


def disengage_kill_switch() -> None:
    """Reset interne (ops/tests). N'altère pas WS_SERVICE_ACCEPT_CONNECTIONS."""
    global _kill_switch_active
    _kill_switch_active = False


def _collect_local_sids(sio: Any) -> set[str]:
    """Collecte les SID locaux du namespace racine.

    Tente plusieurs APIs python-socketio (BaseManager, AsyncRedisManager) car
    `manager.rooms` peut varier selon la version installée.
    """
    sids: set[str] = set()
    try:
        namespace_rooms = getattr(sio.manager, "rooms", None)
        if isinstance(namespace_rooms, dict):
            ns_dict = namespace_rooms.get("/", {})
            if isinstance(ns_dict, dict):
                for members in ns_dict.values():
                    if isinstance(members, dict):
                        sids.update(members.keys())
                    elif isinstance(members, (set, list)):
                        sids.update(members)
    except Exception:
        logger.exception("collect sids via manager.rooms failed")

    if not sids:
        # Fallback : BaseManager garde un mapping eio_sid -> sid via get_participants
        try:
            iterator = sio.manager.get_participants("/", None)
            for entry in iterator:
                if isinstance(entry, tuple) and entry:
                    sids.add(entry[0])
                elif isinstance(entry, str):
                    sids.add(entry)
        except Exception:
            pass
    return sids


async def drain_and_force_disconnect(sio: Any) -> None:
    """Après drain timeout, déconnecte les SID restants."""
    global _force_disconnect_total
    await asyncio.sleep(DRAIN_SEC)
    try:
        sids = _collect_local_sids(sio)
        logger.warning(
            "kill switch drain elapsed=%.0fs collected_sids=%s", DRAIN_SEC, len(sids)
        )
        for sid in list(sids):
            try:
                await sio.disconnect(sid)
                _force_disconnect_total += 1
            except Exception:
                logger.exception("force disconnect failed sid=%s", sid)
        if sids:
            logger.warning(
                "kill switch force disconnect count=%s after drain %.0fs",
                len(sids),
                DRAIN_SEC,
            )
    except Exception:
        logger.exception("kill switch drain failed")


def force_disconnect_total() -> int:
    return _force_disconnect_total
