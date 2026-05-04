"""Agrégation pour GET /api/v1/platform/runtime — hors hot path (Phase 2)."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import sys
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Clés obligatoires pour chaque entrée de ``sections`` (contrat stable — tests non-régression).
RUNTIME_SECTION_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"status", "reason", "checked_at", "data"}
)

# Allowlist publique — ``sections.redis.data`` ne doit exposer **que** ces clés (+ cohérence interne).
REDIS_RUNTIME_PUBLIC_DATA_KEYS: frozenset[str] = frozenset(
    {
        "available",
        "ping_ok",
        "used_memory_bytes",
        "used_memory_human",
        "connected_clients",
        "uptime_in_seconds",
        "evicted_keys",
        "keyspace_hits",
        "keyspace_misses",
    }
)

# Sous-ensemble lu depuis INFO Redis (clés serveur) — ne pas étendre sans mise à jour doc / contrat.
_REDIS_INFO_KEYS = (
    "used_memory",
    "used_memory_human",
    "connected_clients",
    "uptime_in_seconds",
    "evicted_keys",
    "keyspace_hits",
    "keyspace_misses",
)

# Allowlist publique — ``sections.celery.data``.
CELERY_RUNTIME_PUBLIC_DATA_KEYS: frozenset[str] = frozenset(
    {
        "available",
        "inspect_ok",
        "workers_count",
        "workers",
        "broker_transport",
    }
)


def _now_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _section_not_implemented() -> dict[str, Any]:
    return {
        "status": "not_implemented",
        "reason": "not_implemented",
        "checked_at": None,
        "data": None,
    }


def _extract_redis_info_subset(info: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _REDIS_INFO_KEYS:
        if k not in info:
            continue
        if k == "used_memory":
            out["used_memory_bytes"] = info[k]
        else:
            out[k] = info[k]
    return out


def _redis_data_public_only(data: dict[str, Any]) -> dict[str, Any]:
    """Ne retourne que les clés allowlistées (défense en profondeur)."""
    return {k: v for k, v in data.items() if k in REDIS_RUNTIME_PUBLIC_DATA_KEYS}


def _celery_data_public_only(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k in CELERY_RUNTIME_PUBLIC_DATA_KEYS}


def _exception_indicates_timeout(exc: BaseException) -> bool:
    """Détection centralisée des timeouts (ne pas se fier uniquement au message texte)."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    et = type(exc).__name__.lower()
    if "timeout" in et:
        return True
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg


def _celery_stats_exploitable(stats: dict[str, Any] | None, workers: list[str]) -> bool:
    """``stats()`` exploitable : au moins un worker vu par ``ping`` a un payload stats non vide.

    Couvre : ``stats`` None / ``{}`` / clés sans recoupement avec ``ping`` / dicts vides par worker.
    """
    if stats is None or not isinstance(stats, dict) or not stats:
        return False
    if not workers:
        return False
    for worker_name in workers:
        if worker_name not in stats:
            continue
        payload = stats.get(worker_name)
        if isinstance(payload, dict) and len(payload) > 0:
            return True
        if payload is not None and not isinstance(payload, dict):
            return True
    return False


def _build_redis_section() -> dict[str, Any]:
    """INFO Redis minimal, connexion éphémère, timeout court — échec isolé (ne lève pas)."""
    checked = _now_iso_z()
    redis_url = (os.getenv("REDIS_URL") or "").strip() or "redis://redis:6379/0"
    try:
        timeout = float(os.getenv("PLATFORM_RUNTIME_REDIS_INFO_TIMEOUT_SECONDS", "1.5"))
    except ValueError:
        timeout = 1.5

    try:
        import redis as redis_mod

        r = redis_mod.Redis.from_url(
            redis_url,
            socket_timeout=timeout,
            socket_connect_timeout=timeout,
            decode_responses=True,
        )
        try:
            r.ping()
            info_raw = r.info()
        finally:
            with contextlib.suppress(Exception):
                r.close()
    except Exception as e:
        logger.debug("[platform/runtime] redis section: %s", e, exc_info=False)
        return {
            "status": "unknown",
            "reason": "redis_unreachable",
            "checked_at": checked,
            "data": None,
        }

    try:
        if not isinstance(info_raw, dict):
            raise TypeError("redis info is not a dict")
        subset = _extract_redis_info_subset(dict(info_raw))
        # Après PING OK : au moins une métrique mémoire attendue, sinon INFO considéré incomplet.
        if "used_memory_bytes" not in subset and "used_memory_human" not in subset:
            raise ValueError("redis info missing memory fields")
        data = _redis_data_public_only(
            {
                "available": True,
                "ping_ok": True,
                **subset,
            }
        )
        return {
            "status": "ok",
            "reason": None,
            "checked_at": _now_iso_z(),
            "data": data,
        }
    except Exception as e:
        logger.debug("[platform/runtime] redis subset: %s", e, exc_info=False)
        return {
            "status": "degraded",
            "reason": "redis_info_parse_failed",
            "checked_at": checked,
            "data": None,
        }


def _build_celery_section() -> dict[str, Any]:
    """Inspect Celery minimal (ping + stats optionnel) — timeout court, ne lève pas."""
    checked = _now_iso_z()
    try:
        timeout = float(
            os.getenv("PLATFORM_RUNTIME_CELERY_INSPECT_TIMEOUT_SECONDS", "1.5")
        )
    except ValueError:
        timeout = 1.5

    try:
        from celery_app import celery as celery_app
    except Exception as e:
        logger.debug("[platform/runtime] celery import: %s", e, exc_info=False)
        return {
            "status": "unknown",
            "reason": "celery_unreachable",
            "checked_at": checked,
            "data": None,
        }

    broker_transport = getattr(
        celery_app.conf,
        "broker_transport",
        None,
    ) or str(celery_app.conf.get("broker_transport") or "redis")

    try:
        insp = celery_app.control.inspect(timeout=timeout)
        ping_result = insp.ping()
    except Exception as e:
        logger.debug("[platform/runtime] celery inspect ping: %s", e, exc_info=False)
        reason = (
            "celery_inspect_timeout"
            if _exception_indicates_timeout(e)
            else "celery_inspect_failed"
        )
        return {
            "status": "unknown",
            "reason": reason,
            "checked_at": checked,
            "data": _celery_data_public_only(
                {
                    "available": True,
                    "inspect_ok": False,
                    "workers_count": 0,
                    "workers": [],
                    "broker_transport": broker_transport,
                }
            ),
        }

    if ping_result is None or not isinstance(ping_result, dict):
        return {
            "status": "unknown",
            "reason": "celery_unreachable",
            "checked_at": checked,
            "data": _celery_data_public_only(
                {
                    "available": True,
                    "inspect_ok": False,
                    "workers_count": 0,
                    "workers": [],
                    "broker_transport": broker_transport,
                }
            ),
        }

    workers = sorted(ping_result.keys())[:50]
    workers_count = len(workers)

    stats: dict[str, Any] | None = None
    try:
        stats = insp.stats()
    except Exception as e:
        logger.debug("[platform/runtime] celery stats: %s", e, exc_info=False)

    base_data = {
        "available": True,
        "inspect_ok": True,
        "workers_count": workers_count,
        "workers": workers,
        "broker_transport": broker_transport,
    }

    if workers_count == 0:
        return {
            "status": "unknown",
            "reason": "celery_unreachable",
            "checked_at": checked,
            "data": _celery_data_public_only(
                {
                    **base_data,
                    "inspect_ok": False,
                }
            ),
        }

    if not _celery_stats_exploitable(stats, workers):
        return {
            "status": "degraded",
            "reason": "celery_partial_data",
            "checked_at": _now_iso_z(),
            "data": _celery_data_public_only(base_data),
        }

    return {
        "status": "ok",
        "reason": None,
        "checked_at": _now_iso_z(),
        "data": _celery_data_public_only(base_data),
    }


def build_platform_runtime_payload() -> dict[str, Any]:
    """Corps JSON pour GET /api/v1/platform/runtime (forme stable, sections enrichissables).

    Compatibilité : nouvelles sections **additives** uniquement ; chaque section garde
    toujours ``status``, ``reason``, ``checked_at``, ``data`` — une section peut passer
    de ``not_implemented`` à ``ok`` / ``degraded`` / ``unknown`` sans changer de forme.

    Voir ``docs/PLATFORM_STATUS_CONTRACT.md``.
    """
    checked = _now_iso_z()
    return {
        "generated_at": checked,
        "sections": {
            "process": {
                "status": "ok",
                "reason": None,
                "checked_at": checked,
                "data": {
                    "pid": os.getpid(),
                    "python_version": sys.version.split()[0],
                },
            },
            "redis": _build_redis_section(),
            "celery": _build_celery_section(),
            "websocket": _section_not_implemented(),
            "dispatch": _section_not_implemented(),
            "gps_pipeline": _section_not_implemented(),
        },
    }


def assert_runtime_payload_section_contract(payload: dict[str, Any]) -> None:
    """Vérifie la forme contractuelle des sections (utilisable par les tests)."""
    sections = payload.get("sections") or {}
    for name, sec in sections.items():
        if not isinstance(sec, dict):
            raise AssertionError(f"section {name!r} must be a dict")
        keys = frozenset(sec.keys())
        if keys != RUNTIME_SECTION_REQUIRED_KEYS:
            raise AssertionError(
                f"section {name!r} keys {sorted(keys)} != "
                f"{sorted(RUNTIME_SECTION_REQUIRED_KEYS)}"
            )
