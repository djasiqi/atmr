"""État santé device chauffeur (Redis) — canal séparé du GPS.

Le mobile remonte ici l'état de l'application (foreground service vivant,
permissions GPS, OEM battery optimization, taux de succès des fixes…) via
``POST /api/v1/driver/me/device-status``. Cette information permet de
distinguer côté backend :

* "téléphone éteint / app non lancée" → ``presence_status = offline``,
* "app vivante mais GPS bloqué par OEM (One UI, Doze…)" →
  ``presence_status = degraded_constrained`` (cf.
  :func:`services.geolocation.presence.apply_device_health_override`).

Le hash Redis ``driver:{id}:device_health`` est volontairement disjoint des
clés ``driver:{id}:loc:*`` pour ne pas polluer le pipeline GPS et garder un
TTL court (``DEVICE_HEALTH_TTL_SEC``, 120 s par défaut).
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


DEVICE_HEALTH_TTL_SEC = 120
"""Durée de vie du hash Redis device_health (s).

L'override ``apply_device_health_override`` exige en plus une fraîcheur
``< 120 s`` (cf. ``DEVICE_HEALTH_FRESH_SEC`` dans ``presence.py``) pour
éviter de coller un état "constrained" basé sur un heartbeat trop ancien
si le TTL est rallongé en config.
"""


def _redis_key(driver_id: int) -> str:
    return f"driver:{int(driver_id)}:device_health"


def _to_redis_value(value: Any) -> str:
    """Sérialise une valeur scalaire pour HSET (Redis hash = string only)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def write_device_health(
    redis_client: Any,
    driver_id: int,
    payload: dict[str, Any],
    *,
    now_ms: int | None = None,
    ttl_sec: int = DEVICE_HEALTH_TTL_SEC,
) -> bool:
    """Écrit le hash Redis device_health pour ``driver_id``.

    Retourne ``True`` si l'écriture a abouti, ``False`` sinon (Redis absent
    ou erreur réseau). N'élève jamais : le device-status doit dégrader sans
    casser le pipeline GPS principal.
    """
    if redis_client is None:
        return False

    heartbeat_ms = (
        int(now_ms) if isinstance(now_ms, (int, float)) else int(time.time() * 1000)
    )
    mapping: dict[str, str] = {
        "last_heartbeat_at": str(heartbeat_ms),
        "fgs_running": _to_redis_value(payload.get("fgs_running")),
        "battery_optimized": _to_redis_value(payload.get("battery_optimized")),
        "constraint_reason": _to_redis_value(payload.get("constraint_reason")),
        "fg_permission": _to_redis_value(payload.get("fg_permission")),
        "bg_permission": _to_redis_value(payload.get("bg_permission")),
        "gps_provider_enabled": _to_redis_value(payload.get("gps_provider_enabled")),
        "battery_level": _to_redis_value(payload.get("battery_level")),
        "fix_success_rate_last_5min": _to_redis_value(
            payload.get("fix_success_rate_last_5min")
        ),
    }

    key = _redis_key(driver_id)
    try:
        redis_client.hset(key, mapping=mapping)
        with suppress(Exception):
            redis_client.expire(key, int(ttl_sec))
        return True
    except (ConnectionError, OSError, TimeoutError) as exc:
        logger.warning(
            "[device_health] Redis HSET failed (network error: %s) driver_id=%s",
            type(exc).__name__,
            driver_id,
        )
        return False
    except Exception:
        logger.exception("[device_health] Redis HSET failed driver_id=%s", driver_id)
        return False


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:
            return None
    return str(value)


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None or raw == "":
        return None
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def parse_device_health(raw: dict[Any, Any] | None) -> dict[str, Any] | None:
    """Désérialise un hash Redis brut en dict typé.

    Retourne ``None`` si ``raw`` est vide/falsy. Les valeurs absentes ou
    illisibles sont mises à ``None`` (pas d'exception).
    """
    if not raw:
        return None
    decoded: dict[str, str | None] = {}
    for k, v in raw.items():
        key = _decode(k) or ""
        decoded[key] = _decode(v)

    return {
        "last_heartbeat_at": _parse_int(decoded.get("last_heartbeat_at")),
        "fgs_running": _parse_bool(decoded.get("fgs_running")),
        "battery_optimized": _parse_bool(decoded.get("battery_optimized")),
        "constraint_reason": (decoded.get("constraint_reason") or "") or None,
        "fg_permission": (decoded.get("fg_permission") or "") or None,
        "bg_permission": (decoded.get("bg_permission") or "") or None,
        "gps_provider_enabled": _parse_bool(decoded.get("gps_provider_enabled")),
        "battery_level": _parse_float(decoded.get("battery_level")),
        "fix_success_rate_last_5min": _parse_float(
            decoded.get("fix_success_rate_last_5min")
        ),
    }


def read_device_health(redis_client: Any, driver_id: int) -> dict[str, Any] | None:
    """Lit le hash device_health d'un seul driver (None si absent / erreur)."""
    if redis_client is None:
        return None
    key = _redis_key(driver_id)
    try:
        raw = redis_client.hgetall(key)
    except (ConnectionError, OSError, TimeoutError) as exc:
        logger.debug(
            "[device_health] Redis HGETALL failed (%s) driver_id=%s",
            type(exc).__name__,
            driver_id,
        )
        return None
    except Exception:
        logger.debug(
            "[device_health] Redis HGETALL failed driver_id=%s",
            driver_id,
            exc_info=True,
        )
        return None
    return parse_device_health(raw)


def read_device_health_batch(
    redis_client: Any, driver_ids: list[int] | tuple[int, ...]
) -> dict[int, dict[str, Any] | None]:
    """Lit les hash device_health de plusieurs drivers via une pipeline Redis.

    Optimisation pour ``build_company_driver_locations_items`` : évite N
    aller-retours réseau lorsqu'on construit la liste live pour le portail
    entreprise. Retourne toujours un dict ``driver_id -> health|None``.

    .. deprecated::
        Migration vers ``services.driver_device_health.read_driver_device_health_batch``.
        Instrumenté via ``device_health_legacy_read_total``.
    """
    try:
        from services.monitoring.driver_device_health_metrics import (
            record_device_health_legacy_read,
        )

        record_device_health_legacy_read(caller="read_device_health_batch")
    except Exception:
        pass
    out: dict[int, dict[str, Any] | None] = {int(d): None for d in driver_ids}
    if redis_client is None or not driver_ids:
        return out
    try:
        pipe = redis_client.pipeline()
        for did in driver_ids:
            pipe.hgetall(_redis_key(did))
        results = pipe.execute()
    except (ConnectionError, OSError, TimeoutError) as exc:
        logger.debug(
            "[device_health] Redis pipeline HGETALL failed (%s) n=%d",
            type(exc).__name__,
            len(driver_ids),
        )
        return out
    except Exception:
        logger.debug(
            "[device_health] Redis pipeline HGETALL failed n=%d",
            len(driver_ids),
            exc_info=True,
        )
        return out

    for did, raw in zip(driver_ids, results, strict=True):
        out[int(did)] = parse_device_health(raw)
    return out


__all__ = [
    "DEVICE_HEALTH_TTL_SEC",
    "parse_device_health",
    "read_device_health",
    "read_device_health_batch",
    "write_device_health",
]
