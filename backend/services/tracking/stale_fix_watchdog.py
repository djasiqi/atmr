"""Watchdog serveur — kick Socket.IO drivers en mission au tracking gelé.

Trois pathologies couvertes :
  - ``fix_stale`` : le device track mais le dernier fix est trop ancien.
  - ``mobile_tracking_down`` : le device en mission **live** (EN_ROUTE /
    IN_PROGRESS) rapporte ``tracking_active=0`` / ``fgs_running=0`` (ou
    ``constraint_reason=fgs_not_running``). Cas observé après login/logout ou
    FGS tué par l'OS : l'app sait que son tracking est éteint.
  - ``no_fresh_position`` (correctif structurant) : pour un driver dont une
    mission **exige** le tracking (EN_ROUTE/IN_PROGRESS, ou ASSIGNED confirmé
    dans la fenêtre T‑30), **aucune position fraîche réelle** n'atteint le
    pipeline (clé canonical Redis ``driver:{id}:loc:canonical`` absente ou
    périmée) — **indépendamment** de ce que le device rapporte. C'est le seul
    signal non‑falsifiable côté serveur : un FGS zombie peut rapporter
    ``tracking_active=1`` / un fix natif « frais » tout en n'envoyant rien.
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

STALE_FIX_AGE_SEC = int(os.getenv("STALE_FIX_WATCHDOG_AGE_SEC", "300"))
KICK_TTL_SEC = int(os.getenv("STALE_FIX_WATCHDOG_KICK_TTL_SEC", "600"))
# Au-delà de ce délai sans heartbeat device-health, on considère le device
# offline : un kick Socket.IO ne serait pas délivré, inutile de l'émettre.
HEARTBEAT_FRESH_SEC = int(os.getenv("STALE_FIX_WATCHDOG_HEARTBEAT_FRESH_SEC", "900"))
ENABLED = os.getenv("STALE_FIX_WATCHDOG_ENABLED", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
EMIT_FORCE_RESTART = os.getenv("EMIT_FORCE_TRACKING_RESTART", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Élargissement N+1 : kicker aussi les drivers en mission live dont le device
# rapporte explicitement un tracking éteint (tracking_active=0 / fgs_running=0).
KICK_ON_TRACKING_DOWN = os.getenv(
    "STALE_FIX_WATCHDOG_KICK_ON_TRACKING_DOWN", "true"
).lower() not in ("0", "false", "no", "off")

# --- Correctif A : détection par fraîcheur réelle (non-falsifiable) ----------
# Flag dédié pour rollout sûr en prod (peut être coupé sans toucher au reste).
FRESHNESS_ENABLED = os.getenv(
    "STALE_FIX_WATCHDOG_FRESHNESS_ENABLED", "true"
).lower() not in ("0", "false", "no", "off")
# Au-delà de ce délai sans position canonical acceptée alors qu'une mission
# exige le tracking, on kicke (le mobile envoie ~toutes les 8 s : 180 s laisse
# une marge confortable pour le démarrage et les aléas réseau).
FRESHNESS_MAX_SEC = int(os.getenv("STALE_FIX_WATCHDOG_FRESHNESS_MAX_SEC", "180"))
# Fenêtre ASSIGNED : on attend du tracking dès T‑30 avant l'heure prévue et
# jusqu'à T+60 après (course confirmée non encore démarrée mais imminente/à quai).
ASSIGNED_LEAD_BEFORE_SEC = int(
    os.getenv("STALE_FIX_WATCHDOG_ASSIGNED_LEAD_SEC", "1800")
)
ASSIGNED_GRACE_AFTER_SEC = int(
    os.getenv("STALE_FIX_WATCHDOG_ASSIGNED_GRACE_SEC", "3600")
)


def _kick_throttle_key(driver_id: int) -> str:
    return f"kick_sent:driver:{driver_id}"


def _is_false(value) -> bool:
    """True si le snapshot indique explicitement False (et non None/absent)."""
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("0", "false", "no", "off")
    return False


def _heartbeat_is_recent(health: dict, now: datetime) -> bool:
    """Le device a-t-il émis un heartbeat assez récent pour recevoir un kick ?"""
    last_hb = health.get("last_heartbeat_at")
    if not last_hb:
        # Pas d'info heartbeat (ex. snapshot legacy/test) : on ne bloque pas.
        return True
    try:
        hb_age = now.timestamp() - (float(last_hb) / 1000.0)
    except (TypeError, ValueError):
        return True
    return hb_age <= HEARTBEAT_FRESH_SEC


def _coerce_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value) -> datetime | None:
    """Parse un datetime (objet ou ISO string) en tz-aware UTC. None si invalide."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _assigned_in_tracking_window(scheduled_at, time_confirmed, now: datetime) -> bool:
    """ASSIGNED confirmé dont l'heure prévue est dans la fenêtre [T‑30, T+60]."""
    if not time_confirmed:
        return False
    dt = _parse_dt(scheduled_at)
    if dt is None:
        return False
    return (
        (dt.timestamp() - ASSIGNED_LEAD_BEFORE_SEC)
        <= now.timestamp()
        <= (dt.timestamp() + ASSIGNED_GRACE_AFTER_SEC)
    )


def _resolve_kick_reason(
    health: dict, *, live_mission: bool, now: datetime
) -> str | None:
    """Raison du kick selon le self-report device (ou None)."""
    constraint = str(health.get("constraint_reason") or "").lower()

    # Pathologie historique : fix_stale prolongé.
    # NB : on se base sur `last_fix_age_seconds` (âge réel du dernier fix GPS)
    # et non sur `recorded_at` (heure du heartbeat) — ce dernier n'indique pas
    # l'ancienneté du fix et est absent du snapshot Redis parsé. On exige par
    # ailleurs un heartbeat récent pour que le kick Socket.IO soit délivrable.
    if constraint == "fix_stale":
        if not _heartbeat_is_recent(health, now):
            return None
        fix_age = _coerce_int(health.get("last_fix_age_seconds"))
        if fix_age is not None and fix_age < STALE_FIX_AGE_SEC:
            return None
        return "fix_stale"

    if not KICK_ON_TRACKING_DOWN:
        return None

    # Pathologie cold/zombie : réservée aux missions live (EN_ROUTE/IN_PROGRESS)
    # pour éviter de kicker un driver ASSIGNED qui track légitimement en présence.
    if not live_mission:
        return None

    tracking_down = (
        _is_false(health.get("tracking_active"))
        or _is_false(health.get("fgs_running"))
        or constraint == "fgs_not_running"
    )
    if not tracking_down:
        return None
    if not _heartbeat_is_recent(health, now):
        return None
    return "mobile_tracking_down"


def _canonical_freshness_sec(
    redis_client, driver_id: int, now: datetime
) -> float | None:
    """Âge (s) de la dernière position **canonical acceptée**, ou None si absente.

    Lit ``driver:{id}:loc:canonical`` (rempli uniquement sur ``accepted_canonical``).
    Priorité ``received_at`` (horodatage backend, immune au skew device) puis
    ``recorded_at`` / ``ts``.
    """
    try:
        raw = redis_client.hgetall(f"driver:{driver_id}:loc:canonical")
    except Exception:
        return None
    if not raw:
        return None
    data: dict[str, str] = {}
    try:
        for k, v in raw.items():
            kk = k.decode() if isinstance(k, bytes) else str(k)
            vv = v.decode() if isinstance(v, bytes) else str(v)
            data[kk] = vv
    except Exception:
        return None
    for field in ("received_at", "recorded_at", "ts"):
        dt = _parse_dt(data.get(field))
        if dt is not None:
            return max(0.0, (now - dt).total_seconds())
    return None


def _resolve_freshness_kick(
    redis_client, health: dict, driver_id: int, now: datetime
) -> str | None:
    """Kick si aucune position canonical fraîche n'atteint le pipeline.

    Indépendant du self-report device : c'est le seul signal serveur que le
    FGS zombie ne peut pas falsifier. On exige un heartbeat récent (device
    joignable) pour que le kick soit délivrable.
    """
    if not _heartbeat_is_recent(health, now):
        return None
    age = _canonical_freshness_sec(redis_client, driver_id, now)
    if age is not None and age < FRESHNESS_MAX_SEC:
        return None
    return "no_fresh_position"


def run_stale_fix_watchdog_tick() -> dict:
    """Émet force_tracking_restart aux drivers en mission au tracking gelé."""
    if not ENABLED:
        return {"ok": True, "skipped": "disabled"}

    from ext import redis_client
    from models import Booking
    from models.enums import BookingStatus
    from services.driver_device_health import read_driver_device_health_snapshot
    from services.realtime.socketio import emit_force_tracking_restart

    if not redis_client:
        return {"ok": False, "error": "redis_unavailable"}

    now = datetime.now(UTC)
    active_statuses = (
        BookingStatus.ASSIGNED.value,
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    live_statuses = (
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    assigned_status = BookingStatus.ASSIGNED.value
    rows = (
        Booking.query.filter(Booking.status.in_(active_statuses))
        .with_entities(
            Booking.driver_id,
            Booking.id,
            Booking.status,
            Booking.scheduled_time,
            Booking.time_confirmed,
        )
        .all()
    )
    driver_ids: set[int] = set()
    live_driver_ids: set[int] = set()
    # Drivers dont une mission EXIGE une position fraîche (live, ou ASSIGNED T‑30).
    freshness_required: set[int] = set()
    for row in rows:
        did = getattr(row, "driver_id", None)
        if not did:
            continue
        did = int(did)
        driver_ids.add(did)
        # Booking.status est un SAEnum(BookingStatus(str, Enum)) : on normalise
        # vers la valeur string pour comparer de façon robuste (enum membre OU
        # string brute selon le contexte d'appel/test).
        status = getattr(row, "status", None)
        status_value = getattr(status, "value", status)
        if status_value in live_statuses:
            live_driver_ids.add(did)
            freshness_required.add(did)
        elif status_value == assigned_status and _assigned_in_tracking_window(
            getattr(row, "scheduled_time", None),
            getattr(row, "time_confirmed", None),
            now,
        ):
            freshness_required.add(did)

    sent = 0
    skipped = 0
    throttled = 0

    try:
        from services.monitoring.driver_location_metrics import (
            inc_stale_fix_watchdog_kick,
        )
    except Exception:
        inc_stale_fix_watchdog_kick = None  # type: ignore[assignment]

    for driver_id in driver_ids:
        if redis_client.get(_kick_throttle_key(driver_id)):
            throttled += 1
            continue

        health = read_driver_device_health_snapshot(driver_id) or {}
        reason = _resolve_kick_reason(
            health,
            live_mission=driver_id in live_driver_ids,
            now=now,
        )
        # Correctif A : si le self-report ne déclenche rien mais qu'une mission
        # exige le tracking, on vérifie la fraîcheur RÉELLE du pipeline.
        if reason is None and FRESHNESS_ENABLED and driver_id in freshness_required:
            reason = _resolve_freshness_kick(redis_client, health, driver_id, now)

        if reason is None:
            skipped += 1
            continue

        if EMIT_FORCE_RESTART:
            emit_force_tracking_restart(
                driver_id,
                reason=f"server_watchdog_{reason}",
            )
            with contextlib.suppress(Exception):
                redis_client.setex(_kick_throttle_key(driver_id), KICK_TTL_SEC, "1")
            if inc_stale_fix_watchdog_kick is not None:
                with contextlib.suppress(Exception):
                    inc_stale_fix_watchdog_kick(reason=reason)
            sent += 1

    logger.info(
        "[stale_fix_watchdog] sent=%s skipped=%s throttled=%s candidates=%s "
        "live=%s freshness_required=%s",
        sent,
        skipped,
        throttled,
        len(driver_ids),
        len(live_driver_ids),
        len(freshness_required),
    )
    return {
        "ok": True,
        "sent": sent,
        "skipped": skipped,
        "throttled": throttled,
        "candidates": len(driver_ids),
        "live_candidates": len(live_driver_ids),
        "freshness_required": len(freshness_required),
    }
