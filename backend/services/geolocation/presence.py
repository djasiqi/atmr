from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

LocationMode = str
LocationStatus = str
PresenceStatus = str

MISSION_LIVE_THRESHOLDS = (20, 90, 300)  # live, recent, stale upper bounds
AVAILABILITY_THRESHOLDS = (90, 300, 900)

# Fenêtre « dernière position connue » (fallback DB sans clé Redis active)
LAST_KNOWN_DB_MAX_SECONDS: dict[str, int] = {
    "mission_live": 4 * 3600,
    "availability_presence": 24 * 3600,
    "passive_last_known": 24 * 3600,
}

DEVICE_HEALTH_FRESH_SEC = 120
"""Fenêtre de fraîcheur du heartbeat device-status pour qu'il puisse
overrider la présence (>= 120 s : on retombe sur le calcul standard, comme
si aucun health n'était disponible)."""


# Fraîcheur REST (last_seen) : ``resolve_location_freshness_timestamp`` dans
# ``services.company_driver_location_freshness`` (priorité recorded_at > received_at > ts).


def normalize_location_mode(mode: str | None) -> LocationMode:
    if mode in {"mission_live", "availability_presence", "passive_last_known"}:
        return mode
    return "mission_live"


def parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def compute_last_seen_seconds(
    recorded_at: str | None, now: datetime | None = None
) -> int | None:
    dt = parse_iso_utc(recorded_at)
    if dt is None:
        return None
    ref = now or datetime.now(UTC)
    try:
        return max(0, int((ref - dt).total_seconds()))
    except Exception:
        return None


def compute_location_status(
    *,
    mode: str | None,
    last_seen_seconds: int | None,
) -> LocationStatus:
    if last_seen_seconds is None:
        return "offline"
    normalized = normalize_location_mode(mode)
    live_cutoff, recent_cutoff, stale_cutoff = (
        MISSION_LIVE_THRESHOLDS
        if normalized == "mission_live"
        else AVAILABILITY_THRESHOLDS
    )
    if last_seen_seconds <= live_cutoff:
        return "live"
    if last_seen_seconds <= recent_cutoff:
        return "recent"
    if last_seen_seconds <= stale_cutoff:
        return "stale"
    return "offline"


def compute_db_fallback_location_status(
    *,
    mode: str | None,
    last_seen_seconds: int | None,
) -> LocationStatus:
    """Statut carte quand seule la DB driver fournit les coordonnées (Redis expiré)."""
    if last_seen_seconds is None:
        return "offline"
    normalized = normalize_location_mode(mode)
    max_last_known = LAST_KNOWN_DB_MAX_SECONDS.get(
        normalized, LAST_KNOWN_DB_MAX_SECONDS["availability_presence"]
    )
    if last_seen_seconds <= max_last_known:
        return "last_known"
    return "offline"


def presence_status_from_location_status(location_status: str) -> PresenceStatus:
    if location_status in {"live", "recent"}:
        return "online"
    if location_status == "last_known":
        return "degraded"
    if location_status == "stale":
        return "degraded"
    return "offline"


def _is_device_health_fresh(
    device_health: dict[str, Any] | None,
    *,
    now_ms: int | None = None,
    fresh_sec: int = DEVICE_HEALTH_FRESH_SEC,
) -> bool:
    """Vrai si le heartbeat device-status est arrivé il y a moins de ``fresh_sec``."""
    if not device_health:
        return False
    raw = device_health.get("last_heartbeat_at")
    try:
        last_ms = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return False
    if last_ms is None or last_ms <= 0:
        return False
    ref_ms = (
        int(now_ms)
        if isinstance(now_ms, (int, float))
        else int(datetime.now(UTC).timestamp() * 1000)
    )
    age_sec = max(0, (ref_ms - last_ms) / 1000.0)
    return age_sec < float(fresh_sec)


def _has_device_constraint(device_health: dict[str, Any]) -> bool:
    """Vrai si le device_health signale une contrainte OEM / permission."""
    if device_health.get("battery_optimized") is True:
        return True
    reason = device_health.get("constraint_reason")
    return bool(isinstance(reason, str) and reason.strip())


def apply_device_health_override(
    presence_status: PresenceStatus,
    location_status: LocationStatus,
    device_health: dict[str, Any] | None,
    *,
    now_ms: int | None = None,
    fresh_sec: int = DEVICE_HEALTH_FRESH_SEC,
) -> tuple[PresenceStatus, LocationStatus]:
    """Override "app vivante mais GPS bloqué" → ``degraded_constrained``.

    Règle :

    * si ``presence_status`` est ``offline`` ou ``degraded``,
    * ET le ``device_health`` est récent (heartbeat < ``fresh_sec``),
    * ET le device signale une contrainte (``battery_optimized=True`` ou
      ``constraint_reason`` non vide),

    on remplace le couple ``(presence_status, location_status)`` par
    ``("degraded_constrained", "degraded_constrained")``. Sinon on
    retourne le couple inchangé (passe-through strict — c'est ce qui
    garantit "si device_health absent, comportement identique à avant").
    """
    if presence_status not in {"offline", "degraded"}:
        return presence_status, location_status
    if not device_health:
        return presence_status, location_status
    if not _is_device_health_fresh(device_health, now_ms=now_ms, fresh_sec=fresh_sec):
        return presence_status, location_status
    if not _has_device_constraint(device_health):
        return presence_status, location_status
    return "degraded_constrained", "degraded_constrained"
