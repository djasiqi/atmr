from __future__ import annotations

from datetime import UTC, datetime

LocationMode = str
LocationStatus = str
PresenceStatus = str

MISSION_LIVE_THRESHOLDS = (20, 90, 300)  # live, recent, stale upper bounds
AVAILABILITY_THRESHOLDS = (90, 300, 900)


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


def presence_status_from_location_status(location_status: str) -> PresenceStatus:
    if location_status in {"live", "recent"}:
        return "online"
    if location_status == "stale":
        return "degraded"
    return "offline"
