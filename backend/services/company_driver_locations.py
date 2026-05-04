"""Liste positions chauffeurs (Redis + DB). Partagé par GET /me/drivers/locations et /me/drivers/live.

Modèle A (défaut produit) : vérité « officielle » = canon Redis (cf. ``driver:{id}:loc:canonical``).
Option D (cible structurelle future) : aligner la projection REST sur la même vérité affichable
que le fanout — évolution coordonnée, pas un changement local isolé.
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import selectinload

from ext import redis_client
from models import Booking, Driver
from models.base import _as_bool
from models.enums import BookingStatus
from services.company_driver_location_freshness import (
    last_seen_seconds_from_db_last_position_update,
    last_seen_seconds_from_location_fields,
)
from services.geolocation.presence import (
    compute_location_status,
    presence_status_from_location_status,
)

logger = logging.getLogger(__name__)

LAT_MIN, LAT_MAX = -90, 90
LON_MIN, LON_MAX = -180, 180

# Throttle log "no driver loc in Redis" : 1 fois / 10 min par company (aligné companies.py)
_NO_DRIVER_LOC_LOG_LAST: dict[int, float] = {}
_NO_DRIVER_LOC_LOG_INTERVAL_SEC = 600


def build_company_driver_locations_items(
    cid: int,
    *,
    is_demo_company: bool,
    bbox_ne_lat: float | None = None,
    bbox_ne_lng: float | None = None,
    bbox_sw_lat: float | None = None,
    bbox_sw_lng: float | None = None,
) -> list[dict[str, Any]]:
    """Construit la liste `locations` (même sémantique que l'ancien handler Flask)."""
    # selectinload(user) : évite N+1 sur driver.user (first_name, email demo) dans la boucle ci-dessous
    drivers = (
        Driver.query.options(selectinload(Driver.user)).filter_by(company_id=cid).all()
    )
    if not drivers:
        return []

    use_bbox = (
        bbox_ne_lat is not None
        and bbox_ne_lng is not None
        and bbox_sw_lat is not None
        and bbox_sw_lng is not None
        and LAT_MIN <= bbox_sw_lat <= LAT_MAX
        and LAT_MIN <= bbox_ne_lat <= LAT_MAX
        and LON_MIN <= bbox_sw_lng <= LON_MAX
        and LON_MIN <= bbox_ne_lng <= LON_MAX
    )

    redis_results: list[Any] = []
    redis_ok = False
    if redis_client:
        try:
            pipe = redis_client.pipeline()
            for driver in drivers:
                canonical_key = f"driver:{driver.id}:loc:canonical"
                legacy_key = f"driver:{driver.id}:loc"
                pipe.hgetall(canonical_key)
                pipe.hgetall(legacy_key)
            redis_results = pipe.execute()
            redis_ok = True
        except (ConnectionError, OSError, TimeoutError) as e:
            logger.debug(
                "[drivers/locations] Redis pipeline failed: %s",
                type(e).__name__,
            )
            redis_results = [None] * (len(drivers) * 2)
        except Exception:
            redis_results = [None] * (len(drivers) * 2)

    driver_ids = [d.id for d in drivers]
    active_bookings_map: dict[int, dict[str, Any]] = {}
    if driver_ids:
        active_statuses = (
            BookingStatus.ASSIGNED.value,
            BookingStatus.EN_ROUTE.value,
            BookingStatus.IN_PROGRESS.value,
        )
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        active_bookings = (
            Booking.query.filter(
                Booking.driver_id.in_(driver_ids),
                Booking.status.in_(active_statuses),
                or_(
                    Booking.scheduled_time.is_(None),
                    Booking.scheduled_time >= cutoff,
                ),
            )
            .with_entities(
                Booking.driver_id,
                Booking.id,
                Booking.status,
                Booking.pickup_location,
                Booking.dropoff_location,
            )
            .order_by(Booking.updated_at.desc())
            .all()
        )
        _MAX_LOCATION_STR_LEN = 30
        for b in active_bookings:
            driver_id_val = getattr(b, "driver_id", None)
            if driver_id_val and driver_id_val not in active_bookings_map:
                pickup = getattr(b, "pickup_location", None) or ""
                dropoff = getattr(b, "dropoff_location", None) or ""
                pickup_str = str(pickup)
                dropoff_str = str(dropoff)
                client_short = (
                    (pickup[:_MAX_LOCATION_STR_LEN] + "…")
                    if len(pickup_str) > _MAX_LOCATION_STR_LEN
                    else pickup
                )
                raw_status = getattr(b, "status", None)
                mission_status_val = getattr(raw_status, "value", None) or str(
                    raw_status or ""
                )
                active_bookings_map[driver_id_val] = {
                    "current_booking_id": getattr(b, "id", None),
                    "client_short": client_short
                    or (
                        dropoff[:_MAX_LOCATION_STR_LEN] + "…"
                        if len(dropoff_str) > _MAX_LOCATION_STR_LEN
                        else dropoff
                    ),
                    "mission_status": mission_status_val,
                }

    now = datetime.now(UTC)
    locations: list[dict[str, Any]] = []
    for i, driver in enumerate(drivers):
        lat, lon, ts = None, None, None
        used_db_fallback = False
        loc_data: dict[str, Any] = {}
        canonical_idx = i * 2
        legacy_idx = canonical_idx + 1
        h = redis_results[canonical_idx] if canonical_idx < len(redis_results) else None
        if not h:
            h = redis_results[legacy_idx] if legacy_idx < len(redis_results) else None
        if h:
            for k, v in h.items():
                try:
                    loc_data[k.decode()] = v.decode()
                except Exception:
                    loc_data[k.decode()] = v
            for kf in ("lat", "lon", "speed", "heading", "accuracy"):
                if kf in loc_data:
                    with suppress(Exception):
                        loc_data[kf] = float(loc_data[kf])
            lat = loc_data.get("lat")
            lon = loc_data.get("lon")
            ts = loc_data.get("ts")

        if lat is None and getattr(driver, "latitude", None) is not None:
            lat = float(driver.latitude)
            lon = float(driver.longitude)
            ts = None
            used_db_fallback = True

        if lat is None or lon is None:
            continue

        last_seen_seconds = None
        is_stale = True
        if loc_data:
            last_seen_seconds = last_seen_seconds_from_location_fields(
                loc_data, now=now
            )
        elif used_db_fallback:
            driver_email = str(
                getattr(getattr(driver, "user", None), "email", "") or ""
            ).lower()
            if is_demo_company or driver_email.endswith("@demo.local"):
                last_seen_seconds = 0
            else:
                last_seen_seconds = last_seen_seconds_from_db_last_position_update(
                    getattr(driver, "last_position_update", None),
                    now=now,
                )

        is_active = _as_bool(getattr(driver, "is_active", True))
        active_booking = active_bookings_map.get(driver.id, {})
        has_active_booking = bool(active_booking.get("current_booking_id"))
        mission_status = active_booking.get("mission_status")
        location_mode_raw = loc_data.get("location_mode", "availability_presence")
        expected_mode = (
            "mission_live"
            if mission_status
            in (
                BookingStatus.ASSIGNED.value,
                BookingStatus.EN_ROUTE.value,
                BookingStatus.IN_PROGRESS.value,
            )
            else "availability_presence"
        )
        location_mode = location_mode_raw or expected_mode
        location_status = compute_location_status(
            mode=location_mode, last_seen_seconds=last_seen_seconds
        )
        if last_seen_seconds is not None and h and loc_data:
            with suppress(Exception):
                from services.monitoring.driver_location_metrics import (
                    observe_canonical_staleness_seconds,
                )

                observe_canonical_staleness_seconds(
                    location_mode=str(location_mode),
                    last_seen_seconds=float(last_seen_seconds),
                )
        presence_status = presence_status_from_location_status(location_status)
        is_stale = location_status in {"stale", "offline"}

        if not is_active:
            status = "offline"
        elif mission_status in (
            BookingStatus.EN_ROUTE.value,
            BookingStatus.IN_PROGRESS.value,
        ):
            status = "busy"
        elif mission_status == BookingStatus.ASSIGNED.value:
            status = "assigned"
        else:
            status = "available"

        if not is_active:
            presence_status = "offline"
            location_status = "offline"
        offline_reason = (
            "no_signal"
            if location_status == "offline"
            else ("location_stale" if presence_status == "degraded" else "")
        )
        is_available = status == "available"

        first_name = getattr(getattr(driver, "user", None), "first_name", None)
        accuracy_m: float | None = None
        if loc_data.get("accuracy") is not None:
            with suppress(Exception):
                accuracy_m = float(loc_data["accuracy"])

        loc_item: dict[str, Any] = {
            "driver_id": driver.id,
            "id": driver.id,
            "latitude": float(lat),
            "longitude": float(lon),
            "lat": float(lat),
            "lon": float(lon),
            "first_name": first_name,
            "timestamp": ts,
            "recorded_at": loc_data.get("recorded_at") or ts,
            "received_at": loc_data.get("received_at"),
            "last_seen_seconds": last_seen_seconds,
            "is_stale": is_stale,
            "status": status,
            "mission_status": mission_status,
            "presence_status": presence_status,
            "location_status": location_status,
            "location_mode": location_mode,
            "is_available": is_available,
            "offline_reason": offline_reason,
        }
        if accuracy_m is not None:
            loc_item["accuracy_m"] = accuracy_m
        if has_active_booking:
            loc_item["current_booking_id"] = active_booking.get("current_booking_id")
            loc_item["client_short"] = active_booking.get("client_short", "")
            if status == "busy":
                logger.debug(
                    "[drivers/locations] Chauffeur en course: driver_id=%s booking_id=%s mission_status=%s client=%s",
                    driver.id,
                    active_booking.get("current_booking_id"),
                    mission_status,
                    str(active_booking.get("client_short", ""))[:50],
                )
        locations.append(loc_item)

    status_counts = {"available": 0, "assigned": 0, "busy": 0, "offline": 0}
    for loc in locations:
        s = loc.get("status")
        if s in status_counts:
            status_counts[s] += 1
    try:
        from services.monitoring.websocket_metrics import ws_metrics

        ws_metrics.set_drivers_status_total(
            available=status_counts["available"],
            assigned=status_counts["assigned"],
            busy=status_counts["busy"],
            offline=status_counts["offline"],
        )
    except Exception:
        pass

    if use_bbox:
        assert bbox_sw_lat is not None
        assert bbox_ne_lat is not None
        assert bbox_sw_lng is not None
        assert bbox_ne_lng is not None
        lat_min = min(bbox_sw_lat, bbox_ne_lat)
        lat_max = max(bbox_sw_lat, bbox_ne_lat)
        lng_min = min(bbox_sw_lng, bbox_ne_lng)
        lng_max = max(bbox_sw_lng, bbox_ne_lng)
        locations = [
            loc
            for loc in locations
            if lat_min <= loc["latitude"] <= lat_max
            and lng_min <= loc["longitude"] <= lng_max
        ]

    if redis_ok and cid and len(drivers) > 0 and len(locations) == 0:
        now_ts = time.monotonic()
        last = _NO_DRIVER_LOC_LOG_LAST.get(cid, 0)
        if now_ts - last >= _NO_DRIVER_LOC_LOG_INTERVAL_SEC:
            _NO_DRIVER_LOC_LOG_LAST[cid] = now_ts
            logger.warning(
                "No driver loc in Redis for company_id=%s (drivers_count=%d)",
                cid,
                len(drivers),
            )

    return locations


def merge_drivers_with_locations(
    drivers: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aligné sur le merge client dans companyService.fetchCompanyDriversCanonical."""
    by_id: dict[int, dict[str, Any]] = {}
    for loc in locations:
        raw_id = loc.get("driver_id") or loc.get("id")
        if raw_id is None:
            continue
        try:
            did = int(raw_id)
        except Exception:
            continue
        by_id[did] = loc

    out: list[dict[str, Any]] = []
    for drv in drivers:
        raw = drv.get("id")
        try:
            did = int(raw) if raw is not None else None
        except Exception:
            did = None
        if did is None:
            out.append(drv)
            continue
        loc = by_id.get(did)
        if not loc:
            out.append(drv)
            continue

        merged = {
            **drv,
            "latitude": loc.get("lat") or loc.get("latitude") or drv.get("latitude"),
            "longitude": loc.get("lon") or loc.get("longitude") or drv.get("longitude"),
            "last_seen_seconds": loc.get("last_seen_seconds")
            if loc.get("last_seen_seconds") is not None
            else drv.get("last_seen_seconds"),
            "location_status": loc.get("location_status")
            if loc.get("location_status") is not None
            else drv.get("location_status"),
            "presence_status": loc.get("presence_status")
            if loc.get("presence_status") is not None
            else drv.get("presence_status"),
            "location_mode": loc.get("location_mode")
            if loc.get("location_mode") is not None
            else drv.get("location_mode"),
            "recorded_at": loc.get("recorded_at")
            if loc.get("recorded_at") is not None
            else drv.get("recorded_at"),
            "received_at": loc.get("received_at")
            if loc.get("received_at") is not None
            else drv.get("received_at"),
            "mission_status": loc.get("mission_status")
            if loc.get("mission_status") is not None
            else drv.get("mission_status"),
            "status": loc.get("status")
            if loc.get("status") is not None
            else drv.get("status"),
            "current_booking_id": loc.get("current_booking_id")
            if loc.get("current_booking_id") is not None
            else drv.get("current_booking_id"),
            "client_short": loc.get("client_short")
            if loc.get("client_short") is not None
            else (drv.get("client_short") or ""),
        }
        if loc.get("accuracy_m") is not None:
            merged["accuracy_m"] = loc["accuracy_m"]
        out.append(merged)
    return out
