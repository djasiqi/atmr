"""Ingestion GPS interne (ws-service → backend).

F-01 : auth fail-closed, validation atomique, rate-limit.
F-02 : ACK uniquement après commit PostgreSQL (sync_db) ; Kafka hors frontière.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import Blueprint, jsonify, request

from ext import db, limiter, redis_client
from services.security.internal_service_auth import (
    authorize_internal_request,
    ingest_enabled,
    rate_limit_principal,
)
from services.tracking.event_payload_hash import PayloadHashError
from services.tracking.ingest_durability import (
    PayloadConflictError,
    attempt_redis_canonical_repair,
    mark_repair_done_if_current,
    persist_tracking_batch,
    prepare_tracking_batch,
    require_durability_mode,
)
from services.tracking.ingest_idempotency import (
    mark_done,
    release_pending,
    try_reserve,
)
from services.tracking.location_event_id import (
    normalize_recorded_at_utc_canonical,
    resolve_location_event_id,
    validate_raw_location_event_id,
)

internal_tracking_bp = Blueprint("internal_tracking", __name__)
logger = logging.getLogger(__name__)

_MAX_POINTS_PER_BATCH = 50
_LAT_MIN, _LAT_MAX = -90.0, 90.0
_LON_MIN, _LON_MAX = -180.0, 180.0
_ALLOWED_LOCATION_MODES = frozenset(
    {"mission_live", "background", "foreground", "unknown"}
)
_SKEW_FUTURE = timedelta(minutes=5)
_SKEW_PAST = timedelta(hours=24)
_HEADER_EVENT_ID = "X-Location-Event-ID"

_DEFAULT_RATE_LIMIT = os.getenv(
    "INTERNAL_TRACKING_INGEST_RATE_LIMIT", "6000 per minute"
)


def _rate_limit_key() -> str:
    return rate_limit_principal()


def _safe_finite_float(value: Any, *, lo: float, hi: float) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    if parsed < lo or parsed > hi:
        return None
    return parsed


def _resolve_driver_tenant(driver_id: int) -> tuple[int | None, str | None]:
    try:
        from models.company import Company
        from models.driver import Driver

        driver = Driver.query.filter_by(id=driver_id).first()
    except Exception as exc:
        logger.warning(
            "[internal_tracking] lookup driver_id=%s failed: %s",
            driver_id,
            type(exc).__name__,
        )
        return None, "driver_lookup_failed"

    if driver is None:
        return None, "driver_not_found"
    if not bool(getattr(driver, "is_active", False)):
        return None, "driver_inactive"
    company_id = getattr(driver, "company_id", None)
    if not isinstance(company_id, int):
        return None, "driver_without_tenant"

    try:
        company = Company.query.filter_by(id=company_id).first()
    except Exception as exc:
        logger.warning(
            "[internal_tracking] lookup company_id=%s failed: %s",
            company_id,
            type(exc).__name__,
        )
        return None, "company_lookup_failed"

    if company is None:
        return None, "company_not_found"
    if not bool(getattr(company, "is_approved", False)):
        return None, "company_not_approved"
    return company_id, None


def _validate_recorded_at(raw: Any) -> tuple[str | None, str | None]:
    canon = normalize_recorded_at_utc_canonical(raw)
    if canon is None:
        return None, "invalid_recorded_at"
    try:
        dt = datetime.fromisoformat(canon.replace("Z", "+00:00"))
    except ValueError:
        return None, "invalid_recorded_at"
    now = datetime.now(UTC)
    if dt > now + _SKEW_FUTURE:
        return None, "recorded_at_too_future"
    if dt < now - _SKEW_PAST:
        return None, "recorded_at_too_old"
    return canon, None


def _normalize_point(
    raw: Any,
    *,
    driver_id: int,
    header_event_id: str | None,
    batch_size: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "invalid_point"

    lat = _safe_finite_float(raw.get("latitude"), lo=_LAT_MIN, hi=_LAT_MAX)
    lon = _safe_finite_float(raw.get("longitude"), lo=_LON_MIN, hi=_LON_MAX)
    if lat is None or lon is None:
        return None, "invalid_coordinates"

    mode = raw.get("location_mode") or "mission_live"
    if not isinstance(mode, str) or mode not in _ALLOWED_LOCATION_MODES:
        return None, "invalid_location_mode"

    recorded_at, ts_err = _validate_recorded_at(
        raw.get("recorded_at") or raw.get("timestamp")
    )
    if ts_err or recorded_at is None:
        return None, ts_err or "invalid_recorded_at"

    body_raw = raw.get("location_event_id")
    if body_raw is None:
        body_raw = raw.get("tracking_event_id")
    body_id, body_err = validate_raw_location_event_id(body_raw)
    if body_err:
        return None, body_err

    header_id, header_err = validate_raw_location_event_id(header_event_id)
    if header_err:
        return None, header_err

    if batch_size > 1:
        raw_id = body_id
    else:
        if header_id and body_id and header_id != body_id:
            return None, "location_event_id_conflict"
        raw_id = header_id or body_id

    event_id = resolve_location_event_id(
        driver_id=driver_id,
        latitude=lat,
        longitude=lon,
        recorded_at=recorded_at,
        raw_id=raw_id,
    )

    accuracy = raw.get("accuracy")
    heading = raw.get("heading")
    speed = raw.get("speed")
    mission_id = raw.get("mission_id")
    sequence_id = raw.get("sequence_id")

    payload: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "accuracy": _safe_finite_float(accuracy, lo=0.0, hi=50_000.0)
        if accuracy is not None
        else None,
        "heading": _safe_finite_float(heading, lo=0.0, hi=360.0)
        if heading is not None
        else None,
        "speed": _safe_finite_float(speed, lo=0.0, hi=150.0)
        if speed is not None
        else None,
        "recorded_at": recorded_at,
        "mission_id": mission_id if isinstance(mission_id, (int, str)) else None,
        "location_mode": mode,
        "location_event_id": event_id,
    }
    if isinstance(sequence_id, int) and not isinstance(sequence_id, bool):
        payload["sequence_id"] = sequence_id

    for opt_key, opt_raw in (
        ("accuracy", accuracy),
        ("heading", heading),
        ("speed", speed),
    ):
        if opt_raw is not None and payload[opt_key] is None:
            return None, f"invalid_{opt_key}"

    return payload, None


@internal_tracking_bp.route("/api/internal/tracking/ingest", methods=["POST"])
@limiter.limit(_DEFAULT_RATE_LIMIT, key_func=_rate_limit_key)
def tracking_ingest():
    try:
        require_durability_mode()
    except RuntimeError as exc:
        logger.error("[internal_tracking] %s", exc)
        return jsonify({"error": "durable_ingest_unavailable"}), 503

    if not ingest_enabled():
        return jsonify({"error": "ingest_disabled"}), 503

    ok, auth_error = authorize_internal_request(request.headers)
    if not ok:
        status = 503 if auth_error == "missing_token" else 401
        return jsonify({"error": auth_error or "unauthorized"}), status

    # Rate-limit partagé : Redis préféré ; si KO on continue (PG autorité F-02)
    redis_ok = False
    if redis_client is not None:
        try:
            redis_client.ping()
            redis_ok = True
        except Exception:
            redis_ok = False
    if not redis_ok:
        logger.warning(
            "[internal_tracking] redis unavailable — continue sync_db sans accélérateur"
        )

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid_payload"}), 400

    driver_id = data.get("driver_id")
    points = data.get("points")
    if not isinstance(driver_id, int) or isinstance(driver_id, bool) or driver_id <= 0:
        return jsonify({"error": "invalid_driver_id"}), 400
    if not isinstance(points, list):
        return jsonify({"error": "invalid_payload"}), 400
    if len(points) == 0:
        return jsonify({"error": "empty_batch"}), 400
    if len(points) > _MAX_POINTS_PER_BATCH:
        return jsonify(
            {
                "error": "batch_too_large",
                "max_points": _MAX_POINTS_PER_BATCH,
            }
        ), 400

    header_event_raw = request.headers.get(_HEADER_EVENT_ID)
    if (
        len(points) > 1
        and header_event_raw is not None
        and str(header_event_raw).strip()
    ):
        return jsonify({"error": "header_event_id_not_allowed_for_batch"}), 400

    tenant_company_id, tenant_error = _resolve_driver_tenant(driver_id)
    if tenant_error or tenant_company_id is None:
        return jsonify({"error": tenant_error or "driver_without_tenant"}), 403

    claimed_company = data.get("company_id")
    if (
        isinstance(claimed_company, int)
        and not isinstance(claimed_company, bool)
        and claimed_company != tenant_company_id
    ):
        return jsonify({"error": "company_mismatch"}), 403

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in points:
        payload, err = _normalize_point(
            raw,
            driver_id=driver_id,
            header_event_id=header_event_raw if len(points) == 1 else None,
            batch_size=len(points),
        )
        if err or payload is None:
            return jsonify({"error": err or "invalid_point"}), 400
        eid = payload["location_event_id"]
        if eid in seen_ids:
            return jsonify({"error": "duplicate_location_event_id_in_batch"}), 400
        seen_ids.add(eid)
        normalized.append(payload)

    client_batch_id = data.get("batch_id")
    try:
        prepared = prepare_tracking_batch(
            driver_id=driver_id,
            company_id=tenant_company_id,
            points=normalized,
            source="internal_http",
            client_batch_id=str(client_batch_id) if client_batch_id else None,
        )
    except ValueError as exc:
        code = str(exc)
        if code == "batch_id_mismatch":
            return jsonify({"error": "batch_id_mismatch"}), 400
        return jsonify({"error": code}), 400
    except PayloadHashError as exc:
        return jsonify({"error": exc.code}), 400

    # Réservations Redis (accélérateur) — jamais bloquantes
    reserved: list[tuple[str, str]] = []
    for pt in prepared.points:
        eid = str(pt.payload["location_event_id"])
        outcome, nonce = try_reserve(driver_id=driver_id, location_event_id=eid)
        if outcome == "reserved" and nonce:
            reserved.append((eid, nonce))

    try:
        with db.session.begin():
            result = persist_tracking_batch(prepared=prepared, session=db.session)
    except PayloadConflictError as exc:
        for eid, nonce in reserved:
            release_pending(driver_id=driver_id, location_event_id=eid, nonce=nonce)
        return jsonify(
            {
                "ok": False,
                "batch_id": prepared.batch_id,
                "error_code": exc.code,
                "conflicting_event_ids": exc.conflicting_event_ids,
                "durability": "none",
            }
        ), 409
    except ValueError as exc:
        for eid, nonce in reserved:
            release_pending(driver_id=driver_id, location_event_id=eid, nonce=nonce)
        return jsonify({"error": str(exc)}), 403
    except Exception:
        for eid, nonce in reserved:
            release_pending(driver_id=driver_id, location_event_id=eid, nonce=nonce)
        logger.exception("[internal_tracking] persist failed")
        return jsonify({"error": "ingest_persistence_failed"}), 503

    # Post-commit best-effort
    nonce_by_eid = dict(reserved)
    for eid in list(result.event_ids_persisted) + list(result.event_ids_duplicate):
        mark_done(
            driver_id=driver_id,
            location_event_id=eid,
            nonce=nonce_by_eid.get(eid),
        )

    # Tentative Redis canonical pour le point le plus récent persisté
    try:
        latest = None
        for pt in prepared.points:
            eid = str(pt.payload["location_event_id"])
            if (
                eid in result.event_ids_persisted or eid in result.event_ids_duplicate
            ) and (latest is None or pt.recorded_at > latest.recorded_at):
                latest = pt
        if latest is not None:
            eid = str(latest.payload["location_event_id"])
            ok_redis = attempt_redis_canonical_repair(
                driver_id=driver_id,
                company_id=tenant_company_id,
                latitude=latest.latitude,
                longitude=latest.longitude,
                recorded_at=latest.recorded_at,
                location_event_id=eid,
                location_mode=str(
                    latest.payload.get("location_mode") or "mission_live"
                ),
            )
            if ok_redis:
                mark_repair_done_if_current(
                    driver_id=driver_id,
                    location_event_id=eid,
                    target_recorded_at=latest.recorded_at,
                )
    except Exception:
        logger.warning(
            "[internal_tracking] post-commit repair attempt failed", exc_info=True
        )

    logger.info(
        "[internal_tracking] ingest driver_id=%s company_id=%s "
        "persisted=%s duplicates=%s received=%s batch_id=%s audience=%s",
        driver_id,
        tenant_company_id,
        result.persisted,
        result.duplicates,
        result.received,
        result.batch_id,
        rate_limit_principal(),
    )
    return jsonify(
        {
            "ok": True,
            "trace_id": result.trace_id,
            "batch_id": result.batch_id,
            "durability": "postgres_committed",
            "received": result.received,
            "persisted": result.persisted,
            "duplicates": result.duplicates,
            "driver_id": driver_id,
            "company_id": tenant_company_id,
        }
    ), 200
