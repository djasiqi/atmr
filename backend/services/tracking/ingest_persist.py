"""Persistance métier des messages Kafka ``driver.location.raw``."""

from __future__ import annotations

import logging
from typing import Any

from application.drivers.update_driver_location import (
    UpdateDriverLocationCommand,
    UpdateDriverLocationResult,
    UpdateDriverLocationUseCase,
)
from drivers.infrastructure.adapters.location_adapter import create_location_update_fn
from services.tracking.location_event_id import resolve_location_event_id

logger = logging.getLogger(__name__)


def _payload_coords(payload: dict[str, Any]) -> tuple[float, float]:
    lat_val = payload.get("latitude", payload.get("lat"))
    lon_val = payload.get("longitude", payload.get("lon"))
    if lat_val is None or lon_val is None:
        raise ValueError("invalid_payload_coords")
    return float(lat_val), float(lon_val)


def persist_driver_location_from_kafka(
    message_obj: dict[str, Any],
    *,
    driver_id: int,
) -> tuple[dict[str, Any], UpdateDriverLocationResult]:
    """Exécute le use-case de localisation et enrichit le message pour ``processed``."""
    from celery_app import get_flask_app

    payload = message_obj.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid_payload")

    lat, lon = _payload_coords(payload)
    recorded_at = str(
        payload.get("recorded_at") or payload.get("timestamp") or payload.get("ts") or ""
    )
    if not recorded_at:
        from datetime import UTC, datetime

        recorded_at = datetime.now(UTC).isoformat()

    top_level_event_id = message_obj.get("location_event_id")
    payload_event_id = payload.get("location_event_id") or payload.get("tracking_event_id")
    raw_event_id = (
        str(top_level_event_id).strip()
        if isinstance(top_level_event_id, str) and top_level_event_id.strip()
        else (
            str(payload_event_id).strip()
            if payload_event_id is not None and str(payload_event_id).strip()
            else None
        )
    )
    location_event_id = resolve_location_event_id(
        driver_id=driver_id,
        latitude=lat,
        longitude=lon,
        recorded_at=recorded_at,
        raw_id=raw_event_id,
    )

    company_id_obj = message_obj.get("company_id")
    company_id: int | None = (
        int(company_id_obj) if isinstance(company_id_obj, int) else None
    )
    if company_id is None:
        payload_company = payload.get("company_id")
        if isinstance(payload_company, int):
            company_id = payload_company
        elif isinstance(payload_company, str) and payload_company.isdigit():
            company_id = int(payload_company)

    speed_raw = payload.get("speed_mps", payload.get("speed"))
    heading_raw = payload.get("heading")
    accuracy_raw = payload.get("accuracy_m", payload.get("accuracy"))
    speed = float(speed_raw) if speed_raw is not None else None
    heading = float(heading_raw) if heading_raw is not None else None
    accuracy = float(accuracy_raw) if accuracy_raw is not None else None
    location_mode = str(payload.get("location_mode") or "mission_live")
    sent_at = payload.get("sent_at")
    mission_id_raw = payload.get("mission_id")
    mission_id: int | None = None
    if isinstance(mission_id_raw, int):
        mission_id = mission_id_raw
    elif isinstance(mission_id_raw, str) and mission_id_raw.isdigit():
        mission_id = int(mission_id_raw)

    app = get_flask_app()
    with app.app_context():
        from services.geolocation.location import get_location_service
        from services.monitoring.location_correlation_log import (
            log_driver_location_processed,
        )

        loc_svc = get_location_service()
        norm_mode = loc_svc.resolve_normalized_location_mode(
            company_id, location_mode
        )
        uc = UpdateDriverLocationUseCase(update_location_fn=create_location_update_fn())
        uc_result = uc.execute(
            UpdateDriverLocationCommand(
                driver_id=driver_id,
                latitude=lat,
                longitude=lon,
                speed=speed if speed is not None and speed > 0 else None,
                heading=heading if heading is not None and heading >= 0 else None,
                accuracy=accuracy if accuracy is not None and accuracy > 0 else None,
                ts=recorded_at,
                recorded_at=recorded_at,
                sent_at=str(sent_at) if sent_at else None,
                location_mode=location_mode,
                is_background=bool(payload.get("is_background", False)),
                mission_id=mission_id,
                metrics_transport="kafka",
                location_event_id=location_event_id,
                emit_geofence=True,
                company_id=company_id,
            )
        )
        if not uc_result.dedup_skipped:
            log_driver_location_processed(
                driver_id=driver_id,
                company_id=company_id,
                transport="kafka",
                location_mode=norm_mode,
                accept_status=uc_result.accept_status,
                accept_reason=uc_result.accept_reason,
                location_event_id=location_event_id,
            )
            if uc_result.accept_status == "accepted_canonical" and recorded_at is not None:
                try:
                    from datetime import UTC, datetime

                    from services.monitoring.driver_location_metrics import (
                        observe_tracking_position_freshness_seconds,
                    )

                    rec_dt = datetime.fromisoformat(
                        str(recorded_at).replace("Z", "+00:00")
                    )
                    if rec_dt.tzinfo is None:
                        rec_dt = rec_dt.replace(tzinfo=UTC)
                    age = (datetime.now(UTC) - rec_dt).total_seconds()
                    observe_tracking_position_freshness_seconds(
                        freshness_seconds=age,
                        company_id=company_id,
                        location_mode=norm_mode,
                    )
                except Exception:
                    pass

    enriched_payload = {
        **payload,
        "location_event_id": location_event_id,
    }
    if company_id is not None:
        enriched_payload.setdefault("company_id", company_id)

    enriched = {
        **message_obj,
        "payload": enriched_payload,
        "company_id": company_id,
        "location_event_id": location_event_id,
        "persist_result": {
            "accept_status": uc_result.accept_status,
            "accept_reason": uc_result.accept_reason,
            "snapped_lat": uc_result.snapped_lat,
            "snapped_lon": uc_result.snapped_lon,
            "dedup_skipped": uc_result.dedup_skipped,
        },
        "pipeline_stages": [
            {"stage": "ACK_INGESTED", "ok": True},
            {"stage": "ACK_PROCESSED", "ok": uc_result.accept_status.startswith("accepted")},
        ],
    }
    return enriched, uc_result
