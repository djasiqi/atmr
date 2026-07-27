"""Pont Kafka → TX PostgreSQL + outbox (Phase 1 / Annexe A.1).

Remplace le publish ``processed`` dans le consumer RAW.
"""

from __future__ import annotations

import logging
from typing import Any

from services.tracking.location_event_id import resolve_location_event_id
from services.tracking.persist_with_outbox import (
    PersistConflictError,
    persist_location_event_with_outbox,
)
from services.tracking.session_registry import (
    SessionRegistryError,
    resolve_authoritative_session,
)

logger = logging.getLogger(__name__)


class PersistKafkaOutboxError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


def _payload_coords(payload: dict[str, Any]) -> tuple[float, float]:
    lat_val = payload.get("latitude", payload.get("lat"))
    lon_val = payload.get("longitude", payload.get("lon"))
    if lat_val is None or lon_val is None:
        raise ValueError("invalid_payload_coords")
    return float(lat_val), float(lon_val)


def persist_driver_location_with_outbox_from_kafka(
    message_obj: dict[str, Any],
    *,
    driver_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persiste dans une TX (ledger+events+state+driver+outbox) puis commit PG.

    Ne publie pas Kafka ``processed`` — rôle de ``outbox_publisher``.
    """
    from celery_app import get_flask_app

    payload = message_obj.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid_payload")

    lat, lon = _payload_coords(payload)
    recorded_at = str(
        payload.get("recorded_at")
        or payload.get("timestamp")
        or payload.get("ts")
        or ""
    )
    if not recorded_at:
        from datetime import UTC, datetime

        recorded_at = datetime.now(UTC).isoformat()

    top_level_event_id = message_obj.get("location_event_id")
    payload_event_id = payload.get("location_event_id") or payload.get(
        "tracking_event_id"
    )
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
    if company_id is None:
        raise PersistKafkaOutboxError("company_id_missing")

    tracking_session_id = str(
        payload.get("tracking_session_id")
        or message_obj.get("tracking_session_id")
        or ""
    ).strip()
    if not tracking_session_id:
        raise PersistKafkaOutboxError("tracking_session_id_missing")

    sequence_raw = payload.get("sequence_id", message_obj.get("sequence_id"))
    if sequence_raw is None:
        raise PersistKafkaOutboxError("sequence_id_missing")
    sequence_id = int(sequence_raw)

    claimed_gen_raw = payload.get(
        "session_generation", message_obj.get("session_generation")
    )
    claimed_generation = int(claimed_gen_raw) if claimed_gen_raw is not None else None

    speed_raw = payload.get("speed_mps", payload.get("speed"))
    heading_raw = payload.get("heading")
    accuracy_raw = payload.get("accuracy_m", payload.get("accuracy"))
    speed = float(speed_raw) if speed_raw is not None else None
    heading = float(heading_raw) if heading_raw is not None else None
    accuracy = float(accuracy_raw) if accuracy_raw is not None else None
    location_mode = str(payload.get("location_mode") or "mission_live")
    mission_id_raw = payload.get("mission_id")
    mission_id: int | None = None
    if isinstance(mission_id_raw, int):
        mission_id = mission_id_raw
    elif isinstance(mission_id_raw, str) and mission_id_raw.isdigit():
        mission_id = int(mission_id_raw)

    source = str(message_obj.get("source") or payload.get("source") or "kafka")

    app = get_flask_app()
    with app.app_context():
        from sqlalchemy.exc import SQLAlchemyError

        from ext import db

        try:
            try:
                auth = resolve_authoritative_session(
                    db.session,
                    driver_id=driver_id,
                    company_id=company_id,
                    tracking_session_id=tracking_session_id,
                    claimed_generation=claimed_generation,
                    sequence_id=sequence_id,
                )
            except SessionRegistryError as exc:
                raise PersistKafkaOutboxError(exc.code, exc.message) from exc

            session_generation = int(auth["session_generation"])
            # Annexe A.3 : superseded → persisté + watermark, jamais Redis/fanout
            publish_realtime = str(auth.get("status") or "") != "superseded"

            try:
                result = persist_location_event_with_outbox(
                    db.session,
                    driver_id=driver_id,
                    company_id=company_id,
                    location_event_id=location_event_id,
                    tracking_session_id=tracking_session_id,
                    session_generation=session_generation,
                    sequence_id=sequence_id,
                    latitude=lat,
                    longitude=lon,
                    recorded_at=recorded_at,
                    source=source,
                    location_mode=location_mode,
                    accuracy_m=accuracy,
                    speed_mps=speed,
                    heading=heading,
                    mission_id=mission_id,
                    publish_realtime=publish_realtime,
                )
                db.session.commit()
            except PersistConflictError as exc:
                db.session.rollback()
                raise PersistKafkaOutboxError(exc.code, str(exc)) from exc
            except SQLAlchemyError:
                db.session.rollback()
                raise
        finally:
            db.session.remove()

    enriched_payload = {
        **payload,
        "location_event_id": location_event_id,
        "session_generation": session_generation,
        "tracking_session_id": tracking_session_id,
        "sequence_id": sequence_id,
        "company_id": company_id,
    }
    enriched = {
        **message_obj,
        "payload": enriched_payload,
        "company_id": company_id,
        "location_event_id": location_event_id,
        "session_generation": session_generation,
        "tracking_session_id": tracking_session_id,
        "sequence_id": sequence_id,
        "persist_result": result,
        "pipeline_stages": [
            {"stage": "ACK_INGESTED", "ok": True},
            {"stage": "ACK_POSTGRES_OUTBOX", "ok": result.get("status") != "error"},
        ],
    }
    return enriched, result
