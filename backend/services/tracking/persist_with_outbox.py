"""Persistance TX atomique + outbox (Phase 1 / Annexe A.1).

Ordre : ledger → driver_location_events → session_state → driver → outbox
Puis commit PG ; le consumer RAW commit l'offset ensuite.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class PersistConflictError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code


def _payload_hash(payload: dict[str, Any]) -> str:
    """Alias v1 : hash legacy sans ``capture_id`` (P0-D)."""
    from services.tracking.location_idempotency import legacy_payload_hash

    return legacy_payload_hash(payload)


def _parse_recorded_at(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, str) and raw:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return datetime.now(UTC)


def persist_location_event_with_outbox(
    session: Session,
    *,
    driver_id: int,
    company_id: int,
    location_event_id: str,
    tracking_session_id: str,
    session_generation: int,
    sequence_id: int,
    latitude: float,
    longitude: float,
    recorded_at: Any,
    source: str,
    location_mode: str = "mission_live",
    accuracy_m: float | None = None,
    speed_mps: float | None = None,
    heading: float | None = None,
    mission_id: int | None = None,
    capture_id: str | None = None,
    schema_version: str = "tracking-event-payload-v1",
    extra_payload: dict[str, Any] | None = None,
    publish_realtime: bool = True,
) -> dict[str, Any]:
    """Persiste un point et écrit l'outbox dans la même TX. Ne commit pas.

    ``publish_realtime=False`` : session superseded — persistée mais pas Redis/fanout.
    """
    recorded = _parse_recorded_at(recorded_at)
    from services.tracking.capture_id import resolve_effective_capture_id
    from services.tracking.processed_envelope import build_persisted_location_envelope

    merged_extra = dict(extra_payload or {})
    if capture_id:
        merged_extra["capture_id"] = capture_id
    effective_capture = resolve_effective_capture_id(
        merged_extra,
        location_event_id=location_event_id,
    )
    # Payload métier pour hash v1 : sans capture_id (parité prod / anti faux conflict).
    hash_payload = {
        "driver_id": driver_id,
        "company_id": company_id,
        "location_event_id": location_event_id,
        "tracking_session_id": tracking_session_id,
        "session_generation": session_generation,
        "sequence_id": sequence_id,
        "latitude": latitude,
        "longitude": longitude,
        "recorded_at": recorded.isoformat(),
        "location_mode": location_mode,
        "source": source,
        "accuracy_m": accuracy_m,
        "speed_mps": speed_mps,
        "heading": heading,
        "mission_id": mission_id,
        "schema_version": schema_version,
    }
    # Envelope / outbox peuvent encore porter capture_id — hors hash.
    payload = {
        **hash_payload,
        "capture_id": effective_capture,
        **{k: v for k, v in merged_extra.items() if k != "capture_id"},
    }
    payload["capture_id"] = effective_capture
    payload["location_event_id"] = location_event_id
    phash = _payload_hash(hash_payload)

    # Lock watermark session
    session.execute(
        text(
            """
            SELECT id FROM tracking_session_state
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            FOR UPDATE
            """
        ),
        {"driver_id": driver_id, "sid": tracking_session_id},
    )

    # ON CONFLICT sans cible : couvre uq_tracking_ingest_driver_event ET
    # uq_tracking_ingest_session_sequence (sinon UniqueViolation → fail-stop poison).
    inserted = session.execute(
        text(
            """
            INSERT INTO tracking_ingest_events (
                driver_id, company_id, location_event_id, capture_id,
                event_payload_hash, payload_schema_version, source, recorded_at,
                tracking_session_id, sequence_id, session_generation
            ) VALUES (
                :driver_id, :company_id, :eid, :capture_id,
                :phash, :schema, :source, :recorded_at,
                :sid, :seq, :gen
            )
            ON CONFLICT DO NOTHING
            RETURNING location_event_id
            """
        ),
        {
            "driver_id": driver_id,
            "company_id": company_id,
            "eid": location_event_id,
            "capture_id": effective_capture,
            "phash": phash,
            "schema": schema_version,
            "source": source,
            "recorded_at": recorded,
            "sid": tracking_session_id,
            "seq": sequence_id,
            "gen": session_generation,
        },
    ).first()

    if inserted is None:
        existing = (
            session.execute(
                text(
                    """
                SELECT
                    i.event_payload_hash,
                    i.driver_id,
                    i.location_event_id,
                    i.tracking_session_id,
                    i.sequence_id,
                    i.session_generation,
                    i.recorded_at,
                    l.raw_latitude,
                    l.raw_longitude,
                    l.accuracy_m,
                    l.speed_mps,
                    l.heading
                FROM tracking_ingest_events i
                LEFT JOIN driver_location_events l
                  ON l.driver_id = i.driver_id
                 AND l.location_event_id = i.location_event_id
                WHERE i.driver_id = :driver_id AND i.location_event_id = :eid
                """
                ),
                {"driver_id": driver_id, "eid": location_event_id},
            )
            .mappings()
            .first()
        )
        if existing is not None:
            from services.tracking.location_idempotency import (
                DuplicateDecision,
                compare_persisted_event,
            )

            decision = compare_persisted_event(
                existing_row=existing,
                incoming_payload=hash_payload,
                incoming_hash=phash,
            )
            if decision == DuplicateDecision.DUPLICATE_EXACT_HASH:
                return {
                    "status": "duplicate",
                    "reason": "same_event_already_persisted",
                    "location_event_id": location_event_id,
                    "duplicate_decision": decision.value,
                }
            if decision == DuplicateDecision.DUPLICATE_LEGACY_EQUIVALENT:
                return {
                    "status": "duplicate",
                    "reason": "legacy_business_equivalent",
                    "location_event_id": location_event_id,
                    "duplicate_decision": decision.value,
                }
            raise PersistConflictError("event_id_payload_conflict")

        # Même (driver, session, sequence) déjà pris par un autre location_event_id
        # (rejeu Kafka / recyclage compteur Redis http-legacy).
        seq_owner = (
            session.execute(
                text(
                    """
                SELECT location_event_id FROM tracking_ingest_events
                WHERE driver_id = :driver_id
                  AND tracking_session_id = :sid
                  AND sequence_id = :seq
                """
                ),
                {
                    "driver_id": driver_id,
                    "sid": tracking_session_id,
                    "seq": sequence_id,
                },
            )
            .mappings()
            .first()
        )
        if seq_owner is not None:
            logger.warning(
                "[persist_outbox] sequence déjà persistée driver_id=%s sid=%s "
                "seq=%s existing_eid=%s new_eid=%s — conflit déterministe",
                driver_id,
                tracking_session_id,
                sequence_id,
                seq_owner["location_event_id"],
                location_event_id,
            )
            return {
                "status": "duplicate",
                "reason": "session_sequence_already_persisted",
                "location_event_id": location_event_id,
                "existing_location_event_id": str(seq_owner["location_event_id"]),
            }

        logger.warning(
            "[persist_outbox] INSERT DO NOTHING sans ligne lisible "
            "driver_id=%s eid=%s sid=%s seq=%s — duplicate_unproven",
            driver_id,
            location_event_id,
            tracking_session_id,
            sequence_id,
        )
        return {
            "status": "duplicate",
            "reason": "duplicate_unproven",
            "location_event_id": location_event_id,
        }

    session.execute(
        text(
            """
            INSERT INTO driver_location_events (
                driver_id, company_id, location_event_id, capture_id,
                tracking_session_id,
                session_generation, sequence_id, recorded_at,
                raw_latitude, raw_longitude, accuracy_m, speed_mps, heading,
                location_mode, mission_id, source, event_payload_hash,
                payload_schema_version
            ) VALUES (
                :driver_id, :company_id, :eid, :capture_id, :sid,
                :gen, :seq, :recorded_at,
                :lat, :lon, :acc, :spd, :hdg,
                :mode, :mission_id, :source, :phash, :schema
            )
            """
        ),
        {
            "driver_id": driver_id,
            "company_id": company_id,
            "eid": location_event_id,
            "capture_id": effective_capture,
            "sid": tracking_session_id,
            "gen": session_generation,
            "seq": sequence_id,
            "recorded_at": recorded,
            "lat": latitude,
            "lon": longitude,
            "acc": accuracy_m,
            "spd": speed_mps,
            "hdg": heading,
            "mode": location_mode,
            "mission_id": mission_id,
            "source": source,
            "phash": phash,
            "schema": schema_version,
        },
    )

    # Watermark contigu
    state = (
        session.execute(
            text(
                """
            SELECT contiguous_persisted_through, max_seen_sequence
            FROM tracking_session_state
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            """
            ),
            {"driver_id": driver_id, "sid": tracking_session_id},
        )
        .mappings()
        .first()
    )

    contiguous = int(state["contiguous_persisted_through"]) if state else 0
    max_seen = int(state["max_seen_sequence"]) if state else 0
    max_seen = max(max_seen, sequence_id)
    if sequence_id == contiguous + 1:
        contiguous = sequence_id
        # Avancer tant que les suivants existent
        while True:
            nxt = session.execute(
                text(
                    """
                    SELECT 1 FROM driver_location_events
                    WHERE driver_id = :driver_id AND tracking_session_id = :sid
                      AND sequence_id = :seq
                    LIMIT 1
                    """
                ),
                {
                    "driver_id": driver_id,
                    "sid": tracking_session_id,
                    "seq": contiguous + 1,
                },
            ).first()
            if nxt is None:
                break
            contiguous += 1
            session.execute(
                text(
                    """
                    UPDATE tracking_sequence_gaps
                    SET resolved_at = NOW()
                    WHERE driver_id = :driver_id AND tracking_session_id = :sid
                      AND sequence_from <= :seq AND sequence_to >= :seq
                      AND resolved_at IS NULL
                    """
                ),
                {
                    "driver_id": driver_id,
                    "sid": tracking_session_id,
                    "seq": contiguous,
                },
            )
    elif sequence_id > contiguous + 1:
        session.execute(
            text(
                """
                INSERT INTO tracking_sequence_gaps (
                    driver_id, tracking_session_id, sequence_from, sequence_to
                ) VALUES (
                    :driver_id, :sid, :from_seq, :to_seq
                )
                """
            ),
            {
                "driver_id": driver_id,
                "sid": tracking_session_id,
                "from_seq": contiguous + 1,
                "to_seq": sequence_id - 1,
            },
        )

    now = datetime.now(UTC)
    session.execute(
        text(
            """
            INSERT INTO tracking_session_state (
                driver_id, company_id, tracking_session_id, session_generation,
                contiguous_persisted_through, max_seen_sequence,
                first_seen_at, last_seen_at
            ) VALUES (
                :driver_id, :company_id, :sid, :gen,
                :contiguous, :max_seen, :now, :now
            )
            ON CONFLICT (driver_id, tracking_session_id) DO UPDATE SET
                contiguous_persisted_through = EXCLUDED.contiguous_persisted_through,
                max_seen_sequence = GREATEST(
                    tracking_session_state.max_seen_sequence, EXCLUDED.max_seen_sequence
                ),
                last_seen_at = EXCLUDED.last_seen_at
            """
        ),
        {
            "driver_id": driver_id,
            "company_id": company_id,
            "sid": tracking_session_id,
            "gen": session_generation,
            "contiguous": contiguous,
            "max_seen": max_seen,
            "now": now,
        },
    )

    # Projection driver — ordre (generation, sequence), pas recorded_at
    session.execute(
        text(
            """
            UPDATE driver
            SET latitude = :lat,
                longitude = :lon,
                last_position_update = :recorded_at,
                last_location_event_id = :eid,
                last_tracking_session_generation = :gen,
                last_tracking_sequence_id = :seq
            WHERE id = :driver_id
              AND (
                last_tracking_session_generation IS NULL
                OR (last_tracking_session_generation, last_tracking_sequence_id)
                   < (:gen, :seq)
              )
            """
        ),
        {
            "lat": latitude,
            "lon": longitude,
            "recorded_at": recorded,
            "eid": location_event_id,
            "gen": session_generation,
            "seq": sequence_id,
            "driver_id": driver_id,
        },
    )

    if publish_realtime:
        envelope = build_persisted_location_envelope(
            driver_id=driver_id,
            company_id=company_id,
            capture_id=effective_capture,
            location_event_id=location_event_id,
            tracking_session_id=tracking_session_id,
            session_generation=session_generation,
            sequence_id=sequence_id,
            latitude=latitude,
            longitude=longitude,
            recorded_at=recorded.isoformat(),
            mission_id=mission_id,
            location_mode=location_mode,
            source=source,
            accuracy_m=accuracy_m,
            speed_mps=speed_mps,
            heading=heading,
            extra_payload=extra_payload,
        )
        session.execute(
            text(
                """
                INSERT INTO tracking_event_outbox (
                    event_id, event_type, driver_id, location_event_id,
                    session_generation, sequence_id, payload
                ) VALUES (
                    :event_id, 'persisted', :driver_id, :eid,
                    :gen, :seq, CAST(:payload AS jsonb)
                )
                ON CONFLICT (event_id) DO NOTHING
                """
            ),
            {
                "event_id": location_event_id,
                "driver_id": driver_id,
                "eid": location_event_id,
                "gen": session_generation,
                "seq": sequence_id,
                "payload": json.dumps(envelope, default=str),
            },
        )

    return {
        "status": "persisted",
        "reason": "inserted",
        "location_event_id": location_event_id,
        "capture_id": effective_capture,
        "contiguous_persisted_through": contiguous,
        "publish_realtime": publish_realtime,
    }
