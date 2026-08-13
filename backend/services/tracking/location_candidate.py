"""P5-B — candidat GPS, preuve durable PostgreSQL, promotion canonical.

Ordre obligatoire quand ``TRACKING_PG_FIRST_CANONICAL_ENABLED=true`` :

    evaluate → persist_with_outbox → COMMIT PG → DurableLocationProof → promote

Jamais Redis canonical / GEO avant commit PostgreSQL.
Flag ``false`` (défaut) : comportement historique inchangé.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_CANONICAL_TTL_SEC = int(os.getenv("DRIVER_LOC_TTL_SEC", "1200"))


def is_pg_first_canonical_enabled() -> bool:
    return os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class LocationCandidate:
    """Point GPS candidat avant persistance durable / promotion canonique."""

    driver_id: int
    latitude: float
    longitude: float
    recorded_at: datetime | None = None
    mission_id: int | None = None
    location_mode: str = "mission_live"
    accuracy: float | None = None
    transport: str = "http"
    raw_lat: float | None = None
    raw_lon: float | None = None
    meta: dict[str, Any] | None = None
    capture_id: str | None = None
    location_event_id: str | None = None
    tracking_session_id: str | None = None
    session_generation: int | None = None
    sequence_id: int | None = None
    company_id: int | None = None
    speed: float | None = None
    heading: float | None = None
    source: str = "gps"


@dataclass(frozen=True, slots=True)
class DurableLocationProof:
    """Preuve qu'un point est durablement en PostgreSQL. Constructible seulement
    après commit réussi (``pg_committed=True``)."""

    driver_id: int
    company_id: int
    capture_id: str | None
    location_event_id: str
    tracking_session_id: str | None
    session_generation: int | None
    sequence_id: int | None
    mission_id: int | None
    recorded_at: datetime | None
    latitude: float
    longitude: float
    accept_status: str
    admission_reason: str
    live_eligible: bool
    canonical_eligible: bool
    pg_committed: bool
    location_mode: str = "mission_live"
    speed: float | None = None
    heading: float | None = None
    accuracy: float | None = None
    source: str = "gps"
    received_at: datetime | None = None
    sent_at: datetime | None = None
    is_background: bool = False
    transport: str = "http"

    def __post_init__(self) -> None:
        if not self.pg_committed:
            raise ValueError(
                "DurableLocationProof exige pg_committed=True "
                "(aucune promotion sans commit PostgreSQL)"
            )


def build_durable_location_proof(
    *,
    pg_committed: bool,
    driver_id: int,
    company_id: int,
    capture_id: str | None,
    location_event_id: str,
    tracking_session_id: str | None,
    session_generation: int | None,
    sequence_id: int | None,
    mission_id: int | None,
    recorded_at: datetime | None,
    latitude: float,
    longitude: float,
    accept_status: str,
    admission_reason: str = "",
    live_eligible: bool = True,
    canonical_eligible: bool = True,
    **kwargs: Any,
) -> DurableLocationProof:
    if not pg_committed:
        raise ValueError("preuve durable refusée: PostgreSQL non commité")
    return DurableLocationProof(
        driver_id=driver_id,
        company_id=company_id,
        capture_id=capture_id,
        location_event_id=location_event_id,
        tracking_session_id=tracking_session_id,
        session_generation=session_generation,
        sequence_id=sequence_id,
        mission_id=mission_id,
        recorded_at=recorded_at,
        latitude=latitude,
        longitude=longitude,
        accept_status=accept_status,
        admission_reason=admission_reason,
        live_eligible=live_eligible,
        canonical_eligible=canonical_eligible,
        pg_committed=True,
        **kwargs,
    )


def evaluate_location_candidate(
    candidate: LocationCandidate,
    *,
    context: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Évalue un candidat (admissibilité). Ne touche pas Redis canonical."""
    if candidate.latitude is None or candidate.longitude is None:
        return {
            "ok": False,
            "disposition": "reject",
            "reason": "missing_coordinates",
            "candidate_driver_id": candidate.driver_id,
        }
    return {
        "ok": True,
        "disposition": "persist",
        "reason": "candidate_admitted",
        "candidate_driver_id": candidate.driver_id,
        "capture_id": candidate.capture_id,
    }


def _decode_hash(raw: Any) -> dict[str, str]:
    data: dict[str, str] = {}
    if not raw:
        return data
    try:
        for key, val in raw.items():
            kk = key.decode() if isinstance(key, bytes) else str(key)
            vv = val.decode() if isinstance(val, bytes) else str(val)
            data[kk] = vv
    except Exception:
        return {}
    return data


def _existing_gen_seq(existing: dict[str, str]) -> tuple[int, int] | None:
    gen = _coerce_int(existing.get("session_generation"))
    seq = _coerce_int(existing.get("sequence_id"))
    if gen is None or seq is None:
        return None
    return gen, seq


def _format_iso(value: datetime | None) -> str:
    if value is None:
        return ""
    dt = value if value.tzinfo else value.replace(tzinfo=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def promote_location_candidate(
    proof: DurableLocationProof,
    *,
    redis_client: Any | None = None,
    evaluation: dict[str, Any] | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Promeut Redis canonical + GEO uniquement avec une preuve PG.

    Ordre monotone ``(session_generation, sequence_id)`` : un replay plus
    ancien ne recule pas la carte.
    """
    if not isinstance(proof, DurableLocationProof) or not proof.pg_committed:
        return {
            "ok": False,
            "promoted": False,
            "reason": "missing_durable_proof",
        }
    if not proof.canonical_eligible or proof.accept_status != "accepted_canonical":
        return {
            "ok": True,
            "promoted": False,
            "reason": "not_canonical_eligible",
            "capture_id": proof.capture_id,
        }

    client = redis_client
    if client is None:
        try:
            from ext import redis_client as ext_redis

            client = ext_redis
        except Exception:
            client = None
    if client is None:
        logger.warning(
            "[p5b_promote] redis indisponible driver_id=%s capture_id=%s",
            proof.driver_id,
            proof.capture_id,
        )
        return {
            "ok": False,
            "promoted": False,
            "reason": "redis_unavailable",
            "capture_id": proof.capture_id,
        }

    canonical_key = f"driver:{proof.driver_id}:loc:canonical"
    legacy_key = f"driver:{proof.driver_id}:loc"
    existing = _decode_hash(client.hgetall(canonical_key) or {})
    if not existing:
        existing = _decode_hash(client.hgetall(legacy_key) or {})

    incoming_gen = proof.session_generation
    incoming_seq = proof.sequence_id
    current = _existing_gen_seq(existing)
    if (
        incoming_gen is not None
        and incoming_seq is not None
        and current is not None
        and (incoming_gen, incoming_seq) <= current
    ):
        logger.info(
            "[p5b_promote] skip stale gen/seq driver_id=%s capture_id=%s "
            "incoming=(%s,%s) current=%s",
            proof.driver_id,
            proof.capture_id,
            incoming_gen,
            incoming_seq,
            current,
        )
        return {
            "ok": True,
            "promoted": False,
            "reason": "stale_generation_sequence",
            "capture_id": proof.capture_id,
        }

    received = proof.received_at or datetime.now(UTC)
    recorded_iso = _format_iso(proof.recorded_at)
    received_iso = _format_iso(received)
    sent_iso = _format_iso(proof.sent_at) if proof.sent_at else received_iso
    mapping = {
        "company_id": str(proof.company_id),
        "lat": str(proof.latitude),
        "lon": str(proof.longitude),
        "speed": str(proof.speed) if proof.speed is not None else "",
        "heading": str(proof.heading) if proof.heading is not None else "",
        "accuracy": str(proof.accuracy) if proof.accuracy is not None else "",
        "ts": recorded_iso or received_iso,
        "recorded_at": recorded_iso,
        "sent_at": sent_iso,
        "received_at": received_iso,
        "location_mode": proof.location_mode,
        "is_background": "1" if proof.is_background else "0",
        "mission_id": str(proof.mission_id) if proof.mission_id is not None else "",
        "degraded_context": "0",
        "source": proof.source,
        "location_event_id": proof.location_event_id or "",
        "capture_id": proof.capture_id or "",
        "tracking_session_id": proof.tracking_session_id or "",
        "session_generation": (
            str(proof.session_generation)
            if proof.session_generation is not None
            else ""
        ),
        "sequence_id": str(proof.sequence_id) if proof.sequence_id is not None else "",
    }
    client.hset(canonical_key, mapping=mapping)
    client.expire(canonical_key, _CANONICAL_TTL_SEC)
    client.hset(legacy_key, mapping=mapping)
    client.expire(legacy_key, _CANONICAL_TTL_SEC)

    geo_updated = False
    if proof.company_id:
        try:
            geo_key = f"driver_locations:geo:{proof.company_id}"
            client.geoadd(
                geo_key, [proof.longitude, proof.latitude, str(proof.driver_id)]
            )
            client.expire(geo_key, _CANONICAL_TTL_SEC)
            geo_updated = True
        except Exception as geo_err:
            logger.debug("[p5b_promote] GEOADD failed: %s", geo_err)

    try:
        from services.monitoring.driver_location_metrics import (
            inc_canonical_redis_write,
        )

        inc_canonical_redis_write(
            location_mode=proof.location_mode, transport=proof.transport
        )
    except Exception:
        logger.debug("[p5b_promote] metric skipped", exc_info=True)

    logger.info(
        "[p5b_promote] canonical driver_id=%s capture_id=%s event_id=%s gen=%s seq=%s",
        proof.driver_id,
        proof.capture_id,
        proof.location_event_id,
        proof.session_generation,
        proof.sequence_id,
    )
    return {
        "ok": True,
        "promoted": True,
        "reason": "promoted_after_pg",
        "capture_id": proof.capture_id,
        "geo_updated": geo_updated,
    }
