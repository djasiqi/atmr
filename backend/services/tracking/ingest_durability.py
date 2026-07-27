"""Persistance batch GPS F-02 — transaction atomique, ACK = commit PostgreSQL."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.tracking.event_payload_hash import (
    PAYLOAD_SCHEMA_VERSION,
    PayloadHashError,
    compute_batch_id,
    compute_event_payload_hash_from_point,
)

logger = logging.getLogger(__name__)


class PayloadConflictError(Exception):
    def __init__(self, code: str, conflicting_event_ids: list[str]) -> None:
        super().__init__(code)
        self.code = code
        self.conflicting_event_ids = conflicting_event_ids


@dataclass(frozen=True)
class PreparedPoint:
    payload: dict[str, Any]
    event_payload_hash: str
    recorded_at: datetime
    latitude: float
    longitude: float


@dataclass(frozen=True)
class PreparedTrackingBatch:
    driver_id: int
    company_id: int
    source: str
    batch_id: str
    points: tuple[PreparedPoint, ...]
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class BatchPersistResult:
    received: int
    persisted: int
    duplicates: int
    batch_id: str
    trace_id: str
    event_ids_persisted: tuple[str, ...]
    event_ids_duplicate: tuple[str, ...]
    durability: str = "postgres_committed"


def _durability_mode() -> str:
    return os.getenv("INTERNAL_TRACKING_DURABILITY_MODE", "sync_db").strip().lower()


def require_durability_mode() -> None:
    mode = _durability_mode()
    if mode != "sync_db":
        raise RuntimeError(
            f"INTERNAL_TRACKING_DURABILITY_MODE invalide: {mode!r} (requis: sync_db)"
        )


def prepare_tracking_batch(
    *,
    driver_id: int,
    company_id: int,
    points: list[dict[str, Any]],
    source: str = "internal_http",
    client_batch_id: str | None = None,
) -> PreparedTrackingBatch:
    """OSRM/hash hors transaction. Lève PayloadHashError / ValueError."""
    require_durability_mode()
    prepared: list[PreparedPoint] = []
    events_for_batch: list[tuple[str, str]] = []

    for point in points:
        phash, _obj = compute_event_payload_hash_from_point(point)
        recorded_raw = str(point["recorded_at"])
        recorded_at = datetime.fromisoformat(recorded_raw.replace("Z", "+00:00"))
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=UTC)
        else:
            recorded_at = recorded_at.astimezone(UTC)
        # Coords persistées = valeurs float d'origine (validation déjà faite) ;
        # le hash utilise lat_e6 pour l'identité.
        lat = float(point["latitude"])
        lon = float(point["longitude"])
        enriched = {**point, "event_payload_hash": phash}
        prepared.append(
            PreparedPoint(
                payload=enriched,
                event_payload_hash=phash,
                recorded_at=recorded_at,
                latitude=lat,
                longitude=lon,
            )
        )
        events_for_batch.append((str(point["location_event_id"]), phash))

    batch_id = compute_batch_id(
        driver_id=driver_id,
        company_id=company_id,
        events=events_for_batch,
    )
    if client_batch_id is not None and str(client_batch_id).strip():
        if str(client_batch_id).strip().lower() != batch_id:
            raise ValueError("batch_id_mismatch")

    return PreparedTrackingBatch(
        driver_id=driver_id,
        company_id=company_id,
        source=source,
        batch_id=batch_id,
        points=tuple(prepared),
    )


def persist_tracking_batch(
    *,
    prepared: PreparedTrackingBatch,
    session: Session,
) -> BatchPersistResult:
    """Opérations transactionnelles uniquement — flush(), pas commit()."""
    require_durability_mode()

    # Re-vérifier chauffeur/tenant dans la TX
    row = (
        session.execute(
            text(
                """
            SELECT d.id, d.company_id, d.is_active, c.is_approved
            FROM driver d
            JOIN company c ON c.id = d.company_id
            WHERE d.id = :driver_id
            FOR UPDATE OF d
            """
            ),
            {"driver_id": prepared.driver_id},
        )
        .mappings()
        .first()
    )
    if row is None:
        raise ValueError("driver_not_found")
    if not row["is_active"]:
        raise ValueError("driver_inactive")
    if int(row["company_id"]) != int(prepared.company_id):
        raise PayloadConflictError("tenant_mismatch", [])
    if not row["is_approved"]:
        raise ValueError("company_not_approved")

    persisted_ids: list[str] = []
    duplicate_ids: list[str] = []
    conflict_payload: list[str] = []
    conflict_tenant: list[str] = []
    latest_for_driver: PreparedPoint | None = None

    for pt in prepared.points:
        eid = str(pt.payload["location_event_id"])
        inserted = session.execute(
            text(
                """
                INSERT INTO tracking_ingest_events (
                    driver_id, company_id, location_event_id,
                    event_payload_hash, payload_schema_version,
                    source, recorded_at
                ) VALUES (
                    :driver_id, :company_id, :location_event_id,
                    :event_payload_hash, :schema_version,
                    :source, :recorded_at
                )
                ON CONFLICT (driver_id, location_event_id) DO NOTHING
                RETURNING location_event_id
                """
            ),
            {
                "driver_id": prepared.driver_id,
                "company_id": prepared.company_id,
                "location_event_id": eid,
                "event_payload_hash": pt.event_payload_hash,
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "source": prepared.source,
                "recorded_at": pt.recorded_at,
            },
        ).first()

        if inserted is not None:
            persisted_ids.append(eid)
            if (
                latest_for_driver is None
                or pt.recorded_at > latest_for_driver.recorded_at
            ):
                latest_for_driver = pt
            _upsert_repair_pending(session, prepared.driver_id, eid, pt.recorded_at)
            continue

        existing = (
            session.execute(
                text(
                    """
                SELECT company_id, event_payload_hash, payload_schema_version
                FROM tracking_ingest_events
                WHERE driver_id = :driver_id AND location_event_id = :eid
                """
                ),
                {"driver_id": prepared.driver_id, "eid": eid},
            )
            .mappings()
            .first()
        )
        if existing is None:
            # Course rare — traiter comme erreur
            raise RuntimeError("ledger_conflict_race")

        if int(existing["company_id"]) != int(prepared.company_id):
            conflict_tenant.append(eid)
            continue
        if (
            str(existing["event_payload_hash"]) != pt.event_payload_hash
            or str(existing["payload_schema_version"]) != PAYLOAD_SCHEMA_VERSION
        ):
            conflict_payload.append(eid)
            continue

        duplicate_ids.append(eid)
        _upsert_repair_pending(session, prepared.driver_id, eid, pt.recorded_at)

    if conflict_tenant:
        raise PayloadConflictError("tenant_mismatch", conflict_tenant)
    if conflict_payload:
        raise PayloadConflictError("event_id_payload_conflict", conflict_payload)

    if latest_for_driver is not None:
        session.execute(
            text(
                """
                UPDATE driver
                SET latitude = :lat,
                    longitude = :lon,
                    last_position_update = :recorded_at
                WHERE id = :driver_id
                  AND (
                    last_position_update IS NULL
                    OR last_position_update < :recorded_at
                  )
                """
            ),
            {
                "lat": latest_for_driver.latitude,
                "lon": latest_for_driver.longitude,
                "recorded_at": latest_for_driver.recorded_at,
                "driver_id": prepared.driver_id,
            },
        )

    session.flush()
    return BatchPersistResult(
        received=len(prepared.points),
        persisted=len(persisted_ids),
        duplicates=len(duplicate_ids),
        batch_id=prepared.batch_id,
        trace_id=prepared.trace_id,
        event_ids_persisted=tuple(persisted_ids),
        event_ids_duplicate=tuple(duplicate_ids),
    )


def _upsert_repair_pending(
    session: Session,
    driver_id: int,
    location_event_id: str,
    target_recorded_at: datetime,
) -> None:
    session.execute(
        text(
            """
            INSERT INTO tracking_derived_repair_pending (
                driver_id, location_event_id, repair_kind,
                target_recorded_at, status, attempts
            ) VALUES (
                :driver_id, :eid, 'redis_canonical',
                :target_recorded_at, 'pending', 0
            )
            ON CONFLICT (driver_id, location_event_id, repair_kind)
            DO UPDATE SET
                target_recorded_at = EXCLUDED.target_recorded_at,
                status = 'pending',
                updated_at = now()
            WHERE EXCLUDED.target_recorded_at
                  >= tracking_derived_repair_pending.target_recorded_at
            """
        ),
        {
            "driver_id": driver_id,
            "eid": location_event_id,
            "target_recorded_at": target_recorded_at,
        },
    )


def _parse_redis_recorded_at(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    text_v = str(raw).strip()
    if not text_v:
        return None
    try:
        dt = datetime.fromisoformat(text_v.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def attempt_redis_canonical_repair(
    *,
    driver_id: int,
    company_id: int | None,
    latitude: float,
    longitude: float,
    recorded_at: datetime,
    location_event_id: str,
    location_mode: str = "mission_live",
) -> bool:
    """Best-effort post-commit — ne doit jamais lever vers la route.

    Une réparation plus ancienne ne masque jamais une position Redis plus récente.
    """
    try:
        from ext import redis_client

        if redis_client is None:
            return False
        target = recorded_at.astimezone(UTC)
        canonical_key = f"driver:{driver_id}:loc:canonical"
        legacy_key = f"driver:{driver_id}:loc"
        existing_raw = redis_client.hget(canonical_key, "recorded_at")
        existing_dt = _parse_redis_recorded_at(existing_raw)
        if existing_dt is not None and existing_dt > target:
            # Position déjà plus récente — réparation obsolete considérée OK
            return True
        recorded_iso = target.isoformat()
        mapping = {
            "company_id": str(company_id) if company_id is not None else "",
            "lat": str(latitude),
            "lon": str(longitude),
            "recorded_at": recorded_iso,
            "ts": recorded_iso,
            "location_mode": location_mode,
            "location_event_id": location_event_id,
            "source": "internal_http_f02",
        }
        redis_client.hset(canonical_key, mapping=mapping)
        redis_client.expire(canonical_key, 1200)
        redis_client.hset(legacy_key, mapping=mapping)
        redis_client.expire(legacy_key, 1200)
        return True
    except Exception as exc:
        logger.warning(
            "[ingest_durability] redis canonical repair failed driver_id=%s eid=%s: %s",
            driver_id,
            location_event_id,
            type(exc).__name__,
        )
        return False


def process_pending_repairs(*, limit: int = 50) -> dict[str, int]:
    """Worker : reprend les repair_pending (Redis down au post-commit)."""
    from ext import db

    rows = (
        db.session.execute(
            text(
                """
            SELECT r.id, r.driver_id, r.location_event_id, r.target_recorded_at,
                   e.company_id, e.recorded_at,
                   d.latitude, d.longitude
            FROM tracking_derived_repair_pending r
            JOIN tracking_ingest_events e
              ON e.driver_id = r.driver_id
             AND e.location_event_id = r.location_event_id
            JOIN driver d ON d.id = r.driver_id
            WHERE r.status = 'pending'
              AND r.repair_kind = 'redis_canonical'
            ORDER BY r.target_recorded_at ASC
            LIMIT :lim
            """
            ),
            {"lim": limit},
        )
        .mappings()
        .all()
    )

    done = 0
    failed = 0
    for row in rows:
        ok = attempt_redis_canonical_repair(
            driver_id=int(row["driver_id"]),
            company_id=int(row["company_id"])
            if row["company_id"] is not None
            else None,
            latitude=float(row["latitude"] or 0.0),
            longitude=float(row["longitude"] or 0.0),
            recorded_at=row["target_recorded_at"],
            location_event_id=str(row["location_event_id"]),
        )
        if ok:
            mark_repair_done_if_current(
                driver_id=int(row["driver_id"]),
                location_event_id=str(row["location_event_id"]),
                target_recorded_at=row["target_recorded_at"],
            )
            done += 1
        else:
            try:
                db.session.execute(
                    text(
                        """
                        UPDATE tracking_derived_repair_pending
                        SET attempts = attempts + 1,
                            last_error = 'redis_canonical_failed',
                            updated_at = now()
                        WHERE id = :id
                        """
                    ),
                    {"id": row["id"]},
                )
                db.session.commit()
            except Exception:
                db.session.rollback()
            failed += 1
    return {"processed": len(rows), "done": done, "failed": failed}


def mark_repair_done_if_current(
    *,
    driver_id: int,
    location_event_id: str,
    target_recorded_at: datetime,
) -> None:
    """Marque repair done hors TX principale (session courte)."""
    try:
        from ext import db

        db.session.execute(
            text(
                """
                UPDATE tracking_derived_repair_pending
                SET status = 'done', updated_at = now()
                WHERE driver_id = :driver_id
                  AND location_event_id = :eid
                  AND repair_kind = 'redis_canonical'
                  AND target_recorded_at <= :target_recorded_at
                  AND status = 'pending'
                """
            ),
            {
                "driver_id": driver_id,
                "eid": location_event_id,
                "target_recorded_at": target_recorded_at,
            },
        )
        db.session.commit()
    except Exception as exc:
        logger.warning(
            "[ingest_durability] mark_repair_done failed: %s", type(exc).__name__
        )
        try:
            from ext import db

            db.session.rollback()
        except Exception:
            pass
