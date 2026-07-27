"""Stockage PG non autoritaire pour le comparateur shadow."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

COMPARE_WINDOW_S = float(os.getenv("TRACKING_SHADOW_COMPARE_WINDOW_S", "30"))
EXPIRE_AFTER_S = float(os.getenv("TRACKING_SHADOW_OBSERVATION_TTL_S", "86400"))
LAG_THRESHOLD_MESSAGES = int(os.getenv("TRACKING_SHADOW_LAG_THRESHOLD", "100"))


def _database_url() -> str:
    for key in (
        "DATABASE_URL_DIRECT",
        "SQLALCHEMY_DATABASE_URI_DIRECT",
        "POSTGRES_URL",
    ):
        url = os.getenv(key)
        if url:
            break
    else:
        url = os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")
    if not url:
        raise RuntimeError("DATABASE_URL manquant pour shadow observations")
    url = url.replace("postgres://", "postgresql://", 1)
    if "@pgbouncer:" in url or "@atmr-pgbouncer" in url:
        url = url.replace("@pgbouncer:", "@postgres:").replace(
            "@atmr-pgbouncer:", "@postgres:"
        )
        url = url.replace(":6432/", ":5432/")
    return url


def _engine() -> Engine:
    return create_engine(_database_url(), pool_pre_ping=True)


def mark_comparison_unavailable(
    *,
    driver_id: int,
    location_event_id: str,
    company_id: int | None,
    side: str,
    engine: Engine | None = None,
) -> bool:
    """Upsert best-effort : publication Kafka KO → comparison_unavailable."""
    if not location_event_id or driver_id <= 0:
        return False
    eng = engine or _engine()
    now = datetime.now(UTC)
    expires = now + timedelta(seconds=EXPIRE_AFTER_S)
    try:
        with eng.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO tracking_shadow_observations (
                        driver_id, location_event_id, company_id,
                        comparison_state, result, compared_at, expires_at,
                        created_at, updated_at
                    ) VALUES (
                        :driver_id, :eid, :company_id,
                        'comparison_unavailable', :result, :now, :expires,
                        :now, :now
                    )
                    ON CONFLICT (driver_id, location_event_id) DO UPDATE SET
                        comparison_state = 'comparison_unavailable',
                        result = EXCLUDED.result,
                        compared_at = EXCLUDED.compared_at,
                        updated_at = EXCLUDED.updated_at,
                        company_id = COALESCE(
                            EXCLUDED.company_id,
                            tracking_shadow_observations.company_id
                        )
                    """
                ),
                {
                    "driver_id": driver_id,
                    "eid": location_event_id,
                    "company_id": company_id,
                    "result": f"publish_failed_{side}",
                    "now": now,
                    "expires": expires,
                },
            )
            conn.commit()
        return True
    except Exception:
        logger.exception(
            "[shadow] mark_comparison_unavailable failed driver=%s eid=%s",
            driver_id,
            location_event_id,
        )
        return False


def upsert_direct_observation(
    *,
    driver_id: int,
    location_event_id: str,
    company_id: int | None,
    fingerprint: str,
    accept_status: str,
    accept_reason: str,
    engine: Engine | None = None,
    consumer_lag: int = 0,
) -> str:
    """Enregistre le côté direct ; compare si shadow déjà présent."""
    return _upsert_side(
        side="direct",
        driver_id=driver_id,
        location_event_id=location_event_id,
        company_id=company_id,
        fingerprint=fingerprint,
        accept_status=accept_status,
        accept_reason=accept_reason,
        engine=engine,
        consumer_lag=consumer_lag,
    )


def upsert_shadow_observation(
    *,
    driver_id: int,
    location_event_id: str,
    company_id: int | None,
    fingerprint: str,
    accept_status: str,
    accept_reason: str,
    engine: Engine | None = None,
    consumer_lag: int = 0,
) -> str:
    """Enregistre le côté shadow ; compare si direct déjà présent."""
    return _upsert_side(
        side="shadow",
        driver_id=driver_id,
        location_event_id=location_event_id,
        company_id=company_id,
        fingerprint=fingerprint,
        accept_status=accept_status,
        accept_reason=accept_reason,
        engine=engine,
        consumer_lag=consumer_lag,
    )


def _upsert_side(
    *,
    side: str,
    driver_id: int,
    location_event_id: str,
    company_id: int | None,
    fingerprint: str,
    accept_status: str,
    accept_reason: str,
    engine: Engine | None,
    consumer_lag: int,
) -> str:
    eng = engine or _engine()
    now = datetime.now(UTC)
    grace = COMPARE_WINDOW_S
    if consumer_lag > LAG_THRESHOLD_MESSAGES:
        # Lag élevé : pas de missing définitif — deadline repoussée.
        grace = COMPARE_WINDOW_S * 10
    deadline = now + timedelta(seconds=grace)
    expires = now + timedelta(seconds=EXPIRE_AFTER_S)
    waiting_state = "waiting_shadow" if side == "direct" else "waiting_direct"

    with eng.connect() as conn:
        row = (
            conn.execute(
                text(
                    """
                SELECT direct_fingerprint, direct_accept_status, direct_accept_reason,
                       shadow_fingerprint, shadow_accept_status, shadow_accept_reason,
                       comparison_state
                FROM tracking_shadow_observations
                WHERE driver_id = :driver_id AND location_event_id = :eid
                FOR UPDATE
                """
                ),
                {"driver_id": driver_id, "eid": location_event_id},
            )
            .mappings()
            .first()
        )

        if row is None:
            if side == "direct":
                conn.execute(
                    text(
                        """
                        INSERT INTO tracking_shadow_observations (
                            driver_id, location_event_id, company_id,
                            fingerprint_schema_version,
                            direct_fingerprint, direct_accept_status,
                            direct_accept_reason, direct_seen_at,
                            comparison_deadline_at, comparison_state,
                            expires_at, created_at, updated_at
                        ) VALUES (
                            :driver_id, :eid, :company_id, 1,
                            :fp, :astatus, :areason, :now,
                            :deadline, :state, :expires, :now, :now
                        )
                        """
                    ),
                    {
                        "driver_id": driver_id,
                        "eid": location_event_id,
                        "company_id": company_id,
                        "fp": fingerprint,
                        "astatus": accept_status,
                        "areason": accept_reason,
                        "now": now,
                        "deadline": deadline,
                        "state": waiting_state,
                        "expires": expires,
                    },
                )
            else:
                conn.execute(
                    text(
                        """
                        INSERT INTO tracking_shadow_observations (
                            driver_id, location_event_id, company_id,
                            fingerprint_schema_version,
                            shadow_fingerprint, shadow_accept_status,
                            shadow_accept_reason, shadow_seen_at,
                            comparison_deadline_at, comparison_state,
                            expires_at, created_at, updated_at
                        ) VALUES (
                            :driver_id, :eid, :company_id, 1,
                            :fp, :astatus, :areason, :now,
                            :deadline, :state, :expires, :now, :now
                        )
                        """
                    ),
                    {
                        "driver_id": driver_id,
                        "eid": location_event_id,
                        "company_id": company_id,
                        "fp": fingerprint,
                        "astatus": accept_status,
                        "areason": accept_reason,
                        "now": now,
                        "deadline": deadline,
                        "state": waiting_state,
                        "expires": expires,
                    },
                )
            conn.commit()
            return waiting_state

        if row["comparison_state"] == "comparison_unavailable":
            conn.commit()
            return "comparison_unavailable"

        direct_fp = row["direct_fingerprint"]
        shadow_fp = row["shadow_fingerprint"]
        direct_status = row["direct_accept_status"]
        shadow_status = row["shadow_accept_status"]

        if side == "direct":
            direct_fp = fingerprint
            direct_status = accept_status
            conn.execute(
                text(
                    """
                    UPDATE tracking_shadow_observations SET
                        direct_fingerprint = :fp,
                        direct_accept_status = :astatus,
                        direct_accept_reason = :areason,
                        direct_seen_at = :now,
                        company_id = COALESCE(:company_id, company_id),
                        comparison_deadline_at = COALESCE(
                            comparison_deadline_at, :deadline
                        ),
                        updated_at = :now
                    WHERE driver_id = :driver_id AND location_event_id = :eid
                    """
                ),
                {
                    "fp": fingerprint,
                    "astatus": accept_status,
                    "areason": accept_reason,
                    "now": now,
                    "company_id": company_id,
                    "deadline": deadline,
                    "driver_id": driver_id,
                    "eid": location_event_id,
                },
            )
        else:
            shadow_fp = fingerprint
            shadow_status = accept_status
            conn.execute(
                text(
                    """
                    UPDATE tracking_shadow_observations SET
                        shadow_fingerprint = :fp,
                        shadow_accept_status = :astatus,
                        shadow_accept_reason = :areason,
                        shadow_seen_at = :now,
                        company_id = COALESCE(:company_id, company_id),
                        comparison_deadline_at = COALESCE(
                            comparison_deadline_at, :deadline
                        ),
                        updated_at = :now
                    WHERE driver_id = :driver_id AND location_event_id = :eid
                    """
                ),
                {
                    "fp": fingerprint,
                    "astatus": accept_status,
                    "areason": accept_reason,
                    "now": now,
                    "company_id": company_id,
                    "deadline": deadline,
                    "driver_id": driver_id,
                    "eid": location_event_id,
                },
            )

        state = _compare_state(
            direct_fp=direct_fp,
            shadow_fp=shadow_fp,
            direct_status=direct_status,
            shadow_status=shadow_status,
        )
        if state in (
            "matched",
            "payload_mismatch",
            "acceptance_mismatch",
        ):
            conn.execute(
                text(
                    """
                    UPDATE tracking_shadow_observations SET
                        comparison_state = :state,
                        result = :state,
                        compared_at = :now,
                        updated_at = :now
                    WHERE driver_id = :driver_id AND location_event_id = :eid
                    """
                ),
                {
                    "state": state,
                    "now": now,
                    "driver_id": driver_id,
                    "eid": location_event_id,
                },
            )
        elif state in ("waiting_direct", "waiting_shadow"):
            conn.execute(
                text(
                    """
                    UPDATE tracking_shadow_observations SET
                        comparison_state = :state,
                        updated_at = :now
                    WHERE driver_id = :driver_id AND location_event_id = :eid
                    """
                ),
                {
                    "state": state,
                    "now": now,
                    "driver_id": driver_id,
                    "eid": location_event_id,
                },
            )
        conn.commit()
        return state


def _compare_state(
    *,
    direct_fp: str | None,
    shadow_fp: str | None,
    direct_status: str | None,
    shadow_status: str | None,
) -> str:
    if direct_fp is None and shadow_fp is not None:
        return "waiting_direct"
    if shadow_fp is None and direct_fp is not None:
        return "waiting_shadow"
    if direct_fp is None and shadow_fp is None:
        return "waiting_shadow"
    if direct_status and shadow_status and direct_status != shadow_status:
        # Alignement grossier accepted* vs accepted
        d_ok = str(direct_status).startswith("accepted")
        s_ok = str(shadow_status).startswith("accepted")
        if d_ok != s_ok:
            return "acceptance_mismatch"
    if direct_fp != shadow_fp:
        return "payload_mismatch"
    return "matched"


def expire_waiting_observations(
    *,
    engine: Engine | None = None,
    consumer_lag: int = 0,
) -> list[dict[str, Any]]:
    """Passe en expired les waiting dont deadline dépassée (sauf si lag élevé)."""
    if consumer_lag > LAG_THRESHOLD_MESSAGES:
        return []
    eng = engine or _engine()
    now = datetime.now(UTC)
    expired: list[dict[str, Any]] = []
    with eng.connect() as conn:
        rows = (
            conn.execute(
                text(
                    """
                SELECT driver_id, location_event_id, comparison_state,
                       direct_fingerprint, shadow_fingerprint
                FROM tracking_shadow_observations
                WHERE comparison_state IN ('waiting_direct', 'waiting_shadow')
                  AND comparison_deadline_at IS NOT NULL
                  AND comparison_deadline_at < :now
                FOR UPDATE SKIP LOCKED
                LIMIT 200
                """
                ),
                {"now": now},
            )
            .mappings()
            .all()
        )
        for row in rows:
            result = (
                "shadow_missing_in_direct"
                if row["comparison_state"] == "waiting_direct"
                else "shadow_missing_in_kafka"
            )
            conn.execute(
                text(
                    """
                    UPDATE tracking_shadow_observations SET
                        comparison_state = 'expired',
                        result = :result,
                        compared_at = :now,
                        updated_at = :now
                    WHERE driver_id = :driver_id AND location_event_id = :eid
                    """
                ),
                {
                    "result": result,
                    "now": now,
                    "driver_id": row["driver_id"],
                    "eid": row["location_event_id"],
                },
            )
            expired.append(
                {
                    "driver_id": row["driver_id"],
                    "location_event_id": row["location_event_id"],
                    "result": result,
                }
            )
        conn.commit()
    return expired


def extend_deadlines_after_lag_recovery(
    *,
    engine: Engine | None = None,
) -> int:
    """Après retour sous seuil de lag : nouvelle grâce de 30s sur waiting."""
    eng = engine or _engine()
    now = datetime.now(UTC)
    new_deadline = now + timedelta(seconds=COMPARE_WINDOW_S)
    with eng.connect() as conn:
        result = conn.execute(
            text(
                """
                UPDATE tracking_shadow_observations SET
                    comparison_deadline_at = :deadline,
                    updated_at = :now
                WHERE comparison_state IN ('waiting_direct', 'waiting_shadow')
                """
            ),
            {"deadline": new_deadline, "now": now},
        )
        conn.commit()
        return int(result.rowcount or 0)
