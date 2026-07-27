"""Registre de sessions tracking — autorité session_generation (Annexe A.2 / A.3)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class SessionRegistryError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def _parse_started_at(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(UTC)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError as exc:
        raise SessionRegistryError(
            "invalid_started_at", "tracking_session_started_at invalide"
        ) from exc


def register_tracking_session(
    session: Session,
    *,
    driver_id: int,
    company_id: int,
    tracking_session_id: str,
    tracking_session_started_at: str | None,
) -> dict[str, Any]:
    """Ouverture idempotente. Supersede l'ancienne session active (multi-appareils)."""
    sid = (tracking_session_id or "").strip()
    if not sid:
        raise SessionRegistryError(
            "tracking_session_id_missing", "tracking_session_id requis"
        )

    existing = (
        session.execute(
            text(
                """
            SELECT tracking_session_id, session_generation, status, final_sequence_id
            FROM tracking_sessions
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            FOR UPDATE
            """
            ),
            {"driver_id": driver_id, "sid": sid},
        )
        .mappings()
        .first()
    )

    if existing is not None:
        return {
            "tracking_session_id": existing["tracking_session_id"],
            "session_generation": int(existing["session_generation"]),
            "first_sequence_id": 1,
            "status": str(existing["status"]),
        }

    # Supersede session active éventuelle
    session.execute(
        text(
            """
            UPDATE tracking_sessions
            SET status = 'superseded', updated_at = NOW()
            WHERE driver_id = :driver_id AND status = 'active'
            """
        ),
        {"driver_id": driver_id},
    )

    generation = session.execute(
        text("SELECT nextval('tracking_session_generation_seq')")
    ).scalar_one()
    started_at = _parse_started_at(tracking_session_started_at)

    session.execute(
        text(
            """
            INSERT INTO tracking_sessions (
                driver_id, company_id, tracking_session_id, session_generation,
                status, started_at
            ) VALUES (
                :driver_id, :company_id, :sid, :generation,
                'active', :started_at
            )
            """
        ),
        {
            "driver_id": driver_id,
            "company_id": company_id,
            "sid": sid,
            "generation": int(generation),
            "started_at": started_at,
        },
    )

    session.execute(
        text(
            """
            INSERT INTO tracking_session_state (
                driver_id, company_id, tracking_session_id, session_generation,
                contiguous_persisted_through, max_seen_sequence,
                first_seen_at, last_seen_at
            ) VALUES (
                :driver_id, :company_id, :sid, :generation,
                0, 0, :started_at, :started_at
            )
            ON CONFLICT (driver_id, tracking_session_id) DO NOTHING
            """
        ),
        {
            "driver_id": driver_id,
            "company_id": company_id,
            "sid": sid,
            "generation": int(generation),
            "started_at": started_at,
        },
    )

    return {
        "tracking_session_id": sid,
        "session_generation": int(generation),
        "first_sequence_id": 1,
        "status": "active",
    }


def close_tracking_session(
    session: Session,
    *,
    driver_id: int,
    tracking_session_id: str,
    final_sequence_id: int | None,
) -> dict[str, Any]:
    sid = (tracking_session_id or "").strip()
    row = (
        session.execute(
            text(
                """
            SELECT status FROM tracking_sessions
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            FOR UPDATE
            """
            ),
            {"driver_id": driver_id, "sid": sid},
        )
        .mappings()
        .first()
    )
    if row is None:
        raise SessionRegistryError(
            "tracking_session_not_registered",
            "Session inconnue",
            http_status=404,
        )

    closed_at = datetime.now(UTC)
    session.execute(
        text(
            """
            UPDATE tracking_sessions
            SET status = 'closed',
                closed_at = :closed_at,
                final_sequence_id = :final_seq,
                updated_at = NOW()
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            """
        ),
        {
            "driver_id": driver_id,
            "sid": sid,
            "closed_at": closed_at,
            "final_seq": final_sequence_id,
        },
    )
    session.execute(
        text(
            """
            UPDATE tracking_session_state
            SET closed_at = :closed_at
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            """
        ),
        {"driver_id": driver_id, "sid": sid, "closed_at": closed_at},
    )
    return {
        "tracking_session_id": sid,
        "status": "closed",
        "closed_at": closed_at.isoformat(),
        "final_sequence_id": final_sequence_id,
    }


def resolve_authoritative_session(
    session: Session,
    *,
    driver_id: int,
    company_id: int,
    tracking_session_id: str,
    claimed_generation: int | None,
    sequence_id: int | None,
) -> dict[str, Any]:
    """Vérifie session + génération (ne jamais faire confiance au mobile)."""
    sid = (tracking_session_id or "").strip()
    row = (
        session.execute(
            text(
                """
            SELECT driver_id, company_id, session_generation, status, final_sequence_id
            FROM tracking_sessions
            WHERE driver_id = :driver_id AND tracking_session_id = :sid
            """
            ),
            {"driver_id": driver_id, "sid": sid},
        )
        .mappings()
        .first()
    )

    if row is None:
        raise SessionRegistryError(
            "tracking_session_not_registered",
            "Session non enregistrée",
            http_status=400,
        )
    if int(row["driver_id"]) != int(driver_id):
        raise SessionRegistryError(
            "session_forbidden", "Session d'un autre chauffeur", http_status=403
        )
    if int(row["company_id"]) != int(company_id):
        raise SessionRegistryError(
            "session_forbidden", "Tenant mismatch", http_status=403
        )

    auth_gen = int(row["session_generation"])
    if claimed_generation is not None and int(claimed_generation) != auth_gen:
        raise SessionRegistryError(
            "session_generation_mismatch",
            "session_generation ne correspond pas au registre",
            http_status=409,
        )

    status = str(row["status"])
    if status == "closed" and sequence_id is not None:
        final_seq = row["final_sequence_id"]
        if final_seq is not None and int(sequence_id) > int(final_seq):
            raise SessionRegistryError(
                "sequence_after_session_close",
                "sequence_id au-delà de final_sequence_id",
                http_status=400,
            )

    return {
        "session_generation": auth_gen,
        "status": status,
        "final_sequence_id": row["final_sequence_id"],
    }
