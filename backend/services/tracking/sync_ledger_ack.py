"""P0-E — ACK ledger sync : preuve durable + ownership du commit PG.

``persist_location_event_with_outbox`` ne commit pas. Ce module est le seul
propriétaire de la TX pour le chemin HTTP sync durable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.orm import Session

from services.tracking.persist_with_outbox import (
    PersistConflictError,
    persist_location_event_with_outbox,
)

logger = logging.getLogger(__name__)


def _maybe_promote_sync_ledger(
    *,
    persist_result: dict[str, Any],
    driver_id: int,
    company_id: int,
    capture_id: str | None,
    location_event_id: str,
    tracking_session_id: str | None,
    session_generation: int | None,
    sequence_id: int | None,
    mission_id: int | None,
    recorded_at: Any,
    latitude: float,
    longitude: float,
    location_mode: str,
    speed_mps: float | None,
    heading: float | None,
    accuracy_m: float | None,
    source: str,
) -> None:
    from services.tracking.location_candidate import (
        build_durable_location_proof,
        is_pg_first_canonical_enabled,
        promote_location_candidate,
    )
    from services.tracking.persist_with_outbox import _parse_recorded_at

    if not is_pg_first_canonical_enabled():
        return
    if persist_result.get("status") != "persisted":
        return
    try:
        proof = build_durable_location_proof(
            pg_committed=True,
            driver_id=driver_id,
            company_id=company_id,
            capture_id=capture_id or persist_result.get("capture_id"),
            location_event_id=location_event_id,
            tracking_session_id=tracking_session_id,
            session_generation=session_generation,
            sequence_id=sequence_id,
            mission_id=mission_id,
            recorded_at=_parse_recorded_at(recorded_at),
            latitude=latitude,
            longitude=longitude,
            accept_status="accepted_canonical",
            location_mode=location_mode,
            speed=speed_mps,
            heading=heading,
            accuracy=accuracy_m,
            source=source,
            transport="http",
        )
        promote_location_candidate(proof)
    except Exception:
        logger.warning(
            "[sync_ledger_ack] promotion canonical échouée driver_id=%s "
            "capture_id=%s (PG déjà commité)",
            driver_id,
            capture_id,
            exc_info=True,
        )


OutcomeKind = Literal[
    "durable_ok",
    "conflict_409",
    "ledger_failed_503",
    "ids_missing",
]


@dataclass(frozen=True)
class SyncLedgerAckResult:
    """Résultat du helper propriétaire de TX ledger."""

    kind: OutcomeKind
    reason: str
    location_event_id: str | None = None
    tracking_session_id: str | None = None
    session_generation: int | None = None
    sequence_id: int | None = None
    existing_location_event_id: str | None = None
    persist_result: dict[str, Any] | None = None


def durable_proof(persist_result: dict[str, Any] | None) -> bool:
    """True ssi le dict outbox prouve un événement déjà/à-insérer durable.

    Ne prouve **pas** le commit PG — le commit reste obligatoire ensuite.
    """
    if not isinstance(persist_result, dict):
        return False
    status = str(persist_result.get("status") or "")
    reason = str(persist_result.get("reason") or "")
    return (status == "persisted" and reason == "inserted") or (
        status == "duplicate" and reason == "same_event_already_persisted"
    )


def _parse_optional_int(raw: Any) -> int | None:
    if raw is None or raw is False:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def extract_sync_ledger_ids(
    payload: dict[str, Any] | None,
) -> tuple[str | None, int | None, int | None]:
    """Extrait session / génération / séquence depuis le body HTTP."""
    if not isinstance(payload, dict):
        return None, None, None
    raw_sid = payload.get("tracking_session_id")
    sid = str(raw_sid).strip() if raw_sid is not None and str(raw_sid).strip() else None
    gen = _parse_optional_int(payload.get("session_generation"))
    seq = _parse_optional_int(payload.get("sequence_id"))
    return sid, gen, seq


def try_commit_sync_ledger_ack(
    session: Session,
    *,
    driver_id: int,
    company_id: int,
    location_event_id: str,
    tracking_session_id: str | None,
    session_generation: int | None,
    sequence_id: int | None,
    latitude: float,
    longitude: float,
    recorded_at: Any,
    source: str = "http",
    location_mode: str = "mission_live",
    accuracy_m: float | None = None,
    speed_mps: float | None = None,
    heading: float | None = None,
    mission_id: int | None = None,
    capture_id: str | None = None,
) -> SyncLedgerAckResult:
    """Persiste le ledger + commit. Jamais ``durable_ok`` si commit KO.

    Ownership TX :
    - preuve non durable → rollback
    - PersistConflictError → rollback → 409
    - exception / commit KO → rollback → 503

    Ownership claim Redis (P0-C-LEDGER-SERVER) :
    - ce module ne release **pas** le claim ; le caller (route HTTP) doit
      ``release_location_event_id`` sur tout chemin non ``durable_ok``
      (notamment ``ids_missing``), sinon claim orphelin.
    """
    if (
        not tracking_session_id
        or session_generation is None
        or sequence_id is None
        or not location_event_id
    ):
        return SyncLedgerAckResult(
            kind="ids_missing",
            reason="ledger_ids_missing",
            location_event_id=location_event_id or None,
            tracking_session_id=tracking_session_id,
            session_generation=session_generation,
            sequence_id=sequence_id,
        )

    try:
        persist_result = persist_location_event_with_outbox(
            session,
            driver_id=driver_id,
            company_id=company_id,
            location_event_id=location_event_id,
            tracking_session_id=tracking_session_id,
            session_generation=int(session_generation),
            sequence_id=int(sequence_id),
            latitude=latitude,
            longitude=longitude,
            recorded_at=recorded_at,
            source=source,
            location_mode=location_mode,
            accuracy_m=accuracy_m,
            speed_mps=speed_mps,
            heading=heading,
            mission_id=mission_id,
            capture_id=capture_id,
        )
    except PersistConflictError as exc:
        try:
            session.rollback()
        except Exception:
            logger.warning(
                "[sync_ledger_ack] rollback après PersistConflictError échoué",
                exc_info=True,
            )
        return SyncLedgerAckResult(
            kind="conflict_409",
            reason=str(exc.code or "event_id_payload_conflict"),
            location_event_id=location_event_id,
            tracking_session_id=tracking_session_id,
            session_generation=session_generation,
            sequence_id=sequence_id,
        )
    except Exception:
        logger.exception(
            "[sync_ledger_ack] persist_outbox KO driver_id=%s eid=%s",
            driver_id,
            location_event_id,
        )
        try:
            session.rollback()
        except Exception:
            logger.warning(
                "[sync_ledger_ack] rollback après exception persist échoué",
                exc_info=True,
            )
        return SyncLedgerAckResult(
            kind="ledger_failed_503",
            reason="ledger_persist_failed",
            location_event_id=location_event_id,
            tracking_session_id=tracking_session_id,
            session_generation=session_generation,
            sequence_id=sequence_id,
        )

    if not durable_proof(persist_result):
        try:
            session.rollback()
        except Exception:
            logger.warning(
                "[sync_ledger_ack] rollback après preuve non durable échoué",
                exc_info=True,
            )
        reason = str(persist_result.get("reason") or "duplicate_unproven")
        existing = persist_result.get("existing_location_event_id")
        if reason == "session_sequence_already_persisted":
            return SyncLedgerAckResult(
                kind="conflict_409",
                reason=reason,
                location_event_id=location_event_id,
                tracking_session_id=tracking_session_id,
                session_generation=session_generation,
                sequence_id=sequence_id,
                existing_location_event_id=(
                    str(existing) if existing is not None else None
                ),
                persist_result=persist_result,
            )
        # duplicate_unproven et autres → retryable infra
        return SyncLedgerAckResult(
            kind="ledger_failed_503",
            reason=reason if reason else "duplicate_unproven",
            location_event_id=location_event_id,
            tracking_session_id=tracking_session_id,
            session_generation=session_generation,
            sequence_id=sequence_id,
            persist_result=persist_result,
        )

    try:
        session.commit()
    except Exception:
        logger.exception(
            "[sync_ledger_ack] commit KO driver_id=%s eid=%s — jamais persisted_sync",
            driver_id,
            location_event_id,
        )
        try:
            session.rollback()
        except Exception:
            logger.warning(
                "[sync_ledger_ack] rollback après commit KO échoué",
                exc_info=True,
            )
        return SyncLedgerAckResult(
            kind="ledger_failed_503",
            reason="ledger_persist_failed",
            location_event_id=location_event_id,
            tracking_session_id=tracking_session_id,
            session_generation=session_generation,
            sequence_id=sequence_id,
            persist_result=persist_result,
        )

    _maybe_promote_sync_ledger(
        persist_result=persist_result,
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
        location_mode=location_mode,
        speed_mps=speed_mps,
        heading=heading,
        accuracy_m=accuracy_m,
        source=source,
    )
    return SyncLedgerAckResult(
        kind="durable_ok",
        reason=str(persist_result.get("reason") or "inserted"),
        location_event_id=location_event_id,
        tracking_session_id=tracking_session_id,
        session_generation=session_generation,
        sequence_id=sequence_id,
        persist_result=persist_result,
    )
