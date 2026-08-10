"""Attache tracking_session_id / sequence_id au chemin HTTP → Kafka (Phase 1 outbox).

Le PUT ``/driver/me/location`` historique n'envoyait pas ces champs. Sans eux,
``persist_kafka_outbox`` DLQ ``tracking_session_id_missing``. Ce module :
- réutilise la session mobile si fournie ;
- sinon réutilise la session Redis active ou en crée une ``http-legacy-{driver_id}`` ;
- alloue ``sequence_id`` (payload ou Redis INCR, seedé depuis PG si la clé expire).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _seed_and_allocate_http_sequence(
    *,
    redis_client: Any,
    session: Session,
    driver_id: int,
    tracking_session_id: str,
) -> int:
    """Alloue un sequence_id monotone via Redis, seedé depuis ``max_seen_sequence`` PG.

    Sans seed, l'expire Redis (24h) ou un flush fait redémarrer le compteur à 1
    alors que PG a déjà ces séquences → UniqueViolation
    ``uq_tracking_ingest_session_sequence`` (poison consumer).
    """
    seq_key = f"tracking:http_seq:{driver_id}:{tracking_session_id}"
    try:
        existing = redis_client.get(seq_key)
    except Exception as exc:
        raise RuntimeError("sequence_id_allocate_failed") from exc

    if existing is None:
        row = (
            session.execute(
                text(
                    """
                    SELECT COALESCE(max_seen_sequence, 0) AS max_seen
                    FROM tracking_session_state
                    WHERE driver_id = :driver_id AND tracking_session_id = :sid
                    """
                ),
                {"driver_id": driver_id, "sid": tracking_session_id},
            )
            .mappings()
            .first()
        )
        seed = int(row["max_seen"]) if row else 0
        try:
            # SET NX : course entre workers — le perdant INCR sur la valeur déjà seedée.
            redis_client.set(seq_key, seed, nx=True, ex=86400)
        except Exception as exc:
            raise RuntimeError("sequence_id_allocate_failed") from exc

    try:
        sequence_id = int(redis_client.incr(seq_key))
        redis_client.expire(seq_key, 86400)
    except Exception as exc:
        raise RuntimeError("sequence_id_allocate_failed") from exc
    return sequence_id


def ensure_http_tracking_session_fields(
    *,
    driver_id: int,
    company_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Retourne une copie du payload enrichie pour l'outbox Kafka.

    Lève RuntimeError si la session ne peut pas être enregistrée.
    """
    from celery_app import get_flask_app
    from ext import db, redis_client
    from services.tracking.session_registry import register_tracking_session

    enriched = dict(payload)
    raw_sid = enriched.get("tracking_session_id")
    sid = str(raw_sid).strip() if raw_sid is not None and str(raw_sid).strip() else ""

    if not sid and redis_client is not None:
        try:
            existing = redis_client.get(f"driver:{driver_id}:active_tracking_session")
            if existing:
                sid = (
                    existing.decode("utf-8", errors="replace")
                    if isinstance(existing, bytes)
                    else str(existing)
                ).strip()
        except Exception:
            logger.debug(
                "[http_session_bridge] lecture session Redis échouée", exc_info=True
            )

    if not sid:
        sid = f"http-legacy-{driver_id}"

    seq_raw = enriched.get("sequence_id")
    sequence_from_payload = seq_raw is not None and str(seq_raw).strip() != ""

    app = get_flask_app()
    with app.app_context():
        try:
            auth = register_tracking_session(
                db.session,
                driver_id=driver_id,
                company_id=company_id,
                tracking_session_id=sid,
                tracking_session_started_at=None,
            )
            if sequence_from_payload:
                sequence_id = int(seq_raw)
            elif redis_client is not None:
                sequence_id = _seed_and_allocate_http_sequence(
                    redis_client=redis_client,
                    session=db.session,
                    driver_id=driver_id,
                    tracking_session_id=sid,
                )
            else:
                raise RuntimeError("sequence_id_allocate_failed")
            db.session.commit()
        except Exception:
            db.session.rollback()
            raise
        finally:
            db.session.remove()

    enriched["tracking_session_id"] = sid
    enriched["session_generation"] = int(auth["session_generation"])
    enriched["sequence_id"] = sequence_id

    if redis_client is not None:
        try:
            redis_client.setex(f"driver:{driver_id}:active_tracking_session", 1800, sid)
        except Exception:
            logger.debug(
                "[http_session_bridge] écriture session Redis échouée", exc_info=True
            )

    return enriched
