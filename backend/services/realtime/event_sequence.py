# backend/services/realtime/event_sequence.py
"""Séquence monotone (Redis) pour la cohérence temps réel — dashboard entreprise.

Fail-closed pour le curseur de snapshot : Redis indisponible → ``None`` + health degraded
(jamais ``0`` présenté comme curseur sain).
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

_SEQ_KEY_PREFIX = "lirie:rt:seq:company:"


def _seq_key(company_id: int) -> str:
    return f"{_SEQ_KEY_PREFIX}{int(company_id)}"


def next_event_seq(company_id: int | None) -> int:
    """Incrémente et retourne le prochain ``event_seq`` (émission WS).

    Retourne 0 si Redis indisponible (l'émetteur omettra un seq valide ;
    le client doit rejeter ``event_seq <= 0`` pour les événements critiques).
    """
    if not company_id:
        return 0
    try:
        from ext import redis_client

        if redis_client is None:
            return 0
        return int(redis_client.incr(_seq_key(int(company_id))))
    except Exception:
        logger.debug(
            "[event_sequence] next_event_seq indisponible (company_id=%s)",
            company_id,
            exc_info=True,
        )
        return 0


def get_snapshot_cursor_status(
    company_id: int,
) -> tuple[int | None, Literal["ok", "degraded"]]:
    """Lit le curseur courant.

    Returns:
        (cursor, "ok") si Redis répond — cursor=0 si aucun événement encore.
        (None, "degraded") si Redis indisponible.
    """
    try:
        from ext import redis_client

        if redis_client is None:
            return None, "degraded"
        raw = redis_client.get(_seq_key(int(company_id)))
        if raw is None:
            return 0, "ok"
        return int(raw), "ok"
    except Exception:
        logger.debug(
            "[event_sequence] get_snapshot_cursor_status indisponible (company_id=%s)",
            company_id,
            exc_info=True,
        )
        return None, "degraded"


def current_snapshot_cursor(company_id: int) -> int | None:
    """Valeur courante du curseur pour le bootstrap, ou ``None`` si Redis dégradé."""
    cursor, _status = get_snapshot_cursor_status(company_id)
    return cursor
