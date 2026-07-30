# backend/services/realtime/event_sequence.py
"""Séquence monotone (Redis) pour la cohérence temps réel — Lot 3 perf espace entreprise.

Contexte:
    Le frontend applique les événements Socket.IO uniquement s'ils sont plus récents
    que le `snapshot_cursor` renvoyé par `GET /companies/me/dashboard/bootstrap`
    (voir docs/perf-company-space-lot3-dashboard.md). `updated_at` seul est insuffisant
    (horloge, granularité, retries) : on utilise un curseur entier strictement croissant.

Principe:
    - Un compteur Redis par entreprise (`INCR`) fournit `event_seq` à chaque émission
      Socket.IO pertinente pour le dashboard (booking_*, dispatch_*, etc.).
    - Le bootstrap lit la valeur courante du même compteur (`GET`, sans l'incrémenter)
      pour produire `snapshot_cursor`. Tout événement émis après coup aura donc
      `event_seq > snapshot_cursor` (le compteur ne fait qu'augmenter).
    - Fail-open : en l'absence de Redis, on retourne 0 (le frontend retombe sur un
      comportement « accepter tout », comme avant ce lot).

Pas de PII stockée : uniquement un entier par entreprise.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SEQ_KEY_PREFIX = "lirie:rt:seq:company:"


def _seq_key(company_id: int) -> str:
    return f"{_SEQ_KEY_PREFIX}{int(company_id)}"


def next_event_seq(company_id: int | None) -> int:
    """Incrémente et retourne le prochain `event_seq` pour l'entreprise (émission WS).

    Retourne 0 si Redis est indisponible ou `company_id` invalide (fail-open : le
    payload n'aura simplement pas de `event_seq`, comportement pré-Lot 3).
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


def current_snapshot_cursor(company_id: int) -> int:
    """Valeur courante du curseur pour le bootstrap (0 si aucun événement émis encore).

    Ne modifie pas le compteur (lecture seule) — les événements futurs incrémenteront
    via `next_event_seq` et resteront donc strictement supérieurs à cette valeur.
    """
    try:
        from ext import redis_client

        if redis_client is None:
            return 0
        raw = redis_client.get(_seq_key(int(company_id)))
        return int(raw) if raw is not None else 0
    except Exception:
        logger.debug(
            "[event_sequence] current_snapshot_cursor indisponible (company_id=%s)",
            company_id,
            exc_info=True,
        )
        return 0
