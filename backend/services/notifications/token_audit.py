# backend/services/notifications/token_audit.py
"""P0.4: Détection collision tokens (driver vs company) pour preuve irréfutable.

Exécuté uniquement en DEBUG_NOTIF_ROUTING=1.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

DEBUG_NOTIF_ROUTING = os.environ.get("DEBUG_NOTIF_ROUTING", "").lower() in (
    "1",
    "true",
    "yes",
)


def _token_hash(token: str, length: int = 8) -> str:
    """Hash SHA256 du token, tronqué (jamais logger le token brut)."""
    if not token:
        return ""
    return hashlib.sha256(token.encode()).hexdigest()[:length]


def check_token_collision(
    *,
    driver_tokens: list[str],
    company_token: str | None,
    driver_id: int,
    company_user_id: int | None,
    trace_id: str | None = None,
) -> bool:
    """Détecte si un token driver est identique au token company (collision).

    Returns:
        True si collision détectée (warning loggé), False sinon.
    """
    if not DEBUG_NOTIF_ROUTING or not company_token:
        return False

    driver_hashes = {_token_hash(t) for t in driver_tokens if t}
    company_hash = _token_hash(company_token)

    if company_hash in driver_hashes:
        collision_hashes = [h for h in driver_hashes if h == company_hash]
        try:
            from ext import app_logger

            app_logger.warning(
                "[token_audit] COLLISION trace_id=%s driver_id=%s company_user_id=%s "
                "collision_count=%s collision_hashes=%s",
                trace_id or "",
                driver_id,
                company_user_id,
                len(collision_hashes),
                collision_hashes,
            )
        except Exception:
            pass
        return True
    return False


def log_push_recipient_proof(
    *,
    trace_id: str | None,
    booking_id: int | None,
    status: str | None,
    recipient_role: str,
    recipient_id: int,
    token_count: int,
    token_hashes: list[str],
    collapse_key: str | None = None,
    dedupe_key: str | None = None,
    routing_version: int | None = None,
    routing_decision: str | None = None,
    source: str | None = None,
    actor_role: str | None = None,
    actor_id: int | None = None,
) -> None:
    """Log JSON structuré pour preuve recipient (DEBUG_NOTIF_ROUTING=1)."""
    if not DEBUG_NOTIF_ROUTING:
        return

    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "booking_id": booking_id,
        "status": status,
        "recipient_role": recipient_role,
        "recipient_id": recipient_id,
        "token_count": token_count,
        "token_hashes": token_hashes,
        "collapse_key": collapse_key,
        "dedupe_key": dedupe_key,
        "routing_version": routing_version,
        "routing_decision": routing_decision,
        "source": source,
        "actor_role": actor_role,
        "actor_id": actor_id,
    }
    try:
        import json

        from ext import app_logger

        app_logger.info(
            "[PUSH_RECIPIENT_PROOF] %s",
            json.dumps(
                {k: v for k, v in payload.items() if v is not None}, default=str
            ),
        )
    except Exception:
        pass
