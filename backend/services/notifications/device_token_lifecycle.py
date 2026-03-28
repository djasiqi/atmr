"""Mise à jour des colonnes lifecycle sur DeviceToken après envoi push (FCM)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from ext import app_logger, db
from models import DeviceToken


def _lifecycle_enabled() -> bool:
    v = os.environ.get("PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def is_push_device_token_lifecycle_enabled() -> bool:
    """True si colonnes lifecycle + invalidation via apply_push_result_to_device_token."""
    return _lifecycle_enabled()


def apply_push_result_to_device_token(
    device_token_id: int,
    result: dict[str, Any],
    *,
    session: Session | None = None,
) -> None:
    """Applique le résultat FCM sur la ligne DeviceToken (ne commit pas).

    Politique PR4 : désactivation immédiate uniquement si result[\"token_invalid\"] is True.
    Autres erreurs : last_push_failure_at, consecutive_push_failures, last_push_error_code.
    """
    if not _lifecycle_enabled():
        return

    sess = session or db.session
    row = sess.get(DeviceToken, device_token_id)
    if row is None:
        app_logger.warning(
            "[device_token_lifecycle] DeviceToken id=%s introuvable", device_token_id
        )
        return

    now = datetime.now(UTC)

    if result.get("ok"):
        row.last_push_success_at = now
        row.last_push_failure_at = None
        row.consecutive_push_failures = 0
        row.last_push_error_code = None
        return

    code = result.get("error_class") or result.get("error") or "unknown"
    row.last_push_failure_at = now
    row.last_push_error_code = str(code)[:64]
    row.consecutive_push_failures = int(row.consecutive_push_failures or 0) + 1

    if result.get("token_invalid"):
        row.is_active = False
