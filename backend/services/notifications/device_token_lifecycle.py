"""Mise à jour des colonnes lifecycle sur DeviceToken après envoi push (FCM)."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ext import app_logger, db
from models import DeviceToken

STALE_TOKEN_DAYS = int(os.getenv("PUSH_DEVICE_TOKEN_STALE_DAYS", "30"))
STALE_TOKEN_MIN_FAILURES = int(os.getenv("PUSH_DEVICE_TOKEN_MIN_FAILURES", "5"))


def _lifecycle_enabled() -> bool:
    v = os.environ.get("PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def is_push_device_token_lifecycle_enabled() -> bool:
    """True si colonnes lifecycle + invalidation via apply_push_result_to_device_token."""
    return _lifecycle_enabled()


def last_failure_reason(row: DeviceToken) -> str | None:
    """Raison lisible de la dernière défaillance push (alias last_push_error_code)."""
    return row.last_push_error_code


def apply_push_result_to_device_token(
    device_token_id: int,
    result: dict[str, Any],
    *,
    session: Session | None = None,
) -> None:
    """Applique le résultat FCM sur la ligne DeviceToken (ne commit pas).

    Politique :
    - désactivation immédiate si token_invalid confirmé (pas configuration_error) ;
    - ``SENDER_ID_MISMATCH`` / configuration_error → jamais de désactivation auto.
    """
    sess = session or db.session

    is_config_error = bool(
        result.get("configuration_error")
        or result.get("error") == "sender_id_mismatch"
        or result.get("delivery_status") == "configuration_error"
    )
    # token_invalid uniquement si confirmé et pas une erreur de configuration
    should_deactivate = bool(result.get("token_invalid")) and not is_config_error
    if result.get("deactivate_token") is False:
        should_deactivate = False
    elif result.get("deactivate_token") is True and not is_config_error:
        should_deactivate = True

    if not _lifecycle_enabled():
        if should_deactivate:
            row = sess.get(DeviceToken, device_token_id)
            if row is not None:
                row.is_active = False
        return

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
    if is_config_error:
        code = f"configuration_error:{code}"
        app_logger.error(
            "[device_token_lifecycle] configuration_error id=%s code=%s "
            "(token conservé actif)",
            device_token_id,
            code,
        )

    row.last_push_failure_at = now
    row.last_push_error_code = str(code)[:64]
    row.consecutive_push_failures = int(row.consecutive_push_failures or 0) + 1

    if should_deactivate:
        row.is_active = False
        app_logger.info(
            "[device_token_lifecycle] token deactivated id=%s reason=%s failures=%s",
            device_token_id,
            row.last_push_error_code,
            row.consecutive_push_failures,
        )
    elif is_config_error:
        # Ne pas compter comme stale pour désactivation
        pass
    elif row.consecutive_push_failures >= STALE_TOKEN_MIN_FAILURES:
        cutoff = now - timedelta(days=STALE_TOKEN_DAYS)
        last_ok = row.last_push_success_at
        is_stale = last_ok is None or (
            isinstance(last_ok, datetime) and last_ok < cutoff
        )
        if is_stale:
            row.is_active = False
            app_logger.info(
                "[device_token_lifecycle] stale token deactivated id=%s "
                "last_success=%s failures=%s reason=%s",
                device_token_id,
                last_ok,
                row.consecutive_push_failures,
                row.last_push_error_code,
            )


def deactivate_stale_device_tokens(
    *,
    session: Session | None = None,  # noqa: ARG001 — réservé API (session injectée)
    stale_days: int | None = None,
    min_failures: int | None = None,
) -> int:
    """Désactive les tokens zombies (anciens + échecs répétés). Retourne le nombre désactivés."""
    days = stale_days if stale_days is not None else STALE_TOKEN_DAYS
    failures = min_failures if min_failures is not None else STALE_TOKEN_MIN_FAILURES
    cutoff = datetime.now(UTC) - timedelta(days=days)

    q = DeviceToken.query.filter(
        DeviceToken.is_active.is_(True),
        DeviceToken.consecutive_push_failures >= failures,
    )
    deactivated = 0
    for row in q.all():
        last_ok = row.last_push_success_at
        if last_ok is None or last_ok < cutoff:
            row.is_active = False
            deactivated += 1
            app_logger.info(
                "[device_token_lifecycle] cron stale deactivate id=%s reason=%s",
                row.id,
                row.last_push_error_code,
            )
    return deactivated
