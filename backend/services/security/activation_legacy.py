"""Compatibilité bornée des jetons itsdangerous pré-Lot 1 (F-03)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_FROM_ENV = "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_FROM_UTC"
_UNTIL_ENV = "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_UNTIL_UTC"
_MAX_WINDOW = timedelta(minutes=35)


class ActivationLegacyConfigError(RuntimeError):
    """Configuration legacy invalide (boot)."""


def _parse_utc_aware(raw: str, *, env_name: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ActivationLegacyConfigError(
            f"{env_name} invalide (ISO-8601 UTC attendu): {raw!r}"
        ) from exc
    if dt.tzinfo is None:
        raise ActivationLegacyConfigError(
            f"{env_name} doit être timezone-aware UTC (pas de datetime naïf): {raw!r}"
        )
    dt_utc = dt.astimezone(UTC)
    # Exiger offset UTC exact (pas seulement convertible)
    if dt.utcoffset() != timedelta(0):
        raise ActivationLegacyConfigError(
            f"{env_name} doit être en UTC (offset +00:00), reçu: {raw!r}"
        )
    return dt_utc


def get_legacy_acceptance_window() -> tuple[datetime, datetime] | None:
    """Retourne (from, until) si legacy activé, sinon None.

    Raises:
        ActivationLegacyConfigError: config partielle / invalide.
    """
    from_raw = (os.getenv(_FROM_ENV) or "").strip()
    until_raw = (os.getenv(_UNTIL_ENV) or "").strip()
    if not from_raw and not until_raw:
        return None
    if bool(from_raw) != bool(until_raw):
        raise ActivationLegacyConfigError(
            f"{_FROM_ENV} et {_UNTIL_ENV} doivent être tous deux vides "
            "ou tous deux renseignés."
        )
    start = _parse_utc_aware(from_raw, env_name=_FROM_ENV)
    end = _parse_utc_aware(until_raw, env_name=_UNTIL_ENV)
    if end <= start:
        raise ActivationLegacyConfigError(
            f"{_UNTIL_ENV} doit être strictement postérieur à {_FROM_ENV}."
        )
    if end - start > _MAX_WINDOW:
        raise ActivationLegacyConfigError(
            "Fenêtre legacy activation > 35 minutes interdite "
            f"(FROM={from_raw}, UNTIL={until_raw})."
        )
    return start, end


def validate_activation_legacy_for_boot(*, config_name: str) -> None:
    """Valide la config legacy au démarrage (production stricte)."""
    try:
        window = get_legacy_acceptance_window()
    except ActivationLegacyConfigError:
        if config_name == "production":
            raise
        logger.warning(
            "[activation_legacy] config invalide hors production — legacy désactivé"
        )
        return
    if window is None:
        logger.info("[activation_legacy] désactivé (FROM/UNTIL vides)")
        return
    start, end = window
    logger.warning(
        "[activation_legacy] activé from=%s until=%s",
        start.isoformat(),
        end.isoformat(),
    )


def is_legacy_acceptance_active(*, now: datetime | None = None) -> bool:
    """True si now est dans la fenêtre absolue (False si désactivé / invalide)."""
    try:
        window = get_legacy_acceptance_window()
    except ActivationLegacyConfigError:
        return False
    if window is None:
        return False
    start, end = window
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    else:
        current = current.astimezone(UTC)
    return start <= current < end


def legacy_window_status() -> dict[str, Any]:
    """Diag ops (sans secrets)."""
    try:
        window = get_legacy_acceptance_window()
    except ActivationLegacyConfigError as exc:
        return {"enabled": False, "error": str(exc)}
    if window is None:
        return {"enabled": False}
    start, end = window
    return {
        "enabled": True,
        "from": start.isoformat(),
        "until": end.isoformat(),
        "active_now": is_legacy_acceptance_active(),
    }
