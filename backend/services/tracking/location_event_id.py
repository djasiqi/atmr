"""Résolution idempotente de ``location_event_id`` pour le pipeline Kafka tracking."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

_MAX_EVENT_ID_LEN = 128
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def normalize_recorded_at_utc_canonical(recorded_at: Any) -> str | None:
    """Normalise ``recorded_at`` en UTC ISO8601 canonique (même instant ⇒ même chaîne)."""
    if recorded_at is None:
        return None
    if isinstance(recorded_at, (int, float)) and not isinstance(recorded_at, bool):
        try:
            ts = float(recorded_at)
            if ts > 1e12:  # millisecondes
                ts = ts / 1000.0
            dt = datetime.fromtimestamp(ts, tz=UTC)
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(recorded_at, str):
        return None
    text = recorded_at.strip()
    if not text:
        return None
    # Remplacer Z / espace pour fromisoformat
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def validate_raw_location_event_id(raw: Any) -> tuple[str | None, str | None]:
    """Valide un identifiant client.

    Returns:
        (id, error_code) — error_code si invalide.
    """
    if raw is None:
        return None, None
    if not isinstance(raw, str):
        return None, "invalid_location_event_id_type"
    stripped = raw.strip()
    if not stripped:
        return None, None
    if len(stripped) > _MAX_EVENT_ID_LEN:
        return None, "location_event_id_too_long"
    if _CONTROL_CHARS.search(stripped):
        return None, "location_event_id_control_chars"
    return stripped, None


def extract_raw_location_event_id(
    *,
    header_value: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Lit l'identifiant brut depuis l'en-tête HTTP ou le corps JSON (sans validation)."""
    raw: Any = header_value
    if raw is None and isinstance(payload, dict):
        for key in ("location_event_id", "tracking_event_id"):
            candidate = payload.get(key)
            if candidate is not None and str(candidate).strip():
                raw = candidate
                break
    if raw is None:
        return None
    if not isinstance(raw, str):
        return str(raw).strip() or None
    stripped = raw.strip()
    return stripped or None


def resolve_location_event_id(
    *,
    driver_id: int,
    latitude: float,
    longitude: float,
    recorded_at: str,
    raw_id: str | None = None,
) -> str:
    """Retourne un event_id client ou un hash déterministe (redelivery Kafka).

    ``recorded_at`` doit déjà être en UTC canonique pour la branche déterministe ;
    une normalisation est appliquée si la chaîne fournie est parsable.
    """
    if raw_id and str(raw_id).strip():
        return str(raw_id).strip()
    canon = normalize_recorded_at_utc_canonical(recorded_at) or str(recorded_at)
    material = f"{driver_id}:{canon}:{latitude:.6f}:{longitude:.6f}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
