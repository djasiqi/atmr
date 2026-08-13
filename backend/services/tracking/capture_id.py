"""P3 — identité stable d'une capture GPS (fix physique).

``capture_id`` est créé une fois côté mobile et ne change jamais au retry.
Aucun UUID aléatoire backend : fallback = ``location_event_id`` déjà fourni.
"""

from __future__ import annotations

from typing import Any


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value).strip()
    return text or None


def extract_raw_capture_id(payload: dict[str, Any] | None) -> str | None:
    """Lit ``capture_id`` / ``captureId`` s'il est fourni par le client."""
    data = payload if isinstance(payload, dict) else {}
    raw = data.get("capture_id")
    if raw is None:
        raw = data.get("captureId")
    return _coerce_optional_str(raw)


def resolve_effective_capture_id(
    payload: dict[str, Any] | None = None,
    *,
    location_event_id: str | None = None,
) -> str | None:
    """Identité effective : capture fournie, sinon location_event_id.

    Ne génère jamais d'UUID côté serveur.
    """
    provided = extract_raw_capture_id(payload)
    if provided:
        return provided
    eid = _coerce_optional_str(location_event_id)
    if eid:
        return eid
    data = payload if isinstance(payload, dict) else {}
    return _coerce_optional_str(
        data.get("location_event_id") or data.get("tracking_event_id")
    )
