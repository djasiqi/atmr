"""Normalisation de statut booking pour la facturation (évite str(SAEnum))."""

from __future__ import annotations

from typing import Any

_CANCELED = frozenset({"CANCELED", "CANCELLED"})


def normalize_booking_status(booking_or_status: Any) -> str:
    """Valeur métier du statut (CANCELED, COMPLETED, …), jamais ``Class.MEMBER``.

    Accepte un booking (lit ``.status``) ou un statut brut (enum / str).
    Contrat : ``.value`` d'abord, jamais ``str(enum).upper() == "CANCELED"``.
    """
    raw = booking_or_status
    if raw is not None and not isinstance(raw, str):
        status = getattr(raw, "status", None)
        if status is not None:
            raw = status
    if raw is None:
        return ""
    value = getattr(raw, "value", raw)
    return str(value or "").upper().strip()


def booking_status_is_canceled(booking_or_status: Any) -> bool:
    return normalize_booking_status(booking_or_status) in _CANCELED
