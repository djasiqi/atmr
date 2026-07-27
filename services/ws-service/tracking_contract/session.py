"""Règles d'ouverture de session tracking."""

from __future__ import annotations

SESSION_STATUSES = frozenset({"active", "superseded", "closed", "expired"})


def first_sequence_id() -> int:
    """Toute nouvelle session commence à sequence_id = 1."""
    return 1
