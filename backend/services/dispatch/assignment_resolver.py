"""Resolver unique de l'Assignment « courant » d'un booking (P0-B).

Invariant MISSION-STATE : la lecture (surface chauffeur composée, détail) et
l'écriture (sync des transitions chauffeur, release, ensure) doivent cibler la
MÊME ligne Assignment. Sans cela, un ARRIVED peut être écrit sur une vieille
ligne jamais relue, et la surface chauffeur régresse.

Le « courant » = l'assignment le plus récent, tie-break (created_at, id) —
identique en SQL et en mémoire.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Iterable

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _sort_key(assignment: Any) -> tuple[datetime, int]:
    created = getattr(assignment, "created_at", None)
    if created is None:
        created_dt = _EPOCH
    elif isinstance(created, datetime):
        created_dt = (
            created if created.tzinfo is not None else created.replace(tzinfo=UTC)
        )
    else:
        # Tests / stubs : entier ou ISO — on ne plante pas, on classe en dernier.
        created_dt = _EPOCH
        if isinstance(created, (int, float)):
            try:
                created_dt = datetime.fromtimestamp(float(created), tz=UTC)
            except (OverflowError, OSError, ValueError):
                created_dt = _EPOCH
    try:
        assignment_id = int(getattr(assignment, "id", 0) or 0)
    except (TypeError, ValueError):
        assignment_id = 0
    return (created_dt, assignment_id)


def pick_current_assignment(assignments: Iterable[Any]) -> Any | None:
    """Sélection en mémoire de l'assignment courant (même règle que le SQL)."""
    current: Any | None = None
    current_key: tuple[datetime, int] | None = None
    for assignment in assignments:
        if assignment is None:
            continue
        key = _sort_key(assignment)
        if current_key is None or key > current_key:
            current = assignment
            current_key = key
    return current


def resolve_current_assignment_for_booking(booking_id: int) -> Any | None:
    """Requête SQL de l'assignment courant — SEULE définition côté DB."""
    from models import Assignment

    return (
        Assignment.query.filter_by(booking_id=booking_id)
        .order_by(
            Assignment.created_at.desc().nullslast(),
            Assignment.id.desc(),
        )
        .first()
    )
