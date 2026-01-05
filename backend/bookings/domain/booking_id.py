"""Value Object : ID de réservation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BookingId:
    """Value Object : ID de réservation."""

    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("BookingId must be positive")
