from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _status_value(status: Any) -> str:
    if status is None:
        return ""
    v = getattr(status, "value", None)
    if isinstance(v, str):
        return v
    return str(status)


def _set_status(obj: Any, status_str: str) -> None:
    current = getattr(obj, "status", None)
    enum_cls = getattr(current, "__class__", None)
    candidate_name = status_str.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate_name):
        obj.status = getattr(enum_cls, candidate_name)
        return
    obj.status = status_str


class _BookingLike(Protocol):
    id: int | None
    status: Any
    company_id: Any


@dataclass(frozen=True, slots=True)
class CancelBookingResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    company_id: int | None = None
    should_trigger_dispatch: bool = False


class CancelBookingUseCase:
    """Use-case Application: annulation d'une réservation (PENDING ou ASSIGNED)."""

    def execute(self, booking: _BookingLike) -> CancelBookingResult:
        status = _status_value(getattr(booking, "status", None))
        # Convertir en minuscules pour la comparaison (les enums retournent souvent en majuscules)
        status_lower = status.lower() if status else ""
        if status_lower not in {"pending", "assigned"}:
            return CancelBookingResult(
                ok=False,
                error={
                    "error": (
                        "Seules les réservations en attente ou confirmées peuvent être annulées"
                    )
                },
                status_code=400,
            )

        _set_status(booking, "canceled")

        cid_obj = getattr(booking, "company_id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None

        return CancelBookingResult(
            ok=True,
            company_id=cid,
            should_trigger_dispatch=cid is not None,
        )
