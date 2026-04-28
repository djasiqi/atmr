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
class CancelBookingInput:
    """Input pour annuler une réservation.

    Attributes:
        booking: La réservation à annuler
    """

    booking: _BookingLike


@dataclass(frozen=True, slots=True)
class CancelBookingOutput:
    """Output pour annuler une réservation.

    Attributes:
        success: True si l'opération a réussi
        company_id: ID de l'entreprise (si succès)
        should_trigger_dispatch: Si True, déclencher le dispatch (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    company_id: int | None = None
    should_trigger_dispatch: bool = False
    error: dict[str, str] | None = None
    status_code: int | None = None


class CancelBookingUseCase:
    """Use-case Application: annulation d'une réservation (PENDING ou ASSIGNED)."""

    def execute(self, input_data: CancelBookingInput) -> CancelBookingOutput:
        booking = input_data.booking
        status = _status_value(getattr(booking, "status", None)).strip().lower()
        if status not in {"pending", "assigned", "awaiting_client_payment"}:
            return CancelBookingOutput(
                success=False,
                error={
                    "error": (
                        "Seules les réservations en attente ou confirmées "
                        "peuvent être annulées"
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

        return CancelBookingOutput(
            success=True,
            company_id=cid,
            should_trigger_dispatch=cid is not None,
        )
