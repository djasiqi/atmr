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
    client_id: Any


@dataclass(frozen=True, slots=True)
class AcceptReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False


class AcceptReservationUseCase:
    """Use-case Application: accepter une réservation (PENDING -> ACCEPTED)."""

    def execute(
        self, booking: _BookingLike, *, company_id: int
    ) -> AcceptReservationResult:
        from models import Company
        from models.client import Client
        from services.auth.portal_phone_verification import is_portal_client
        from services.billing.portal_booking_debtor import (
            resolve_portal_booking_debtor_user_id,
        )
        from services.billing.portal_payment_hold import (
            ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
            resolve_portal_payment_hold,
        )

        company = Company.query.get(company_id)
        if not company or not company.is_approved:
            return AcceptReservationResult(
                ok=False,
                error={"error": "Entreprise non approuvée"},
                status_code=403,
            )

        status_str = _status_value(getattr(booking, "status", None))
        if status_str.upper() != "PENDING":
            return AcceptReservationResult(
                ok=False,
                error={"error": "Reservation not found or cannot be accepted"},
                status_code=400,
            )

        client = None
        client_id = getattr(booking, "client_id", None)
        if client_id is not None:
            client = Client.query.get(int(client_id))
        if client is not None and is_portal_client(client):
            debtor_user_id = resolve_portal_booking_debtor_user_id(booking)
            if debtor_user_id is not None:
                hold = resolve_portal_payment_hold(int(debtor_user_id), int(company_id))
                if hold.is_hold:
                    return AcceptReservationResult(
                        ok=False,
                        error={
                            "error": ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
                            "message": (
                                "Ce client a une facture échue auprès de votre "
                                "entreprise. L'acceptation est refusée."
                            ),
                        },
                        status_code=409,
                    )

        booking.company_id = company_id
        _set_status(booking, "accepted")
        return AcceptReservationResult(ok=True, should_trigger_dispatch=True)
