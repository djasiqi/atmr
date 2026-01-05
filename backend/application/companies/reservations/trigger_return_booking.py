from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from shared.time_utils import now_utc, to_utc

from ._status import set_status


class _BookingLike(Protocol):
    id: int | None
    is_return: Any
    is_round_trip: Any
    parent_booking_id: Any
    customer_name: Any
    pickup_location: Any
    dropoff_location: Any
    amount: Any
    user_id: Any
    client_id: Any
    company_id: Any
    scheduled_time: Any
    status: Any
    booking_type: Any


@dataclass(frozen=True, slots=True)
class TriggerReturnBookingDecision:
    action: str  # "modify_current" | "modify_existing_return" | "create_new"
    return_time: datetime
    should_trigger_dispatch: bool
    trigger_reason: str


@dataclass(frozen=True, slots=True)
class TriggerReturnBookingResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    decision: TriggerReturnBookingDecision | None = None


class TriggerReturnBookingUseCase:
    """Use-case Application: créer/MAJ une réservation retour.

    La route est responsable de:
    - charger l'outbound booking
    - charger un return booking existant (si présent)
    - créer l'instance Booking() si nécessaire
    - commit + dispatch trigger
    """

    def execute(
        self,
        outbound: _BookingLike,
        *,
        return_time_raw: str | None,
        urgent: bool,
        minutes_offset: int,
        now: datetime | None = None,
        has_existing_return: bool,
    ) -> TriggerReturnBookingResult:
        current_now = now or now_utc()

        if urgent or not return_time_raw:
            return_time = current_now + timedelta(minutes=int(minutes_offset))
        else:
            try:
                dt_utc = to_utc(return_time_raw)
            except Exception as e:
                return TriggerReturnBookingResult(
                    ok=False,
                    error={"error": f"Format de date invalide : {e}"},
                    status_code=400,
                )
            if dt_utc is None:
                return TriggerReturnBookingResult(
                    ok=False,
                    error={"error": "Format de date invalide"},
                    status_code=400,
                )
            return_time = dt_utc

        if return_time <= current_now:
            return TriggerReturnBookingResult(
                ok=False,
                error={"error": "L'heure de retour doit être dans le futur."},
                status_code=400,
            )

        if bool(getattr(outbound, "is_return", False)):
            action = "modify_current"
        elif has_existing_return:
            action = "modify_existing_return"
        else:
            action = "create_new"

        # On marque toujours le retour en ACCEPTED pour entrer dans le moteur
        # (la route appliquera le statut sur la booking concernée)
        _ = set_status  # silence unused import in case of future edits

        return TriggerReturnBookingResult(
            ok=True,
            decision=TriggerReturnBookingDecision(
                action=action,
                return_time=return_time,
                should_trigger_dispatch=True,
                trigger_reason="return_request",
            ),
        )
