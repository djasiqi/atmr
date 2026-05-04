from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from shared.time_utils import parse_local_naive

from ._status import set_status, status_value


class _BookingLike(Protocol):
    id: int | None
    status: Any
    is_return: Any
    parent_booking_id: Any
    scheduled_time: Any
    time_confirmed: Any


@dataclass(frozen=True, slots=True)
class ScheduleCompanyReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False
    trigger_reason: str | None = None


class ScheduleCompanyReservationUseCase:
    """Use-case Application: planification d'une réservation
    (scheduled_time + time_confirmed).

    - Autorisé si statut ∈ {PENDING, ACCEPTED, ASSIGNED}
    - Si booking.is_return : nécessite que la course aller soit complétée
      (info fournie par la route)
    - Si statut PENDING : passe en ACCEPTED pour être éligible au moteur
    """

    _ALLOWED: ClassVar[set[str]] = {"pending", "accepted", "assigned"}

    def execute(
        self,
        booking: _BookingLike,
        *,
        scheduled_time_iso: str,
        is_outbound_completed: bool = True,
    ) -> ScheduleCompanyReservationResult:
        st = status_value(getattr(booking, "status", None)).lower()
        if st not in self._ALLOWED:
            return ScheduleCompanyReservationResult(
                ok=False,
                error={
                    "error": (
                        f"Statut '{status_value(getattr(booking, 'status', None))}' "
                        f"non modifiable."
                    )
                },
                status_code=400,
            )

        if (
            bool(getattr(booking, "is_return", False))
            and getattr(booking, "parent_booking_id", None)
            and not is_outbound_completed
        ):
            return ScheduleCompanyReservationResult(
                ok=False,
                error={
                    "error": (
                        "Impossible de planifier un retour. "
                        "La course aller doit être complétée."
                    )
                },
                status_code=400,
            )

        try:
            sched_local = parse_local_naive(scheduled_time_iso)
        except Exception as e:
            return ScheduleCompanyReservationResult(
                ok=False,
                error={"error": f"Format de date invalide: {e}"},
                status_code=400,
            )

        if sched_local is None:
            return ScheduleCompanyReservationResult(
                ok=False,
                error={"error": "Heure planifiée invalide ou manquante."},
                status_code=400,
            )

        booking.scheduled_time = sched_local
        # Règle métier: 00:00 signifie "heure à confirmer".
        is_sentinel_midnight = (
            sched_local.hour == 0
            and sched_local.minute == 0
            and sched_local.second == 0
        )
        booking.time_confirmed = not is_sentinel_midnight

        if st == "pending":
            set_status(booking, "status", "ACCEPTED")

        return ScheduleCompanyReservationResult(
            ok=True,
            should_trigger_dispatch=True,
            trigger_reason="update",
        )
