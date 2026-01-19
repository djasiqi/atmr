from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ._status import set_status, status_value


class _BookingLike(Protocol):
    id: int | None
    status: Any
    scheduled_time: datetime | None
    driver_id: Any


@dataclass(frozen=True, slots=True)
class DeleteOrCancelCompanyReservationResult:
    ok: bool
    action: str | None = None  # "delete" | "cancel"
    message: str | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False
    trigger_reason: str | None = None
    should_delete_assignments: bool = False


class DeleteOrCancelCompanyReservationUseCase:
    """Use-case Application: suppression/annulation intelligente selon
    statut + timing."""

    def execute(
        self,
        booking: _BookingLike,
        *,
        now_utc: datetime | None = None,
        hours_offset: float = -24.0,
    ) -> DeleteOrCancelCompanyReservationResult:
        now = now_utc or datetime.now(UTC)

        scheduled = getattr(booking, "scheduled_time", None)
        if scheduled and scheduled.tzinfo is None:
            # Si naïf: on le traite comme UTC pour l'algorithme
            # (la route peut normaliser si besoin)
            scheduled = scheduled.replace(tzinfo=UTC)

        time_diff_hours = (
            ((scheduled - now).total_seconds() / 3600.0) if scheduled else 0.0
        )

        st = status_value(getattr(booking, "status", None)).lower()

        # Règle 1: PENDING/ACCEPTED → delete (physique)
        if st in {"pending", "accepted"}:
            return DeleteOrCancelCompanyReservationResult(
                ok=True,
                action="delete",
                message="La réservation a été supprimée avec succès.",
                should_trigger_dispatch=True,
                trigger_reason="cancel",
            )

        # Règle 2: ASSIGNED → timing
        if st == "assigned":
            if time_diff_hours < hours_offset:
                return DeleteOrCancelCompanyReservationResult(
                    ok=True,
                    action="delete",
                    message="La réservation a été supprimée avec succès.",
                    should_delete_assignments=True,
                    should_trigger_dispatch=True,
                    trigger_reason="cancel",
                )
            # sinon: cancel + libérer driver
            # ✅ CORRECTION : Libérer le driver AVANT de changer le statut
            # pour éviter l'erreur de validation "driver_id ne peut pas être NULL si status=ASSIGNED"
            # (même si on change vers CANCELED, la validation peut se déclencher entre les deux opérations)
            if getattr(booking, "driver_id", None):
                booking.driver_id = None
            # Maintenant on peut changer le statut en toute sécurité
            set_status(booking, "status", "CANCELED")
            return DeleteOrCancelCompanyReservationResult(
                ok=True,
                action="cancel",
                message="La réservation a été annulée avec succès.",
                should_trigger_dispatch=True,
                trigger_reason="cancel",
            )

        # Règle 3: forbid
        msg_map = {
            "in_progress": "La course est en cours et ne peut pas être annulée.",
            "completed": "La course est terminée et ne peut pas être modifiée.",
            "canceled": "La course est déjà annulée.",
            "return_completed": "La course est terminée et ne peut pas être modifiée.",
        }
        msg = msg_map.get(st) or (
            "Impossible de supprimer/annuler une course avec le statut "
            + f"'{status_value(getattr(booking, 'status', None))}'."
        )
        return DeleteOrCancelCompanyReservationResult(
            ok=False,
            error={"error": msg},
            status_code=403,
        )
