from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from shared.time_utils import now_utc

from ._status import set_status, status_value


class _BookingLike(Protocol):
    id: int | None
    status: Any
    is_return: Any
    completed_at: Any


@dataclass(frozen=True, slots=True)
class CompleteCompanyReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class CompleteCompanyReservationUseCase:
    """Use-case Application: complétion d'une réservation
    (IN_PROGRESS -> COMPLETED/RETURN_COMPLETED).

    ✅ Valide automatiquement les transferts ACCEPTED associés lors de la complétion.
    """

    def execute(self, booking: _BookingLike) -> CompleteCompanyReservationResult:
        st = status_value(getattr(booking, "status", None)).lower()
        if st != "in_progress":
            return CompleteCompanyReservationResult(
                ok=False,
                error={"error": "Réservation introuvable ou pas en cours"},
                status_code=400,
            )

        if bool(getattr(booking, "is_return", False)):
            set_status(booking, "status", "RETURN_COMPLETED")
        else:
            set_status(booking, "status", "COMPLETED")

        booking.completed_at = now_utc()

        # ✅ Validation automatique des transferts ACCEPTED
        self._auto_validate_transfers(booking)

        return CompleteCompanyReservationResult(ok=True)

    def _auto_validate_transfers(self, booking: _BookingLike) -> None:
        """Valide automatiquement les transferts ACCEPTED associés à la course.

        Cette méthode est appelée après que la course ait été marquée comme COMPLETED.
        Elle valide tous les transferts avec le statut ACCEPTED pour permettre la facturation.

        Args:
            booking: La course complétée
        """
        try:
            # Import locaux pour éviter les dépendances circulaires
            from models.booking_transfer import BookingTransfer
            from models.enums import TransferStatus

            booking_id = getattr(booking, "id", None)
            if not booking_id:
                return

            # Récupérer tous les transferts ACCEPTED pour cette course
            accepted_transfers = (
                BookingTransfer.query.filter_by(booking_id=booking_id)
                .filter(BookingTransfer.status == TransferStatus.ACCEPTED)
                .all()
            )

            if not accepted_transfers:
                return

            # Valider automatiquement chaque transfert ACCEPTED
            now = now_utc()
            for transfer in accepted_transfers:
                transfer.is_validated = True
                transfer.validated_at = now
                transfer.validated_by = (
                    None  # Validation automatique (pas d'utilisateur spécifique)
                )
                transfer.status = TransferStatus.COMPLETED
                transfer.completed_at = now

        except Exception:
            # En cas d'erreur, ne pas bloquer la complétion de la course
            # La validation automatique est un bonus, pas une obligation
            pass
