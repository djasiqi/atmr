"""Use-case: mettre à jour le statut d'un paiement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from domain.payment_dto import PaymentDTO
from models.enums import PaymentStatus

logger = logging.getLogger(__name__)


class _PaymentRepositoryPort(Protocol):
    """Port pour le repository Payment."""

    def find_by_id(self, payment_id: int) -> PaymentDTO | None:
        """Trouve un paiement par son ID."""
        ...

    def update_status(
        self, payment_id: int, status: PaymentStatus
    ) -> PaymentDTO | None:
        """Met à jour le statut d'un paiement."""
        ...


@dataclass(frozen=True, slots=True)
class UpdatePaymentStatusInput:
    """Input pour mettre à jour le statut d'un paiement.

    Attributes:
        payment_id: ID du paiement à mettre à jour
        status: Le nouveau statut
    """

    payment_id: int
    status: PaymentStatus


@dataclass(frozen=True, slots=True)
class UpdatePaymentStatusOutput:
    """Output pour mettre à jour le statut d'un paiement.

    Attributes:
        success: True si l'opération a réussi
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class UpdatePaymentStatusUseCase:
    """Use-case Application: mettre à jour le statut d'un paiement (admin uniquement)."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, payment_repo: _PaymentRepositoryPort
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            payment_repo: Repository pour les paiements
        """
        self._payment_repo = payment_repo

    def execute(
        self, input_data: UpdatePaymentStatusInput
    ) -> UpdatePaymentStatusOutput:
        """Met à jour le statut d'un paiement.

        Args:
            input_data: Input avec payment_id et status

        Returns:
            UpdatePaymentStatusOutput avec le résultat de l'opération

        Side-effects:
            - DB: Met à jour Payment.status
            - DB: Commit transaction (ou rollback en cas d'erreur)
        """
        # Validation
        if input_data.payment_id <= 0:
            return UpdatePaymentStatusOutput(
                success=False,
                error={"payment_id": "L'ID paiement doit être positif"},
                status_code=400,
            )

        try:
            updated_payment = self._payment_repo.update_status(
                input_data.payment_id, input_data.status
            )

            if updated_payment is None:
                return UpdatePaymentStatusOutput(
                    success=False,
                    error={"error": "Paiement non trouvé"},
                    status_code=404,
                )

            return UpdatePaymentStatusOutput(success=True)
        except Exception:
            logger.exception("Erreur lors de la mise à jour du statut du paiement")
            return UpdatePaymentStatusOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
