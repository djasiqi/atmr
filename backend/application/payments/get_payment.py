"""Use-case: récupérer un paiement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain.payment_dto import PaymentDTO


class _PaymentRepo(Protocol):
    def find_by_id(self, payment_id: int) -> PaymentDTO | None: ...


@dataclass(frozen=True, slots=True)
class GetPaymentInput:
    """Input pour récupérer un paiement.

    Attributes:
        payment_id: ID du paiement
    """

    payment_id: int


@dataclass(frozen=True, slots=True)
class GetPaymentOutput:
    """Output pour récupérer un paiement.

    Attributes:
        found: True si le paiement a été trouvé
        payment: Paiement trouvé (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    found: bool
    payment: PaymentDTO | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class GetPaymentUseCase:
    """Use-case Application: récupérer un paiement par ID."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, payment_repo: _PaymentRepo
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            payment_repo: Repository pour les paiements
        """
        self._payment_repo = payment_repo

    def execute(self, input_data: GetPaymentInput) -> GetPaymentOutput:
        """Récupère un paiement par son ID.

        Args:
            input_data: Input avec payment_id

        Returns:
            GetPaymentOutput avec le paiement si trouvé
        """
        payment = self._payment_repo.find_by_id(input_data.payment_id)

        if not payment:
            return GetPaymentOutput(
                found=False,
                error={"error": "Paiement non trouvé"},
                status_code=404,
            )

        return GetPaymentOutput(found=True, payment=payment)
