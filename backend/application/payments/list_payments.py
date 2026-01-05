"""Use-case: lister les paiements d'un client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain.payment_dto import PaymentDTO


class _PaymentRepo(Protocol):
    def find_by_client_id(self, client_id: int) -> list[PaymentDTO]: ...


@dataclass(frozen=True, slots=True)
class ListPaymentsInput:
    """Input pour lister les paiements d'un client.

    Attributes:
        client_id: ID du client
    """

    client_id: int


@dataclass(frozen=True, slots=True)
class ListPaymentsOutput:
    """Output pour lister les paiements d'un client.

    Attributes:
        success: True si l'opération a réussi
        payments: Liste des paiements
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    payments: list[PaymentDTO] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class ListPaymentsUseCase:
    """Use-case Application: lister les paiements d'un client."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, payment_repo: _PaymentRepo
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            payment_repo: Repository pour les paiements
        """
        self._payment_repo = payment_repo

    def execute(self, input_data: ListPaymentsInput) -> ListPaymentsOutput:
        """Liste les paiements d'un client.

        Args:
            input_data: Input avec client_id

        Returns:
            ListPaymentsOutput avec la liste des paiements
        """
        # Validation
        if input_data.client_id <= 0:
            return ListPaymentsOutput(
                success=False,
                error={"client_id": "L'ID client doit être positif"},
                status_code=400,
            )

        try:
            payments = self._payment_repo.find_by_client_id(input_data.client_id)
            return ListPaymentsOutput(success=True, payments=payments or [])
        except Exception:
            return ListPaymentsOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
