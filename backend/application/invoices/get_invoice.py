"""Use-case: récupérer une facture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain.invoice_dto import InvoiceDTO


class _InvoiceRepo(Protocol):
    def find_by_id_with_lines(
        self, invoice_id: int, company_id: int
    ) -> InvoiceDTO | None: ...


@dataclass(frozen=True, slots=True)
class GetInvoiceInput:
    """Input pour récupérer une facture.

    Attributes:
        invoice_id: ID de la facture
        company_id: ID de l'entreprise
    """

    invoice_id: int
    company_id: int


@dataclass(frozen=True, slots=True)
class GetInvoiceOutput:
    """Output pour récupérer une facture.

    Attributes:
        found: True si la facture a été trouvée
        invoice: Facture trouvée (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    found: bool
    invoice: InvoiceDTO | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class GetInvoiceUseCase:
    """Use-case Application: récupérer une facture par ID."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, invoice_repo: _InvoiceRepo
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            invoice_repo: Repository pour les factures
        """
        self._invoice_repo = invoice_repo

    def execute(self, input_data: GetInvoiceInput) -> GetInvoiceOutput:
        """Récupère une facture par son ID et company_id.

        Args:
            input_data: Input avec invoice_id et company_id

        Returns:
            GetInvoiceOutput avec la facture si trouvée
        """
        invoice = self._invoice_repo.find_by_id_with_lines(
            input_data.invoice_id, input_data.company_id
        )

        if not invoice:
            return GetInvoiceOutput(
                found=False,
                error={"error": "Facture non trouvée"},
                status_code=404,
            )

        return GetInvoiceOutput(found=True, invoice=invoice)
