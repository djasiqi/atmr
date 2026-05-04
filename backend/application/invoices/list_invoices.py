"""Use-case: lister les factures."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from models.enums import InvoiceStatus

logger = logging.getLogger(__name__)


class _InvoiceRepo(Protocol):
    def find_models_by_company_with_filters_query(
        self,
        company_id: int,
        status: InvoiceStatus | None = None,
        client_id: int | None = None,
        year: int | None = None,
        month: int | None = None,
        with_balance: bool = False,
        with_reminders: bool = False,
        search_query: str | None = None,
    ) -> Any: ...  # Returns SQLAlchemy query


@dataclass(frozen=True, slots=True)
class ListInvoicesInput:
    """Input pour lister les factures.

    Attributes:
        company_id: ID de l'entreprise
        status: Statut de la facture (optionnel)
        client_id: ID du client (optionnel)
        year: Année (optionnel)
        month: Mois (optionnel)
        with_balance: Si True, filtre les factures avec solde > 0
        with_reminders: Si True, filtre les factures avec rappels > 0
        search_query: Recherche textuelle (optionnel)
        page: Numéro de page (défaut: 1)
        per_page: Résultats par page (défaut: 100)
    """

    company_id: int
    status: InvoiceStatus | None = None
    client_id: int | None = None
    year: int | None = None
    month: int | None = None
    with_balance: bool = False
    with_reminders: bool = False
    search_query: str | None = None
    page: int = 1
    per_page: int = 100


@dataclass(frozen=True, slots=True)
class ListInvoicesOutput:
    """Output pour lister les factures.

    Attributes:
        success: True si l'opération a réussi
        invoices: Liste des factures
        total: Nombre total de factures
        page: Numéro de page actuel
        per_page: Nombre de résultats par page
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    invoices: list[Any] | None = None  # List of Invoice models
    total: int | None = None
    page: int | None = None
    per_page: int | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class ListInvoicesUseCase:
    """Use-case Application: lister les factures d'une entreprise avec filtres."""

    MAX_PER_PAGE = 100

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, invoice_repo: _InvoiceRepo
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            invoice_repo: Repository pour les factures
        """
        self._invoice_repo = invoice_repo

    def execute(self, input_data: ListInvoicesInput) -> ListInvoicesOutput:
        """Liste les factures d'une entreprise avec filtres et pagination.

        Args:
            input_data: Input avec filtres et pagination

        Returns:
            ListInvoicesOutput avec la liste des factures et le total
        """
        # Validation
        if input_data.page < 1:
            return ListInvoicesOutput(
                success=False,
                error={"page": "Le numéro de page doit être >= 1"},
                status_code=400,
            )

        if input_data.per_page < 1 or input_data.per_page > self.MAX_PER_PAGE:
            return ListInvoicesOutput(
                success=False,
                error={
                    "per_page": (
                        f"Le nombre par page doit être entre 1 et {self.MAX_PER_PAGE}"
                    )
                },
                status_code=400,
            )

        try:
            query = self._invoice_repo.find_models_by_company_with_filters_query(
                company_id=input_data.company_id,
                status=input_data.status,
                client_id=input_data.client_id,
                year=input_data.year,
                month=input_data.month,
                with_balance=input_data.with_balance,
                with_reminders=input_data.with_reminders,
                search_query=input_data.search_query,
            )

            # Compter le total avant pagination
            total = query.count()

            # Appliquer la pagination
            start_idx = (input_data.page - 1) * input_data.per_page
            invoices = query.offset(start_idx).limit(input_data.per_page).all()

            return ListInvoicesOutput(
                success=True,
                invoices=list(invoices),
                total=total,
                page=input_data.page,
                per_page=input_data.per_page,
            )
        except Exception:
            logger.exception("Erreur lors de la liste des factures")
            return ListInvoicesOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
