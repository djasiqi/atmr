from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from domain.client_dto import ClientDTO


class _ClientRepo(Protocol):
    def find_by_company_with_user_and_search(
        self, company_id: int, search: str | None = None
    ) -> list[ClientDTO]: ...

    def find_models_by_company_with_user_and_search_paginated(
        self,
        company_id: int,
        search: str | None,
        page: int,
        per_page: int,
        sort_by: str | None = None,
        sort_order: str | None = None,
    ) -> tuple[list[Any], int]: ...


#: Tris exposés côté API (whitelist — Lot 5 perf, pas de tri arbitraire).
ALLOWED_CLIENT_SORT_FIELDS = ("name", "created")
ALLOWED_CLIENT_SORT_ORDERS = ("asc", "desc")


@dataclass(frozen=True, slots=True)
class ListCompanyClientsInput:
    """Input pour lister les clients d'une entreprise.

    Attributes:
        company_id: ID de l'entreprise
        search: Recherche textuelle (optionnel)
        page: Numéro de page (commence à 1)
        per_page: Nombre de résultats par page
        sort_by: Champ de tri whitelisté (``name`` | ``created``)
        sort_order: Ordre whitelisté (``asc`` | ``desc``)
    """

    company_id: int
    search: str | None = None
    page: int = 1
    per_page: int = 20
    sort_by: str | None = None
    sort_order: str | None = None


@dataclass(frozen=True, slots=True)
class ListCompanyClientsOutput:
    """Output pour lister les clients d'une entreprise.

    Attributes:
        success: True si l'opération a réussi
        clients: Liste des clients
        total: Nombre total de clients
        page: Numéro de page actuel
        per_page: Nombre de résultats par page
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    clients: list[dict[str, Any]] | None = None
    total: int | None = None
    page: int | None = None
    per_page: int | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class ListCompanyClientsUseCase:
    """Use-case Application: liste + search + pagination (slice)
    pour les clients d'une company."""

    MAX_PER_PAGE = 100  # ✅ Aligné avec la validation de la route backend (Lot 5 perf : max 100)

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, client_repo: _ClientRepo
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            client_repo: Repository pour les clients
        """
        self._client_repo = client_repo

    def execute(self, input_data: ListCompanyClientsInput) -> ListCompanyClientsOutput:
        # Validation
        if input_data.page < 1:
            return ListCompanyClientsOutput(
                success=False,
                error={"page": "Le numéro de page doit être >= 1"},
                status_code=400,
            )

        if input_data.per_page < 1 or input_data.per_page > self.MAX_PER_PAGE:
            return ListCompanyClientsOutput(
                success=False,
                error={
                    "per_page": (
                        f"Le nombre par page doit être entre 1 et {self.MAX_PER_PAGE}"
                    )
                },
                status_code=400,
            )

        try:
            page = max(input_data.page, 1)
            per_page = max(input_data.per_page, 1)

            q = (input_data.search or "").strip()
            sort_by = (
                input_data.sort_by
                if input_data.sort_by in ALLOWED_CLIENT_SORT_FIELDS
                else "name"
            )
            sort_order = (
                input_data.sort_order
                if input_data.sort_order in ALLOWED_CLIENT_SORT_ORDERS
                else "asc"
            )
            page_clients, total = (
                self._client_repo.find_models_by_company_with_user_and_search_paginated(
                    input_data.company_id,
                    q if q else None,
                    page,
                    per_page,
                    sort_by,
                    sort_order,
                )
            )

            # Utiliser serialize au lieu de to_dict() pour inclure default_billing et toutes les relations
            serialized: list[dict[str, Any]] = [c.serialize for c in page_clients]

            return ListCompanyClientsOutput(
                success=True,
                clients=serialized,
                total=total,
                page=page,
                per_page=per_page,
            )
        except Exception:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception("Erreur lors de la liste des clients")
            return ListCompanyClientsOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
