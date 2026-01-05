"""Use-case: récupérer l'entreprise de l'utilisateur courant."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from companies.domain.company import Company
from companies.domain.company_repository import CompanyRepository

logger = logging.getLogger(__name__)


class GetCurrentUserPort(Protocol):
    """Port pour récupérer l'utilisateur courant."""

    def get_current_user(self) -> Any | None:
        """Récupère l'utilisateur actuellement authentifié."""
        ...


@dataclass(frozen=True, slots=True)
class GetCurrentCompanyResult:
    """Résultat du use-case GetCurrentCompany."""

    company: Company | None
    error: dict[str, str] | None
    status_code: int | None


class GetCurrentCompanyUseCase:
    """Use-case Application: récupérer l'entreprise de l'utilisateur courant.

    Remplace l'appel direct à AuthService.get_current_company() dans les routes.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        get_current_user_port: GetCurrentUserPort,
        company_repo: CompanyRepository,
    ) -> None:
        """Initialise le use-case.

        Args:
            get_current_user_port: Port pour récupérer l'utilisateur courant.
            company_repo: Repository pour les entreprises.
        """
        self.get_current_user_port = get_current_user_port
        self.company_repo = company_repo

    def execute(self) -> GetCurrentCompanyResult:
        """Exécute la récupération de l'entreprise courante.

        Returns:
            GetCurrentCompanyResult avec l'entreprise si trouvée, ou erreur.
        """
        user = self.get_current_user_port.get_current_user()
        if not user:
            return GetCurrentCompanyResult(
                company=None,
                error={"error": "Utilisateur non authentifié"},
                status_code=401,
            )

        # Récupérer l'entreprise via le repository du domaine
        company = self.company_repo.find_by_user_id(user.id)
        if not company:
            return GetCurrentCompanyResult(
                company=None,
                error={"error": "Entreprise non trouvée"},
                status_code=404,
            )

        return GetCurrentCompanyResult(company=company, error=None, status_code=None)
