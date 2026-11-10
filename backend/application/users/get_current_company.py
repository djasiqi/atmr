"""Use-case: récupérer l'entreprise de l'utilisateur courant.

Ce use case récupère l'utilisateur authentifié via GetCurrentUserUseCase
puis trouve l'entreprise associée via CompanyRepository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from repositories.company_repository import CompanyRepository
from shared.infrastructure.adapters.auth_adapter import get_current_user_via_use_case


@dataclass(frozen=True, slots=True)
class GetCurrentCompanyResult:
    found: bool
    company: Any | None = None  # Company model
    error: dict[str, str] | None = None
    status_code: int | None = None


class GetCurrentCompanyUseCase:
    """Use-case Application: récupérer l'entreprise de l'utilisateur courant.

    Ce use case utilise GetCurrentUserUseCase pour obtenir l'utilisateur authentifié,
    puis CompanyRepository pour trouver l'entreprise associée.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        company_repo: CompanyRepository | None = None,
        get_current_user_fn: Any | None = None,
    ) -> None:
        """Initialise le use case.

        Args:
            company_repo: Repository pour les entreprises
                (créé par défaut si None)
            get_current_user_fn: Fonction pour obtenir l'utilisateur courant
                (par défaut: get_current_user_via_use_case)
        """
        self.company_repo = company_repo or CompanyRepository()
        self.get_current_user_fn = get_current_user_fn or get_current_user_via_use_case

    def execute(self) -> GetCurrentCompanyResult:
        """Récupère l'entreprise de l'utilisateur actuellement authentifié.

        Returns:
            GetCurrentCompanyResult avec l'entreprise si trouvée
        """
        try:
            user = self.get_current_user_fn()
            if not user:
                return GetCurrentCompanyResult(
                    found=False,
                    error={"error": "Utilisateur non authentifié"},
                    status_code=401,
                )

            # Compte ADMIN plateforme : jamais d'espace entreprise
            role = getattr(user, "role", None)
            role_value = getattr(role, "value", role)
            if str(role_value or "").upper() == "ADMIN":
                return GetCurrentCompanyResult(
                    found=False,
                    error={
                        "error": "Compte administrateur plateforme — "
                        "accès entreprise refusé"
                    },
                    status_code=403,
                )

            from security.portal_scope import resolve_portal_scope

            scope = resolve_portal_scope(user, "company")
            if scope.allowed and scope.company is not None:
                return GetCurrentCompanyResult(found=True, company=scope.company)
            if scope.reason in {
                "cross_tenant",
                "revoked",
                "ambiguous_company",
                "pending",
                "unapproved",
                "ineligible",
            }:
                return GetCurrentCompanyResult(
                    found=False,
                    error={"error": "Accès non autorisé"},
                    status_code=403,
                )
            return GetCurrentCompanyResult(
                found=False,
                error={"error": "Entreprise non trouvée"},
                status_code=404,
            )
        except Exception:
            return GetCurrentCompanyResult(
                found=False,
                error={"error": "Erreur lors de la récupération"},
                status_code=500,
            )
