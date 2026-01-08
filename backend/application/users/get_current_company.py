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
            # ✅ Vérifier d'abord le claim company_id du token JWT
            # (priorité pour les switchs)
            # Cela permet aux tokens créés lors d'un switch
            # (driver <-> company) de fonctionner
            # même si l'utilisateur dans la DB a un user_id différent
            company = None
            try:
                from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
                    get_jwt,
                )

                jwt_data = get_jwt()
                if jwt_data and "company_id" in jwt_data:
                    company_id = jwt_data["company_id"]
                    if company_id:
                        company = self.company_repo.find_model_by_id(int(company_id))
            except Exception:
                # Si on ne peut pas récupérer le claim, continuer avec
                # la recherche par user_id
                pass

            # Si le claim du token n'a pas fonctionné, utiliser la méthode
            # classique (recherche par user_id)
            if not company:
                # 1. Récupérer l'utilisateur authentifié
                user = self.get_current_user_fn()
                if not user:
                    return GetCurrentCompanyResult(
                        found=False,
                        error={"error": "Utilisateur non authentifié"},
                        status_code=401,
                    )

                # 2. Trouver l'entreprise associée à l'utilisateur
                company = self.company_repo.find_model_by_user_id(user.id)

            if not company:
                return GetCurrentCompanyResult(
                    found=False,
                    error={"error": "Entreprise non trouvée"},
                    status_code=404,
                )

            return GetCurrentCompanyResult(found=True, company=company)
        except Exception as e:
            return GetCurrentCompanyResult(
                found=False,
                error={"error": f"Erreur lors de la récupération: {e!s}"},
                status_code=500,
            )
