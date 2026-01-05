from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _CompanyRepo(Protocol):
    def find_all_models_ordered_by_name(self) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class ListCompaniesInput:
    """Input pour lister les entreprises (pas de paramètres pour l'instant)."""

    pass


@dataclass(frozen=True, slots=True)
class ListCompaniesOutput:
    """Output pour lister les entreprises.

    Attributes:
        success: True si l'opération a réussi
        companies: Liste des entreprises
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    companies: list[dict[str, Any]] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class ListCompaniesUseCase:
    """Use-case Application: lister les entreprises (admin)."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, company_repo: _CompanyRepo
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            company_repo: Repository pour les entreprises
        """
        self._company_repo = company_repo

    def execute(self, _input_data: ListCompaniesInput) -> ListCompaniesOutput:
        try:
            models = self._company_repo.find_all_models_ordered_by_name()
            companies: list[dict[str, Any]] = []
            for c in models:
                if hasattr(c, "to_dict") and callable(c.to_dict):
                    d = c.to_dict()
                    if isinstance(d, dict):
                        companies.append(d)
                    else:
                        try:
                            companies.append(dict(d))  # type: ignore[arg-type]
                        except Exception:
                            companies.append(
                                {
                                    "id": getattr(c, "id", None),
                                    "name": getattr(c, "name", None),
                                }
                            )
                elif hasattr(c, "serialize"):
                    ser = c.serialize
                    if isinstance(ser, dict):
                        companies.append(ser)
                    else:
                        companies.append(
                            {
                                "id": getattr(c, "id", None),
                                "name": getattr(c, "name", None),
                            }
                        )
                else:
                    # fallback minimal (évite de crasher si un modèle change)
                    companies.append(
                        {"id": getattr(c, "id", None), "name": getattr(c, "name", None)}
                    )
            return ListCompaniesOutput(success=True, companies=companies)
        except Exception:
            import logging

            logger = logging.getLogger(__name__)
            logger.exception("Erreur lors de la liste des entreprises")
            return ListCompaniesOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
