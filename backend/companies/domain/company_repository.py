"""Interface du repository pour Company (port)."""

from __future__ import annotations

from typing import Any, Protocol

from companies.domain.company import Company
from companies.domain.company_id import CompanyId


class CompanyRepository(Protocol):
    """Port (interface) pour le repository de Company.

    L'implémentation sera dans infrastructure/repositories/.
    """

    def save(self, company: Company) -> None:
        """Sauvegarde une entreprise."""
        ...

    def find_by_id(self, company_id: CompanyId) -> Company | None:
        """Trouve une entreprise par ID."""
        ...

    def find_by_user_id(self, user_id: int) -> Company | None:
        """Trouve une entreprise par user_id."""
        ...

    def find_model_by_user_id(self, user_id: int) -> Any | None:
        """Trouve une entreprise SQLAlchemy par user_id (compatibilité)."""
        ...
