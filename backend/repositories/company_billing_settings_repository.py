"""Repository pour l'accès aux données CompanyBillingSettings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, cast

from models import CompanyBillingSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompanyBillingSettingsDTO:
    """DTO pour CompanyBillingSettings."""

    id: int
    company_id: int
    invoice_prefix: str
    invoice_number_format: str
    vat_applicable: bool
    vat_rate: Decimal | None
    vat_label: str | None
    vat_number: str | None
    payment_terms_days: int


class CompanyBillingSettingsRepositoryPort(Protocol):
    """Port (interface) pour le repository CompanyBillingSettings."""

    def find_or_create(self, company_id: int) -> CompanyBillingSettingsDTO:
        """Trouve ou crée les paramètres de facturation pour une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            CompanyBillingSettingsDTO trouvés ou créés
        """
        ...


class CompanyBillingSettingsRepository:
    """Repository SQLAlchemy pour CompanyBillingSettings."""

    def _to_dto(self, settings: CompanyBillingSettings) -> CompanyBillingSettingsDTO:
        """Convertit un modèle SQLAlchemy CompanyBillingSettings en DTO."""
        return CompanyBillingSettingsDTO(
            id=settings.id,
            company_id=settings.company_id,
            invoice_prefix=cast(str, settings.invoice_prefix),
            invoice_number_format=cast(str, settings.invoice_number_format),
            vat_applicable=bool(settings.vat_applicable),
            vat_rate=settings.vat_rate,
            vat_label=settings.vat_label,
            vat_number=settings.vat_number,
            payment_terms_days=settings.payment_terms_days or 30,
        )

    def find_or_create(self, company_id: int) -> CompanyBillingSettingsDTO:
        """Trouve ou crée les paramètres de facturation pour une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            CompanyBillingSettingsDTO trouvés ou créés

        Side-effects:
            - DB: Crée CompanyBillingSettings si inexistants et commit
        """
        from ext import db

        settings = CompanyBillingSettings.query.filter_by(company_id=company_id).first()

        if not settings:
            settings = CompanyBillingSettings()
            settings.company_id = company_id
            db.session.add(settings)
            db.session.commit()

        return self._to_dto(settings)
