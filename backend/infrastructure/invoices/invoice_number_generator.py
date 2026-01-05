"""Service d'infrastructure pour générer les numéros de facture."""

from __future__ import annotations

import logging
from typing import Protocol

from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsDTO,
)
from repositories.invoice_sequence_repository import InvoiceSequenceDTO

logger = logging.getLogger(__name__)


class InvoiceNumberGeneratorPort(Protocol):
    """Port (interface) pour le générateur de numéro de facture."""

    def generate(
        self,
        company_id: int,
        period_year: int,
        period_month: int,
        billing_settings: CompanyBillingSettingsDTO,
        sequence: InvoiceSequenceDTO,
    ) -> str:
        """Génère un numéro de facture unique.

        Args:
            company_id: ID de l'entreprise
            period_year: Année de facturation
            period_month: Mois de facturation (1-12)
            billing_settings: Paramètres de facturation
            sequence: Séquence pour le mois

        Returns:
            Numéro de facture formaté (ex: "INV-2025-01-0001")
        """
        ...


class InvoiceNumberGenerator:
    """Générateur de numéro de facture."""

    def generate(
        self,
        company_id: int,  # noqa: ARG002
        period_year: int,
        period_month: int,
        billing_settings: CompanyBillingSettingsDTO,
        sequence: InvoiceSequenceDTO,
    ) -> str:
        """Génère un numéro de facture unique.

        Args:
            company_id: ID de l'entreprise (non utilisé mais requis par le port)
            period_year: Année de facturation
            period_month: Mois de facturation (1-12)
            billing_settings: Paramètres de facturation
            sequence: Séquence pour le mois

        Returns:
            Numéro de facture formaté (ex: "INV-2025-01-0001")
        """
        return billing_settings.invoice_number_format.format(
            PREFIX=billing_settings.invoice_prefix,
            YYYY=period_year,
            MM=f"{period_month:02d}",
            SEQ4=f"{sequence.sequence:04d}",
        )
