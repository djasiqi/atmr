"""Use-case: générer le PDF d'une facture.

⚠️ TODO: Ce use case encapsule temporairement PDFService pour permettre
une migration progressive. La logique métier devrait être migrée progressivement
vers ce use case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerateInvoicePdfResult:
    ok: bool
    pdf_url: str | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class GenerateInvoicePdfUseCase:
    """Use-case Application: générer le PDF d'une facture.

    ⚠️ TODO: Migration progressive - Ce use case encapsule PDFService.
    La logique métier devrait être migrée progressivement ici.
    """

    def execute(
        self, *, invoice: Any, force_regenerate: bool = False
    ) -> GenerateInvoicePdfResult:
        """Génère le PDF d'une facture.

        Args:
            invoice: La facture pour laquelle générer le PDF
            force_regenerate: Si True, régénère même si un PDF existe déjà

        Returns:
            GenerateInvoicePdfResult avec l'URL du PDF généré
        """
        # ⚠️ TODO: Migrer la logique métier depuis PDFService vers ce use case
        from services.documents.pdf import PDFService

        try:
            pdf_service = PDFService()
            pdf_url = pdf_service.generate_invoice_pdf(
                invoice, force_regenerate=force_regenerate
            )
            if pdf_url:
                return GenerateInvoicePdfResult(ok=True, pdf_url=pdf_url)
            return GenerateInvoicePdfResult(
                ok=False,
                error={"error": "Impossible de générer le PDF"},
                status_code=500,
            )
        except Exception as e:
            return GenerateInvoicePdfResult(
                ok=False,
                error={"error": f"Erreur lors de la génération du PDF: {e!s}"},
                status_code=500,
            )
