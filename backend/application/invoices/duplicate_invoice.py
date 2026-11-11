"""Use-case: dupliquer une facture.

Ce use case migre la logique métier de InvoiceService.duplicate_invoice()
vers l'architecture DDD.
"""

from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass
from typing import Any

from models import Invoice, InvoiceStatus
from application.invoices.cancel_invoice import CancelInvoiceUseCase
from repositories.booking_repository import BookingRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DuplicateInvoiceInput:
    """Input pour dupliquer une facture.

    Attributes:
        invoice: La facture à dupliquer (modèle SQLAlchemy)
    """

    invoice: Invoice


@dataclass(frozen=True, slots=True)
class DuplicateInvoiceOutput:
    """Output pour dupliquer une facture.

    Attributes:
        success: True si l'opération a réussi
        draft_context: Contexte pour pré-remplir le formulaire (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    draft_context: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class DuplicateInvoiceUseCase:
    """Use-case Application: dupliquer une facture (créer un brouillon correctif).

    Ce use case migre la logique métier de InvoiceService.duplicate_invoice()
    vers l'architecture DDD.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        cancel_invoice_use_case: CancelInvoiceUseCase | None = None,
        booking_repo: BookingRepository | None = None,
    ):
        """Initialise le use case avec injection de dépendances.

        Args:
            cancel_invoice_use_case: Use case pour annuler une facture
            booking_repo: Repository pour les réservations
                (utilisé par CancelInvoiceUseCase)
        """
        booking_repo = booking_repo or BookingRepository()
        self.cancel_invoice_use_case = cancel_invoice_use_case or CancelInvoiceUseCase(
            booking_repo=booking_repo
        )

    def execute(self, input_data: DuplicateInvoiceInput) -> DuplicateInvoiceOutput:
        """Refuse la duplication d'une facture émise.

        Dupliquer annulait la facture d'origine avec ``force`` et libérait
        ses courses. Une facture émise reste en place. Le brouillon se
        modifie directement. L'avoir viendra dans un lot séparé.
        """
        invoice = input_data.invoice

        try:
            if invoice.status == InvoiceStatus.DRAFT:
                msg = (
                    "La facture est déjà un brouillon et peut être modifiée "
                    "directement."
                )
                raise ValueError(msg)
            from application.invoices.issued_invoice_guard import invoice_is_issued

            if invoice_is_issued(invoice):
                return DuplicateInvoiceOutput(
                    success=False,
                    error={
                        "error": (
                            "Une facture émise ne s'annule pas par duplication."
                        ),
                        "error_code": "ISSUED_INVOICE_IMMUTABLE",
                    },
                    status_code=409,
                )
            return DuplicateInvoiceOutput(
                success=False,
                error={"error": "Statut de facture invalide"},
                status_code=400,
            )

        except ValueError as e:
            logger.warning(
                "Erreur de validation lors de la duplication de facture: %s", e
            )
            return DuplicateInvoiceOutput(
                success=False,
                error={"error": str(e)},
                status_code=400,
            )
        except Exception:
            logger.exception("Erreur inattendue lors de la duplication de la facture")
            return DuplicateInvoiceOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
