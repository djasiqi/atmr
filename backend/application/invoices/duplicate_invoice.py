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
        """Prépare un brouillon correctif à partir d'une facture existante.

        Cette méthode annule la facture originale (en libérant les trajets) et renvoie
        le contexte nécessaire côté frontend pour pré-remplir le formulaire de création.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            DuplicateInvoiceOutput avec le contexte du brouillon créé
        """
        invoice = input_data.invoice

        try:
            # 1. Vérifier que la facture n'est pas déjà un brouillon
            if invoice.status == InvoiceStatus.DRAFT:
                msg = (
                    "La facture est déjà un brouillon et peut être modifiée "
                    "directement."
                )
                raise ValueError(msg)

            # 2. Récupérer les lignes avec réservations
            reservation_lines = [line for line in invoice.lines if line.reservation_id]
            if not reservation_lines:
                msg = "Aucune course liée à cette facture ne peut être dupliquée."
                raise ValueError(msg)

            # 3. Construire les overrides à partir des lignes
            bill_to_client_id: int | None = invoice.bill_to_client_id
            billing_type = "third_party" if bill_to_client_id else "direct"
            reservation_ids: list[int] = []
            overrides: dict[str, dict[str, Any]] = {}

            for line in reservation_lines:
                if not line.reservation_id:
                    continue

                reservation_ids.append(line.reservation_id)
                override: dict[str, Any] = {
                    "amount": float(line.line_total),
                }
                if line.vat_rate is not None:
                    override["vat_rate"] = float(line.vat_rate)
                if line.adjustment_note:
                    override["note"] = line.adjustment_note
                overrides[str(line.reservation_id)] = override

            # 4. Construire le payload client
            client_payload: dict[str, Any] | None = None
            if invoice.client:
                client = invoice.client
                user = getattr(client, "user", None)
                client_payload = {
                    "id": client.id,
                    "first_name": getattr(user, "first_name", None) if user else None,
                    "last_name": getattr(user, "last_name", None) if user else None,
                    "username": getattr(user, "username", None) if user else None,
                    "full_name": (
                        f"{getattr(user, 'first_name', '') or ''} "
                        f"{getattr(user, 'last_name', '') or ''}"
                    ).strip()
                    if user
                    else None,
                }

            # 5. Annuler la facture d'origine pour libérer les réservations
            from application.invoices.cancel_invoice import CancelInvoiceInput

            cancel_input = CancelInvoiceInput(invoice=invoice, force=True)
            cancel_result = self.cancel_invoice_use_case.execute(cancel_input)
            if not cancel_result.success:
                msg = "Erreur lors de l'annulation de la facture"
                raise ValueError(msg)

            # 6. Retourner le contexte pour pré-remplir le formulaire
            draft_context = {
                "billing_type": billing_type,
                "client_id": invoice.client_id,
                "bill_to_client_id": invoice.bill_to_client_id,
                "period_year": invoice.period_year,
                "period_month": invoice.period_month,
                "reservation_ids": reservation_ids,
                "overrides": overrides,
                "client": client_payload,
            }

            return DuplicateInvoiceOutput(success=True, draft_context=draft_context)

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
