"""Use-case: dupliquer une facture.

Ce use case migre la logique métier de InvoiceService.duplicate_invoice()
vers l'architecture DDD.
"""

from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass
from decimal import Decimal
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
        """Prépare un duplicata fidèle sans toucher la facture d'origine.

        Une facture émise reste immuable : ni numéro, ni montants, ni lignes,
        ni annulation. Le résultat est seulement le contexte d'un nouveau
        brouillon (client, période, courses, ajustements de ligne).
        L'avoir et la rectificative sont d'autres opérations.
        ``cancel_invoice_use_case`` n'est pas appelé.
        """
        invoice = input_data.invoice

        try:
            if invoice.status == InvoiceStatus.DRAFT:
                msg = (
                    "La facture est déjà un brouillon et peut être modifiée "
                    "directement."
                )
                raise ValueError(msg)

            draft_context = _faithful_draft_context(invoice)
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


def _money(value: object) -> str:
    return format(Decimal(str(value)).quantize(Decimal("0.01")), "f")


def _faithful_draft_context(invoice: Invoice) -> dict[str, object]:
    """Copie de lecture : aucune écriture sur la facture émise."""
    lines = list(invoice.lines or [])
    reservation_ids = sorted(
        {
            int(line.reservation_id)
            for line in lines
            if getattr(line, "reservation_id", None) is not None
        }
    )
    if not reservation_ids:
        msg = "Aucune course liée à cette facture : le duplicata est impossible."
        raise ValueError(msg)

    overrides: dict[str, dict[str, str]] = {}
    for line in lines:
        reservation_id = getattr(line, "reservation_id", None)
        if reservation_id is None:
            continue
        entry: dict[str, str] = {}
        line_total = getattr(line, "line_total", None)
        if line_total is not None:
            entry["line_total"] = _money(line_total)
        vat_rate = getattr(line, "vat_rate", None)
        if vat_rate is not None:
            entry["vat_rate"] = _money(vat_rate)
        note = getattr(line, "adjustment_note", None)
        if note:
            entry["adjustment_note"] = str(note)
        if entry:
            overrides[str(int(reservation_id))] = entry

    draft: dict[str, object] = {
        "source_invoice_id": int(invoice.id),
        "client_id": invoice.client_id,
        "bill_to_client_id": invoice.bill_to_client_id,
        "period_year": invoice.period_year,
        "period_month": invoice.period_month,
        "reservation_ids": reservation_ids,
        "overrides": overrides,
    }
    user = getattr(getattr(invoice, "client", None), "user", None)
    if user is not None:
        first = getattr(user, "first_name", None) or ""
        last = getattr(user, "last_name", None) or ""
        display = f"{first} {last}".strip() or getattr(user, "username", None)
        draft["client"] = {"id": invoice.client_id, "display_name": display}
    return draft
