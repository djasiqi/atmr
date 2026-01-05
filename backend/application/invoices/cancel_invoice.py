"""Use-case: annuler une facture."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from models.enums import InvoiceStatus

logger = logging.getLogger(__name__)


class _BookingRepo(Protocol):
    def find_by_id(self, booking_id: int) -> Any | None: ...  # Returns Booking DTO


class _InvoiceLike(Protocol):
    id: Any  # Accepte int ou Mapped[int] (SQLAlchemy)
    status: Any
    cancelled_at: Any | None
    updated_at: Any | None
    balance_due: Any
    lines: Any  # Accepte list[Any] ou _RelationshipDeclared[Any] (SQLAlchemy)


@dataclass(frozen=True, slots=True)
class CancelInvoiceInput:
    """Input pour annuler une facture.

    Attributes:
        invoice: La facture à annuler
        force: Si True, annule même si le statut n'est pas DRAFT ou CANCELLED
    """

    invoice: _InvoiceLike
    force: bool = False


@dataclass(frozen=True, slots=True)
class CancelInvoiceOutput:
    """Output pour annuler une facture.

    Attributes:
        success: True si l'opération a réussi
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class CancelInvoiceUseCase:
    """Use-case Application: annuler une facture en libérant les réservations associées."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        booking_repo: _BookingRepo,
    ) -> None:
        """Initialise le use case avec ses dépendances.

        Args:
            booking_repo: Repository pour les réservations
        """
        self._booking_repo = booking_repo

    def execute(self, input_data: CancelInvoiceInput) -> CancelInvoiceOutput:
        """Annule une facture en libérant les réservations associées.

        Args:
            input_data: Input avec invoice et force

        Returns:
            CancelInvoiceOutput avec le résultat de l'opération

        Side-effects:
            - DB: Met à jour Invoice.status, Invoice.balance_due, Booking.invoice_line_id
            - DB: Commit transaction (ou rollback en cas d'erreur)
        """
        invoice = input_data.invoice
        current_status = invoice.status
        if isinstance(current_status, InvoiceStatus):
            status_value = current_status
        else:
            # Si c'est une string, convertir
            try:
                status_value = InvoiceStatus(current_status)
            except (ValueError, TypeError):
                return CancelInvoiceOutput(
                    success=False,
                    error={"error": "Statut de facture invalide"},
                    status_code=400,
                )

        if not input_data.force and status_value not in {
            InvoiceStatus.DRAFT,
            InvoiceStatus.CANCELLED,
        }:
            return CancelInvoiceOutput(
                success=False,
                error={
                    "error": "Seules les factures au statut 'draft' peuvent être annulées."
                },
                status_code=400,
            )

        if status_value == InvoiceStatus.CANCELLED:
            # Rien à faire, déjà annulée
            return CancelInvoiceOutput(success=True)

        # Libérer les réservations liées à chaque ligne
        from models import Booking, db

        for line in invoice.lines:
            if hasattr(line, "reservation_id") and line.reservation_id:
                booking_dto = self._booking_repo.find_by_id(line.reservation_id)
                if booking_dto:
                    # Récupérer le modèle SQLAlchemy depuis le DTO pour la compatibilité
                    booking = Booking.query.get(booking_dto.id)
                    if (
                        booking
                        and hasattr(booking, "invoice_line_id")
                        and booking.invoice_line_id == line.id
                    ):
                        booking.invoice_line_id = None
                        booking.updated_at = datetime.now(UTC)

        # Mettre à jour la facture
        invoice_any: Any = invoice
        invoice_any.status = InvoiceStatus.CANCELLED
        invoice_any.cancelled_at = datetime.now(UTC)
        invoice_any.updated_at = datetime.now(UTC)
        invoice_any.balance_due = Decimal("0.00")

        try:
            db.session.commit()
            return CancelInvoiceOutput(success=True)
        except Exception:
            db.session.rollback()
            logger.exception("Erreur lors de l'annulation de la facture")
            return CancelInvoiceOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
