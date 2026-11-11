"""Use-case: annuler une facture."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from models.enums import InvoiceBillingStrategy, InvoiceStatus

logger = logging.getLogger(__name__)


def _is_direct_client_invoice(invoice: Any) -> bool:
    """Détecte une facture client directe (patient payeur, pas clinique/tierce).

    Utilisé pour ne jamais recalculer ni appliquer résolution patient→clinique
    à l'annulation ; préserver l'override « facturer client » (billed_to_type
    reste 'patient').
    """
    billed_to = getattr(invoice, "billed_to_company_id", None)
    strategy = getattr(invoice, "billing_strategy", None)
    bill_to_client = getattr(invoice, "bill_to_client_id", None)
    if billed_to is not None or bill_to_client is not None:
        return False
    try:
        s = (
            strategy
            if isinstance(strategy, InvoiceBillingStrategy)
            else InvoiceBillingStrategy(strategy)
        )
    except (ValueError, TypeError):
        return False
    return s == InvoiceBillingStrategy.S1_PATIENT


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
        force: Si True, les appelants historiques demandent d'annuler hors
            brouillon. Ce drapeau ne contourne plus une facture émise.
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
    error: dict[str, Any] | None = None
    status_code: int | None = None


class CancelInvoiceUseCase:
    """Use-case Application: annuler une facture en libérant les
    réservations associées."""

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
            - DB: Met à jour Invoice.status, Invoice.balance_due,
              Booking.invoice_line_id
            - Si facture client directe (S1_PATIENT, pas clinique/tierce) :
              ré-assertion billed_to_type='patient', billed_to_company_id=None,
              billing_party_id=None sur chaque booking libéré (override
              « facturer client » préservé, jamais de résolution patient→clinique).
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

        from sqlalchemy import select
        from sqlalchemy.orm import Query

        from models import Booking, InvoicePayment, db
        from models import Invoice as InvoiceModel

        working: Any = invoice
        if isinstance(invoice, InvoiceModel):
            locked = db.session.scalar(
                select(InvoiceModel)
                .where(InvoiceModel.id == int(invoice.id))
                .with_for_update()
            )
            if locked is None:
                return CancelInvoiceOutput(
                    success=False,
                    error={"error": "Facture introuvable"},
                    status_code=404,
                )
            db.session.refresh(locked)
            working = locked
            current_status = working.status
            if isinstance(current_status, InvoiceStatus):
                status_value = current_status
            else:
                try:
                    status_value = InvoiceStatus(current_status)
                except (ValueError, TypeError):
                    return CancelInvoiceOutput(
                        success=False,
                        error={"error": "Statut de facture invalide"},
                        status_code=400,
                    )

        if status_value == InvoiceStatus.CANCELLED:
            return CancelInvoiceOutput(success=True)

        if status_value != InvoiceStatus.DRAFT:
            from application.invoices.issued_invoice_guard import (
                invoice_status_value,
                issued_invoice_refusal_payload,
            )

            return CancelInvoiceOutput(
                success=False,
                error=issued_invoice_refusal_payload(
                    invoice_id=getattr(working, "id", None),
                    status=invoice_status_value(working),
                    operation="cancel_invoice",
                ),
                status_code=409,
            )

        # Libérer les réservations liées à chaque ligne

        is_direct_client = _is_direct_client_invoice(working)
        freed_count = 0

        line_ids: set[int] = set()
        for line in working.lines:
            lid = getattr(line, "id", None)
            if lid is not None:
                line_ids.add(int(lid))

        # Tous les bookings pointant vers une ligne de cette facture (y compris
        # segments A/R secondaires non couverts par line.reservation_id seul).
        bookings_by_id: dict[int, Any] = {}
        if line_ids:
            booking_query = Booking.query.filter(Booking.invoice_line_id.in_(line_ids))
            if isinstance(booking_query, Query):
                booking_query = booking_query.with_for_update()
            for b in booking_query.all():
                bookings_by_id[int(b.id)] = b

        for line in working.lines:
            if hasattr(line, "reservation_id") and line.reservation_id:
                booking_dto = self._booking_repo.find_by_id(line.reservation_id)
                if booking_dto:
                    b2 = Booking.query.get(booking_dto.id)
                    if b2 is not None:
                        bookings_by_id[int(b2.id)] = b2

        for booking in bookings_by_id.values():
            if not booking or not hasattr(booking, "invoice_line_id"):
                continue
            if booking.invoice_line_id is None:
                continue
            if int(booking.invoice_line_id) not in line_ids:
                continue
            booking.invoice_line_id = None
            booking.updated_at = datetime.now(UTC)
            # Facture client directe : ne pas réécrire un payeur verrouillé
            # pour un règlement sur place. Sans ce verrou, ré-asserter le client.
            if is_direct_client:
                from application.billing.settlement_payer_lock import (
                    refuse_ordinary_payer_change,
                )

                if refuse_ordinary_payer_change(int(booking.id)) is None:
                    try:
                        booking.billed_to_type = "patient"
                        booking.billed_to_company_id = None
                        booking.billing_party_id = None
                    except Exception:  # best-effort (DTO / champs absents)
                        pass
            freed_count += 1
            if os.getenv("BILLING_DEBUG", "0") == "1":
                _fmt = (
                    "[CancelInvoice] freed booking_id=%s billed_to_type=%s "
                    "billed_to_company_id=%s billing_party_id=%s status=%s invoice_line_id=%s"
                )
                logger.info(
                    _fmt,
                    booking.id,
                    getattr(booking, "billed_to_type", None),
                    getattr(booking, "billed_to_company_id", None),
                    getattr(booking, "billing_party_id", None),
                    getattr(booking, "status", None),
                    getattr(booking, "invoice_line_id", None),
                )

        invoice_id_val = getattr(invoice, "id", None)
        logger.info(
            "Cancel invoice_id=%s freed %s booking(s)",
            invoice_id_val,
            freed_count,
        )

        if isinstance(working, InvoiceModel):
            allocations = InvoicePayment.query.filter(
                InvoicePayment.invoice_id == int(working.id),
                InvoicePayment.cash_collection_id.isnot(None),
            ).all()
            for allocation in allocations:
                db.session.delete(allocation)
            db.session.flush()
            kept = InvoicePayment.query.filter_by(invoice_id=int(working.id)).all()
            working.amount_paid = sum(
                (Decimal(str(payment.amount or 0)) for payment in kept),
                Decimal("0.00"),
            )

        # Le brouillon reste en base, avec ses lignes. Seule l'imputation
        # d'espèces est retirée : la collection et ses événements demeurent.
        invoice_any: Any = working
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
