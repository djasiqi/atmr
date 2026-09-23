"""Enregistrement des créances PORTAL émises par un transporteur.

Le montant est celui de la facture réelle. ``booking.amount`` n'est jamais
lu comme créance. Aucun hold de réservation n'est appliqué ici.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ext import db
from models.booking import Booking
from models.client import Client
from models.client_booking_contract_event import (
    EVENT_BOOKING_CREATED,
    ClientBookingContractEvent,
)
from models.company import Company
from models.enums import BookingStatus
from models.portal_receivable import (
    DISPUTE_ACCEPTED,
    DISPUTE_OPEN,
    DISPUTE_REJECTED,
    PAYMENT_METHODS,
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_ISSUED,
    RECEIVABLE_OVERDUE,
    RECEIVABLE_PAID,
    RECEIVABLE_PARTIALLY_PAID,
    PortalReceivable,
    PortalReceivableDispute,
    PortalReceivableLine,
    PortalReceivablePayment,
)
from services.auth.portal_phone_verification import is_portal_client
from services.billing.portal_payment_hold import (
    hold_effect_for_receivable,
    is_receivable_overdue_for_display,
)

_TWO = Decimal("0.01")

BILLABLE_BOOKING_STATUSES = frozenset(
    {
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
        "completed",
        "return_completed",
    }
)


class PortalReceivableError(Exception):
    """Erreur métier d'enregistrement d'une créance PORTAL."""

    code = "portal_receivable_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.message = message
        if code is not None:
            self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ReceivableLineInput:
    booking_id: int
    invoiced_amount: Decimal
    description: str | None = None


def _money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(_TWO, rounding=ROUND_HALF_UP)


def _status_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


def _is_billable_status(booking: Booking) -> bool:
    status = _status_text(getattr(booking, "status", None))
    return status in BILLABLE_BOOKING_STATUSES or status.upper() in {
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
    }


def _created_event(booking_id: int) -> ClientBookingContractEvent:
    event = (
        ClientBookingContractEvent.query.filter_by(
            booking_id=booking_id, event_type=EVENT_BOOKING_CREATED
        )
        .order_by(ClientBookingContractEvent.id.asc())
        .first()
    )
    if event is None:
        raise PortalReceivableError(
            "Aucune preuve BOOKING_CREATED pour cette course.",
            code="booking_contract_event_missing",
        )
    return event


def _assert_portal_booking_owned(*, booking: Booking, company_id: int) -> Client:
    if int(getattr(booking, "company_id", 0) or 0) != int(company_id):
        raise PortalReceivableError(
            "Cette course n'appartient pas à votre entreprise.",
            code="booking_not_owned",
        )
    client = db.session.get(Client, getattr(booking, "client_id", None))
    if client is None or not is_portal_client(client):
        raise PortalReceivableError(
            "Seules les courses d'un client privé peuvent former une créance PORTAL.",
            code="portal_client_required",
        )
    if not _is_billable_status(booking):
        raise PortalReceivableError(
            "La course doit être terminée avant d'enregistrer une facture.",
            code="booking_not_billable",
        )
    return client


def compute_receivable_status(receivable: PortalReceivable, *, now: datetime | None = None) -> str:
    """Statut dérivé du solde, de l'échéance et des marques dispute/annulation."""
    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        return RECEIVABLE_CANCELLED
    if receivable.status == RECEIVABLE_DISPUTED or receivable.disputed_at is not None:
        return RECEIVABLE_DISPUTED
    balance = _money(receivable.balance_due)
    if balance <= Decimal("0.00"):
        return RECEIVABLE_PAID
    paid = _money(receivable.amount_paid)
    moment = now or datetime.now(UTC)
    due = receivable.due_date
    if due is not None and due.tzinfo is None:
        due = due.replace(tzinfo=UTC)
    if due is not None and due < moment:
        return RECEIVABLE_OVERDUE
    if paid > Decimal("0.00"):
        return RECEIVABLE_PARTIALLY_PAID
    return RECEIVABLE_ISSUED


def refresh_receivable_balances(receivable: PortalReceivable) -> None:
    paid = sum((_money(p.amount) for p in receivable.payments), Decimal("0.00"))
    receivable.amount_paid = paid
    receivable.balance_due = max(_money(receivable.total_amount) - paid, Decimal("0.00"))
    receivable.status = compute_receivable_status(receivable)


def create_portal_receivable(
    *,
    company: Company,
    recorded_by_user_id: int,
    external_invoice_number: str,
    issued_at: datetime,
    due_date: datetime,
    lines: list[ReceivableLineInput],
    currency: str = "CHF",
) -> PortalReceivable:
    """Crée une créance émise. Ne commit pas."""
    number = (external_invoice_number or "").strip()
    if not number:
        raise PortalReceivableError(
            "Le numéro de facture du transporteur est obligatoire.",
            code="external_invoice_number_required",
        )
    if not lines:
        raise PortalReceivableError(
            "Au moins une ligne de course est obligatoire.",
            code="lines_required",
        )
    if due_date < issued_at:
        raise PortalReceivableError(
            "L'échéance ne peut pas précéder la date d'émission.",
            code="due_date_before_issued_at",
        )

    existing = PortalReceivable.query.filter_by(
        creditor_company_id=int(company.id),
        external_invoice_number=number,
    ).one_or_none()
    if existing is not None:
        raise PortalReceivableError(
            "Ce numéro de facture est déjà enregistré pour votre entreprise.",
            code="external_invoice_number_duplicate",
        )

    debtor_user_id: int | None = None
    debtor_name = ""
    debtor_email: str | None = None
    debtor_phone: str | None = None
    debtor_address: str | None = None
    built_lines: list[PortalReceivableLine] = []
    total = Decimal("0.00")

    for line in lines:
        amount = _money(line.invoiced_amount)
        if amount <= Decimal("0.00"):
            raise PortalReceivableError(
                "Chaque montant facturé doit être strictement positif.",
                code="invoiced_amount_invalid",
            )
        booking = db.session.get(Booking, int(line.booking_id))
        if booking is None:
            raise PortalReceivableError(
                f"Course {line.booking_id} introuvable.",
                code="booking_not_found",
            )
        _assert_portal_booking_owned(booking=booking, company_id=int(company.id))
        event = _created_event(int(booking.id))
        if event.debtor_user_id is None:
            raise PortalReceivableError(
                "Le débiteur contractuel de la course est incomplet.",
                code="debtor_unresolved",
            )
        if debtor_user_id is None:
            debtor_user_id = int(event.debtor_user_id)
            debtor_name = str(event.debtor_name_snapshot or event.customer_name_snapshot or "")
            debtor_email = event.debtor_email_snapshot
            debtor_phone = event.debtor_phone_snapshot
            debtor_address = event.debtor_billing_address_snapshot
        elif int(event.debtor_user_id) != int(debtor_user_id):
            raise PortalReceivableError(
                "Une créance ne peut pas mélanger plusieurs débiteurs.",
                code="mixed_debtors",
            )
        total += amount
        built_lines.append(
            PortalReceivableLine(
                booking_id=int(booking.id),
                booking_contract_event_id=int(event.id),
                invoiced_amount=amount,
                description=(line.description or None),
            )
        )

    assert debtor_user_id is not None
    receivable = PortalReceivable(
        creditor_company_id=int(company.id),
        creditor_name_snapshot=str(getattr(company, "name", "") or f"Entreprise {company.id}"),
        debtor_user_id=debtor_user_id,
        debtor_name_snapshot=debtor_name or f"Client {debtor_user_id}",
        debtor_email_snapshot=debtor_email,
        debtor_phone_snapshot=debtor_phone,
        debtor_billing_address_snapshot=debtor_address,
        external_invoice_number=number,
        currency=(currency or "CHF").strip().upper()[:3] or "CHF",
        issued_at=issued_at,
        due_date=due_date,
        total_amount=total,
        amount_paid=Decimal("0.00"),
        balance_due=total,
        status=RECEIVABLE_ISSUED,
        recorded_by_user_id=int(recorded_by_user_id),
    )
    db.session.add(receivable)
    db.session.flush()
    for row in built_lines:
        row.receivable_id = receivable.id
        db.session.add(row)
    db.session.flush()
    receivable.status = compute_receivable_status(receivable)
    return receivable


def add_portal_receivable_payment(
    *,
    receivable: PortalReceivable,
    amount: Decimal,
    paid_at: datetime,
    method: str,
    recorded_by_user_id: int,
    reference: str | None = None,
) -> PortalReceivablePayment:
    """Ajoute un paiement historique. Ne commit pas."""
    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        raise PortalReceivableError(
            "Impossible d'enregistrer un paiement sur une créance annulée.",
            code="receivable_cancelled",
        )
    method_key = (method or "").strip().lower()
    if method_key not in PAYMENT_METHODS:
        raise PortalReceivableError(
            "Méthode de paiement non supportée.",
            code="payment_method_invalid",
        )
    money = _money(amount)
    if money <= Decimal("0.00"):
        raise PortalReceivableError(
            "Le montant du paiement doit être positif.",
            code="payment_amount_invalid",
        )
    payment = PortalReceivablePayment(
        amount=money,
        paid_at=paid_at,
        method=method_key,
        reference=(reference or None),
        recorded_by_user_id=int(recorded_by_user_id),
    )
    # Append sur la relation pour que le solde voie tout de suite le paiement.
    receivable.payments.append(payment)
    db.session.flush()
    refresh_receivable_balances(receivable)
    return payment


def cancel_portal_receivable(
    *,
    receivable: PortalReceivable,
    reason: str,
    actor_user_id: int,
) -> PortalReceivable:
    text = (reason or "").strip()
    if not text:
        raise PortalReceivableError(
            "Le motif d'annulation est obligatoire.",
            code="cancellation_reason_required",
        )
    receivable.cancelled_at = datetime.now(UTC)
    receivable.cancellation_reason = text
    receivable.cancelled_by_user_id = int(actor_user_id)
    receivable.status = RECEIVABLE_CANCELLED
    return receivable


def dispute_portal_receivable(
    *,
    receivable: PortalReceivable,
    reason: str,
    actor_user_id: int,
) -> tuple[PortalReceivable, PortalReceivableDispute]:
    """Ouvre une contestation (idempotente si déjà ouverte)."""
    if receivable.status == RECEIVABLE_CANCELLED or receivable.cancelled_at is not None:
        raise PortalReceivableError(
            "Une créance annulée ne peut pas être contestée.",
            code="receivable_cancelled",
        )
    text = (reason or "").strip()
    if not text:
        raise PortalReceivableError(
            "Le motif de contestation est obligatoire.",
            code="dispute_reason_required",
        )
    open_dispute = (
        PortalReceivableDispute.query.filter_by(
            receivable_id=int(receivable.id), status=DISPUTE_OPEN
        )
        .order_by(PortalReceivableDispute.id.desc())
        .first()
    )
    if open_dispute is not None:
        # Idempotent : même action, pas de second litige ouvert.
        receivable.status = RECEIVABLE_DISPUTED
        return receivable, open_dispute

    dispute = PortalReceivableDispute(
        receivable_id=int(receivable.id),
        reason=text,
        disputed_by_user_id=int(actor_user_id),
        status=DISPUTE_OPEN,
    )
    receivable.disputed_at = datetime.now(UTC)
    receivable.dispute_reason = text
    receivable.disputed_by_user_id = int(actor_user_id)
    receivable.status = RECEIVABLE_DISPUTED
    receivable.disputes.append(dispute)
    db.session.flush()
    return receivable, dispute


def reject_portal_receivable_dispute(
    *,
    receivable: PortalReceivable,
    actor_user_id: int,
    resolution_note: str | None = None,
) -> PortalReceivableDispute:
    """Rejette la contestation ouverte : la créance redevient éligible au hold."""
    open_dispute = (
        PortalReceivableDispute.query.filter_by(
            receivable_id=int(receivable.id), status=DISPUTE_OPEN
        )
        .order_by(PortalReceivableDispute.id.desc())
        .first()
    )
    if open_dispute is None:
        raise PortalReceivableError(
            "Aucune contestation ouverte à rejeter.",
            code="dispute_not_open",
        )
    open_dispute.status = DISPUTE_REJECTED
    open_dispute.resolved_at = datetime.now(UTC)
    open_dispute.resolved_by_user_id = int(actor_user_id)
    open_dispute.resolution_note = (resolution_note or "").strip() or None
    receivable.disputed_at = None
    receivable.dispute_reason = None
    receivable.disputed_by_user_id = None
    # Retirer le statut disputed avant recalcul solde/échéance.
    receivable.status = RECEIVABLE_ISSUED
    refresh_receivable_balances(receivable)
    return open_dispute


def accept_portal_receivable_dispute(
    *,
    receivable: PortalReceivable,
    actor_user_id: int,
    resolution_note: str | None = None,
) -> PortalReceivableDispute:
    """Accepte la contestation : annulation soft de la créance."""
    open_dispute = (
        PortalReceivableDispute.query.filter_by(
            receivable_id=int(receivable.id), status=DISPUTE_OPEN
        )
        .order_by(PortalReceivableDispute.id.desc())
        .first()
    )
    if open_dispute is None:
        raise PortalReceivableError(
            "Aucune contestation ouverte à accepter.",
            code="dispute_not_open",
        )
    open_dispute.status = DISPUTE_ACCEPTED
    open_dispute.resolved_at = datetime.now(UTC)
    open_dispute.resolved_by_user_id = int(actor_user_id)
    open_dispute.resolution_note = (resolution_note or "").strip() or None
    cancel_portal_receivable(
        receivable=receivable,
        reason=open_dispute.resolution_note
        or open_dispute.reason
        or "Contestation acceptée",
        actor_user_id=actor_user_id,
    )
    return open_dispute


def serialize_portal_receivable_for_client(
    receivable: PortalReceivable,
) -> dict[str, Any]:
    """Lecture client : pas d'infos internes inutiles."""
    from models.portal_receivable_dunning import PortalReceivableDunningEvent
    from services.billing.portal_receivable_dunning import serialize_dunning_event

    status = compute_receivable_status(receivable)
    overdue = is_receivable_overdue_for_display(receivable)
    effect = hold_effect_for_receivable(receivable)
    open_dispute = next(
        (d for d in receivable.disputes if d.status == DISPUTE_OPEN),
        None,
    )
    dunning_events = (
        PortalReceivableDunningEvent.query.filter_by(receivable_id=int(receivable.id))
        .order_by(PortalReceivableDunningEvent.id.asc())
        .all()
    )
    return {
        "receivable_id": receivable.id,
        "creditor_company_name": receivable.creditor_name_snapshot,
        "external_invoice_number": receivable.external_invoice_number,
        "issued_at": receivable.issued_at.isoformat() if receivable.issued_at else None,
        "due_date": receivable.due_date.isoformat() if receivable.due_date else None,
        "currency": receivable.currency,
        "total_amount": float(receivable.total_amount),
        "amount_paid": float(receivable.amount_paid),
        "balance_due": float(receivable.balance_due),
        "status": status,
        "is_overdue": overdue,
        "hold_effect": effect,
        "can_dispute": open_dispute is None
        and status not in (RECEIVABLE_CANCELLED, RECEIVABLE_DISPUTED, RECEIVABLE_PAID)
        and float(receivable.balance_due) > 0,
        "dispute_status": open_dispute.status if open_dispute else None,
        "dunning_history": [
            serialize_dunning_event(e, for_client=True) for e in dunning_events
        ],
    }


def serialize_portal_receivable(receivable: PortalReceivable) -> dict[str, Any]:
    return {
        "id": receivable.id,
        "creditor_company_id": receivable.creditor_company_id,
        "creditor_name_snapshot": receivable.creditor_name_snapshot,
        "debtor_user_id": receivable.debtor_user_id,
        "debtor_name_snapshot": receivable.debtor_name_snapshot,
        "debtor_email_snapshot": receivable.debtor_email_snapshot,
        "debtor_phone_snapshot": receivable.debtor_phone_snapshot,
        "debtor_billing_address_snapshot": receivable.debtor_billing_address_snapshot,
        "external_invoice_number": receivable.external_invoice_number,
        "currency": receivable.currency,
        "issued_at": receivable.issued_at.isoformat() if receivable.issued_at else None,
        "due_date": receivable.due_date.isoformat() if receivable.due_date else None,
        "total_amount": float(receivable.total_amount),
        "amount_paid": float(receivable.amount_paid),
        "balance_due": float(receivable.balance_due),
        "status": compute_receivable_status(receivable),
        "disputed_at": receivable.disputed_at.isoformat() if receivable.disputed_at else None,
        "dispute_reason": receivable.dispute_reason,
        "cancelled_at": receivable.cancelled_at.isoformat() if receivable.cancelled_at else None,
        "cancellation_reason": receivable.cancellation_reason,
        "lines": [
            {
                "id": line.id,
                "booking_id": line.booking_id,
                "booking_contract_event_id": line.booking_contract_event_id,
                "invoiced_amount": float(line.invoiced_amount),
                "description": line.description,
            }
            for line in receivable.lines
        ],
        "payments": [
            {
                "id": payment.id,
                "amount": float(payment.amount),
                "paid_at": payment.paid_at.isoformat() if payment.paid_at else None,
                "method": payment.method,
                "reference": payment.reference,
            }
            for payment in receivable.payments
        ],
        "disputes": [
            {
                "id": dispute.id,
                "status": dispute.status,
                "reason": dispute.reason,
                "created_at": (
                    dispute.created_at.isoformat() if dispute.created_at else None
                ),
                "resolved_at": (
                    dispute.resolved_at.isoformat() if dispute.resolved_at else None
                ),
            }
            for dispute in receivable.disputes
        ],
    }
