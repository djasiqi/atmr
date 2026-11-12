"""Simulations Tap to Pay, déclaration d'espèces et exceptions.

Aucun appel réseau, aucun écriture comptable, aucune confirmation
électronique. Une réponse perdue ne lance pas un second débit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

TAP_READY = "ready"
TAP_IN_PROGRESS = "in_progress"
TAP_CONFIRMED = "confirmed"
TAP_REFUSED = "refused"
TAP_UNCERTAIN = "uncertain"
TAP_INTERRUPTED = "interrupted"
TAP_ALREADY_DONE = "already_done"

_TAP_RESULTS = frozenset({TAP_CONFIRMED, TAP_REFUSED, TAP_UNCERTAIN, TAP_INTERRUPTED})
_OPEN_BLOCK = frozenset({TAP_IN_PROGRESS, TAP_UNCERTAIN, TAP_INTERRUPTED})

EXCEPTION_CANNOT_PAY = "cannot_pay"
EXCEPTION_SIGNATURE_REFUSED = "signature_refused"
EXCEPTION_SIGNATURE_UNABLE = "signature_unable"
EXCEPTION_REPRESENTATIVE = "authorized_representative"
EXCEPTION_TECHNICAL = "technical"

_EXCEPTIONS = frozenset(
    {
        EXCEPTION_CANNOT_PAY,
        EXCEPTION_SIGNATURE_REFUSED,
        EXCEPTION_SIGNATURE_UNABLE,
        EXCEPTION_REPRESENTATIVE,
        EXCEPTION_TECHNICAL,
    }
)


class CashDeclarationError(ValueError):
    """La déclaration d'espèces est incomplète."""


@dataclass(frozen=True, slots=True)
class TapAttempt:
    """État d'une tentative simulée. Le prestataire n'est pas contacté."""

    state: str
    attempt_id: str | None
    amount_chf: Decimal
    currency: str
    merchant_company_id: int
    may_start_another_debit: bool


@dataclass(frozen=True, slots=True)
class CashDeclaration:
    """Espèces déclarées. Ce n'est pas un paiement électronique confirmé."""

    booking_id: int
    payment_scope: str
    expected_amount_chf: Decimal
    received_amount_chf: Decimal
    company_id: int
    driver_id: int
    recorded_at: datetime
    cash_reference: str
    status: str
    electronic_confirmation: bool


@dataclass(frozen=True, slots=True)
class SettlementExceptionRecord:
    """Incident tracé. La course réalisée n'est pas annulée."""

    kind: str
    payment_marked_received: bool
    signature_recorded: bool
    service_cancelled: bool
    invoice_auto_issued: bool


def open_tap_attempt(
    *,
    current_state: str,
    current_attempt_id: str | None,
    amount_chf: Decimal,
    currency: str,
    merchant_company_id: int,
    new_attempt_id: str | None,
) -> TapAttempt:
    """Ouvre une tentative simulée, ou refuse un second débit."""
    if currency != "CHF":
        raise ValueError("Devise attendue : CHF.")
    if amount_chf <= 0 or merchant_company_id < 1:
        raise ValueError("Montant ou entreprise marchande invalide.")
    if current_state in {TAP_CONFIRMED, TAP_ALREADY_DONE}:
        return TapAttempt(
            state=TAP_ALREADY_DONE,
            attempt_id=current_attempt_id,
            amount_chf=amount_chf,
            currency=currency,
            merchant_company_id=merchant_company_id,
            may_start_another_debit=False,
        )
    if current_state in _OPEN_BLOCK:
        return TapAttempt(
            state=TAP_UNCERTAIN,
            attempt_id=current_attempt_id,
            amount_chf=amount_chf,
            currency=currency,
            merchant_company_id=merchant_company_id,
            may_start_another_debit=False,
        )
    attempt_id = (new_attempt_id or "").strip()
    if not attempt_id:
        raise ValueError("Identifiant de tentative requis.")
    return TapAttempt(
        state=TAP_IN_PROGRESS,
        attempt_id=attempt_id,
        amount_chf=amount_chf,
        currency=currency,
        merchant_company_id=merchant_company_id,
        may_start_another_debit=False,
    )


def resolve_tap_simulation(attempt: TapAttempt, provider_result: str) -> TapAttempt:
    """Applique un résultat simulé. N'appelle aucun prestataire."""
    if provider_result not in _TAP_RESULTS:
        raise ValueError("Résultat simulé inconnu.")
    if attempt.state != TAP_IN_PROGRESS:
        state = TAP_ALREADY_DONE if attempt.state == TAP_CONFIRMED else attempt.state
        return TapAttempt(
            state=state,
            attempt_id=attempt.attempt_id,
            amount_chf=attempt.amount_chf,
            currency=attempt.currency,
            merchant_company_id=attempt.merchant_company_id,
            may_start_another_debit=False,
        )
    return TapAttempt(
        state=provider_result,
        attempt_id=attempt.attempt_id,
        amount_chf=attempt.amount_chf,
        currency=attempt.currency,
        merchant_company_id=attempt.merchant_company_id,
        may_start_another_debit=False,
    )


def declare_cash(
    *,
    booking_id: int,
    payment_scope: str,
    expected_amount_chf: Decimal,
    received_amount_chf: Decimal | None,
    company_id: int,
    driver_id: int,
    recorded_at: datetime,
    cash_reference: str,
) -> CashDeclaration:
    """Enregistre des espèces réellement comptées, pas un choix de méthode."""
    if received_amount_chf is None:
        raise CashDeclarationError(
            "La sélection du mode espèces ne prouve pas la réception."
        )
    if received_amount_chf <= 0 or expected_amount_chf <= 0:
        raise CashDeclarationError("Montant d'espèces invalide.")
    if payment_scope not in {"booking", "group"}:
        raise CashDeclarationError("Périmètre d'espèces inconnu.")
    reference = cash_reference.strip()
    if booking_id < 1 or company_id < 1 or driver_id < 1 or not reference:
        raise CashDeclarationError("Déclaration d'espèces incomplète.")
    if recorded_at.tzinfo is None:
        raise CashDeclarationError("Horodatage serveur avec fuseau requis.")
    status = "declared" if received_amount_chf == expected_amount_chf else "difference"
    return CashDeclaration(
        booking_id=booking_id,
        payment_scope=payment_scope,
        expected_amount_chf=expected_amount_chf,
        received_amount_chf=received_amount_chf,
        company_id=company_id,
        driver_id=driver_id,
        recorded_at=recorded_at,
        cash_reference=reference,
        status=status,
        electronic_confirmation=False,
    )


def record_exception(kind: str) -> SettlementExceptionRecord:
    """Trace un blocage sans inventer un encaissement ni une signature."""
    if kind not in _EXCEPTIONS:
        raise ValueError("Exception de règlement inconnue.")
    return SettlementExceptionRecord(
        kind=kind,
        payment_marked_received=False,
        signature_recorded=False,
        service_cancelled=False,
        invoice_auto_issued=False,
    )
