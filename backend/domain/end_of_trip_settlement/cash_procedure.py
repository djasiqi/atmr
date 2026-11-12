"""Procédure d'espèces. Aucune ligne n'est écrite en base.

Le choix du mode n'est pas un paiement. La proposition V1 — règlement
intégral et monnaie explicitement déclarée — n'est pas approuvée.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

_CENT = Decimal("0.01")
FULL_PAYMENT_POLICY = "pending_approval"
CHANGE_POLICY = "pending_approval"
PARTIAL_POLICY = "pending_approval"
SURPLUS_POLICY = "pending_approval"
RECEIPT_TEXT_STATUS = "draft"
RECEIPT_TEXT = (
    "Ce reçu confirme la réception du montant indiqué en espèces "
    "pour la prestation de transport référencée. "
    "Il ne remplace pas l'attestation de réalisation du transport."
)
_RECORDABLE = frozenset({"exact", "full_with_explicit_change"})
_NO_RECEIPT = frozenset(
    {
        "mode_selected",
        "patient_announced",
        "simulated_success",
        "declaration_pending",
        "partial_forbidden",
        "surplus_not_returned",
        "change_not_confirmed",
        "change_mismatch",
        "declaration_error",
        "absent",
    }
)


@dataclass(frozen=True, slots=True)
class CashConsultation:
    """Écran préparé. Confirmer n'a pas encore été demandé."""

    booking_id: int
    payment_scope: str
    amount_due_chf: Decimal
    currency: str
    beneficiary_company_id: int
    mode: str
    confirmation_required: bool
    payment_confirmed: bool
    persists: bool


@dataclass(frozen=True, slots=True)
class CashAmounts:
    """Cinq sommes distinctes. Le rapprochement n'est pas fait par le chauffeur."""

    amount_due_chf: Decimal
    amount_tendered_chf: Decimal | None
    change_returned_chf: Decimal | None
    net_collected_chf: Decimal | None
    reconciled_amount_chf: None
    currency: str
    situation: str
    payment_confirmed: bool
    persists: bool
    change_policy: str
    partial_policy: str


@dataclass(frozen=True, slots=True)
class CashDeclarationRecord:
    """Déclaration simulée. Elle ne devient pas un paiement."""

    booking_id: int
    company_id: int
    driver_id: int
    amounts: CashAmounts
    idempotency_key: str
    recorded_at: datetime
    frozen: bool
    persists: bool
    supersedes_key: str | None


@dataclass(frozen=True, slots=True)
class CashReceiptDraft:
    """Justificatif futur. Il n'est pas remis au patient dans ce lot."""

    booking_id: int
    company_id: int
    amount_received_chf: Decimal | None
    currency: str
    issued: bool
    bank_transfer: bool
    platform_fee: bool


@dataclass(frozen=True, slots=True)
class CashReceiptProposal:
    """Modèle à faire approuver. Aucun document n'est émis."""

    issuer_company_id: int
    issuer_name: str
    fiscal_lines: tuple[str, ...]
    receipt_number: str | None
    received_at: datetime | None
    booking_reference: str
    payment_scope: str
    service_date: str | None
    debtor_label: str
    service_label: str
    amount_due_chf: Decimal
    amount_tendered_chf: Decimal | None
    change_returned_chf: Decimal | None
    net_collected_chf: Decimal | None
    balance_chf: Decimal | None
    till_reference: str | None
    statement: str
    statement_status: str
    issued: bool
    transmitted: bool
    duplicate: bool
    replaces_attestation: bool
    replaces_invoice: bool
    generation_allowed: bool
    blocked_reason: str
    medical_details_included: bool


@dataclass(frozen=True, slots=True)
class CashCorrectionEvent:
    """Correction distincte. La déclaration d'origine n'est pas réécrite."""

    original_key: str
    reason: str
    author_role: str
    author_id: int
    recorded_at: datetime
    persists: bool


@dataclass(frozen=True, slots=True)
class TillProjection:
    """États de caisse simulés. Une remise n'est pas un second paiement."""

    state: str
    patient_payment_count: int
    persists: bool


def consult_cash(
    *,
    booking_id: int,
    payment_scope: str,
    amount_due_chf: Decimal,
    beneficiary_company_id: int,
) -> CashConsultation:
    """Affiche le montant dû. Le mode espèces n'encaisse rien."""
    if payment_scope not in {"booking", "group"}:
        raise ValueError("Périmètre d'espèces inconnu.")
    if amount_due_chf <= 0 or beneficiary_company_id < 1 or booking_id < 1:
        raise ValueError("Consultation d'espèces incomplète.")
    return CashConsultation(
        booking_id=booking_id,
        payment_scope=payment_scope,
        amount_due_chf=amount_due_chf.quantize(_CENT),
        currency="CHF",
        beneficiary_company_id=beneficiary_company_id,
        mode="cash",
        confirmation_required=True,
        payment_confirmed=False,
        persists=False,
    )


def describe_cash_amounts(
    *, amount_due_chf: Decimal, amount_tendered_chf: Decimal | None
) -> CashAmounts:
    """Calcule les sommes. Un écart n'est pas un encaissement confirmé."""
    due = amount_due_chf.quantize(_CENT)
    if due <= 0:
        raise ValueError("Montant dû invalide.")
    if amount_tendered_chf is None:
        situation = "declaration_error"
        tendered = None
        change = None
        net = None
    else:
        tendered = amount_tendered_chf.quantize(_CENT)
        if tendered < 0:
            raise ValueError("Montant remis invalide.")
        if tendered == 0:
            situation = "absent"
            change = Decimal("0.00")
            net = Decimal("0.00")
        elif tendered == due:
            situation = "exact"
            change = Decimal("0.00")
            net = due
        elif tendered > due:
            situation = "surplus_not_returned"
            change = None
            net = None
        else:
            situation = "partial_forbidden"
            change = Decimal("0.00")
            net = None
    return CashAmounts(
        amount_due_chf=due,
        amount_tendered_chf=tendered,
        change_returned_chf=change,
        net_collected_chf=net,
        reconciled_amount_chf=None,
        currency="CHF",
        situation=situation,
        payment_confirmed=False,
        persists=False,
        change_policy=CHANGE_POLICY,
        partial_policy=PARTIAL_POLICY,
    )


def submit_cash_declaration(
    existing: tuple[CashDeclarationRecord, ...],
    *,
    consultation: CashConsultation,
    amounts: CashAmounts,
    driver_id: int,
    idempotency_key: str,
    recorded_at: datetime,
    explicit_correction_of: str | None = None,
) -> tuple[tuple[CashDeclarationRecord, ...], str]:
    """Ajoute une déclaration simulée. Une confirmation ne s'efface pas."""
    key = idempotency_key.strip()
    if driver_id < 1 or not key or recorded_at.tzinfo is None:
        return existing, "cash_declaration_incomplete"
    replay = next((item for item in existing if item.idempotency_key == key), None)
    if replay is not None:
        return existing, "replay"
    if amounts.situation not in _RECORDABLE:
        return existing, amounts.situation
    if explicit_correction_of is not None:
        original = next(
            (
                item
                for item in existing
                if item.idempotency_key == explicit_correction_of
            ),
            None,
        )
        if original is None or original.booking_id != consultation.booking_id:
            return existing, "missing_original_entry"
    elif any(item.booking_id == consultation.booking_id for item in existing):
        return existing, "silent_correction_refused"
    record = CashDeclarationRecord(
        booking_id=consultation.booking_id,
        company_id=consultation.beneficiary_company_id,
        driver_id=driver_id,
        amounts=amounts,
        idempotency_key=key,
        recorded_at=recorded_at,
        frozen=amounts.situation in _RECORDABLE,
        persists=False,
        supersedes_key=explicit_correction_of,
    )
    reason = "declared" if amounts.situation == "exact" else amounts.situation
    return (*existing, record), reason


def visible_cash_declarations(
    entries: tuple[CashDeclarationRecord, ...],
    *,
    reader_company_id: int,
    reader_role: str,
    reader_driver_id: int | None,
    driver_may_see_company_register: bool,
) -> tuple[CashDeclarationRecord, ...]:
    """L'entreprise A ne voit pas la caisse de l'entreprise B."""
    own = tuple(item for item in entries if item.company_id == reader_company_id)
    if reader_role == "company":
        return own
    if reader_role == "driver" and driver_may_see_company_register:
        return own
    if reader_role == "driver" and reader_driver_id is not None:
        return tuple(item for item in own if item.driver_id == reader_driver_id)
    return ()


def reconcile_cash(
    entry: CashDeclarationRecord,
    *,
    actor_role: str,
    actor_company_id: int,
    reconciled_amount_chf: Decimal,
) -> str:
    """Le rapprochement est une étape d'entreprise. Il n'écrit rien."""
    if actor_role != "company" or actor_company_id != entry.company_id:
        return "reconciliation_refused"
    if reconciled_amount_chf < 0:
        return "reconciliation_refused"
    if entry.persists:
        return "reconciliation_refused"
    return "reconciliation_prepared"


def assess_cash_settlement(
    *,
    amount_due_chf: Decimal,
    amount_tendered_chf: Decimal | None,
    change_returned_chf: Decimal | None,
    change_handed_back: bool = False,
) -> CashAmounts:
    """Applique la proposition V1. La monnaie n'est jamais devinée."""
    due = amount_due_chf.quantize(_CENT)
    if due <= 0:
        raise ValueError("Montant dû invalide.")
    if amount_tendered_chf is None:
        return describe_cash_amounts(
            amount_due_chf=due, amount_tendered_chf=None
        )
    tendered = amount_tendered_chf.quantize(_CENT)
    if tendered < 0:
        raise ValueError("Montant remis invalide.")
    if tendered < due:
        return describe_cash_amounts(
            amount_due_chf=due, amount_tendered_chf=tendered
        )
    if tendered == due:
        declared_change = (
            None
            if change_returned_chf is None
            else change_returned_chf.quantize(_CENT)
        )
        if declared_change not in (None, Decimal("0.00")):
            return CashAmounts(
                amount_due_chf=due,
                amount_tendered_chf=tendered,
                change_returned_chf=declared_change,
                net_collected_chf=None,
                reconciled_amount_chf=None,
                currency="CHF",
                situation="change_mismatch",
                payment_confirmed=False,
                persists=False,
                change_policy=CHANGE_POLICY,
                partial_policy=PARTIAL_POLICY,
            )
        return describe_cash_amounts(
            amount_due_chf=due, amount_tendered_chf=tendered
        )
    declared = (
        None
        if change_returned_chf is None
        else change_returned_chf.quantize(_CENT)
    )
    expected = (tendered - due).quantize(_CENT)
    if declared is None:
        situation = "surplus_not_returned"
        change = None
        net = None
    elif declared == expected and change_handed_back:
        situation = "full_with_explicit_change"
        change = declared
        net = due
    elif declared == expected:
        situation = "change_not_confirmed"
        change = declared
        net = None
    else:
        situation = "change_mismatch"
        change = declared
        net = None
    return CashAmounts(
        amount_due_chf=due,
        amount_tendered_chf=tendered,
        change_returned_chf=change,
        net_collected_chf=net,
        reconciled_amount_chf=None,
        currency="CHF",
        situation=situation,
        payment_confirmed=False,
        persists=False,
        change_policy=CHANGE_POLICY,
        partial_policy=PARTIAL_POLICY,
    )


def propose_cash_receipt(
    *,
    source: str,
    issuer_company_id: int,
    issuer_name: str,
    booking_reference: str,
    payment_scope: str,
    debtor_label: str,
    service_label: str,
    amounts: CashAmounts,
    service_date: str | None = None,
    received_at: datetime | None = None,
    confirmed_recorded: bool = False,
    fiscal_lines: tuple[str, ...] = (),
    fiscal_verified: bool = False,
) -> CashReceiptProposal:
    """Prépare le reçu. Ce lot ne l'émet pas et ne le transmet pas."""
    if source in _NO_RECEIPT:
        reason = source
    elif amounts.situation in _NO_RECEIPT:
        reason = amounts.situation
    elif not confirmed_recorded:
        reason = "reception_not_recorded"
    else:
        reason = "issuance_disabled"
    balance = amounts.amount_due_chf
    if amounts.net_collected_chf is not None:
        balance = (amounts.amount_due_chf - amounts.net_collected_chf).quantize(_CENT)
    return CashReceiptProposal(
        issuer_company_id=issuer_company_id,
        issuer_name=issuer_name,
        fiscal_lines=fiscal_lines if fiscal_verified else (),
        receipt_number=None,
        received_at=received_at if confirmed_recorded else None,
        booking_reference=booking_reference,
        payment_scope=payment_scope,
        service_date=service_date,
        debtor_label=debtor_label,
        service_label=service_label,
        amount_due_chf=amounts.amount_due_chf,
        amount_tendered_chf=amounts.amount_tendered_chf,
        change_returned_chf=amounts.change_returned_chf,
        net_collected_chf=amounts.net_collected_chf,
        balance_chf=balance,
        till_reference=None,
        statement=RECEIPT_TEXT,
        statement_status=RECEIPT_TEXT_STATUS,
        issued=False,
        transmitted=False,
        duplicate=False,
        replaces_attestation=False,
        replaces_invoice=False,
        generation_allowed=False,
        blocked_reason=reason,
        medical_details_included=False,
    )


def reprint_cash_receipt(proposal: CashReceiptProposal) -> CashReceiptProposal:
    """Duplicata du même numéro. Aucune nouvelle écriture de caisse."""
    return CashReceiptProposal(
        issuer_company_id=proposal.issuer_company_id,
        issuer_name=proposal.issuer_name,
        fiscal_lines=proposal.fiscal_lines,
        receipt_number=proposal.receipt_number,
        received_at=proposal.received_at,
        booking_reference=proposal.booking_reference,
        payment_scope=proposal.payment_scope,
        service_date=proposal.service_date,
        debtor_label=proposal.debtor_label,
        service_label=proposal.service_label,
        amount_due_chf=proposal.amount_due_chf,
        amount_tendered_chf=proposal.amount_tendered_chf,
        change_returned_chf=proposal.change_returned_chf,
        net_collected_chf=proposal.net_collected_chf,
        balance_chf=proposal.balance_chf,
        till_reference=proposal.till_reference,
        statement=proposal.statement,
        statement_status=proposal.statement_status,
        issued=False,
        transmitted=False,
        duplicate=True,
        replaces_attestation=False,
        replaces_invoice=False,
        generation_allowed=False,
        blocked_reason=proposal.blocked_reason,
        medical_details_included=False,
    )


_TILL_STATES = frozenset(
    {
        "declaration_prepared",
        "cash_confirmed_received",
        "awaiting_company_remittance",
        "remittance_recorded",
        "reconciled",
        "variance_to_check",
        "correction_or_dispute",
    }
)


def project_till(stage: str) -> TillProjection:
    """Projette l'état. La remise à l'entreprise ne recrée pas un paiement."""
    if stage not in _TILL_STATES:
        raise ValueError("État de caisse inconnu.")
    collected = stage != "declaration_prepared"
    return TillProjection(
        state=stage,
        patient_payment_count=1 if collected else 0,
        persists=False,
    )


def audit_cash_correction(
    entry: CashDeclarationRecord,
    *,
    reason: str,
    author_role: str,
    author_id: int,
    recorded_at: datetime,
) -> tuple[CashCorrectionEvent | None, str]:
    """Ajoute un événement. Le montant confirmé d'origine reste identique."""
    if not entry.frozen:
        return None, "not_confirmed"
    if author_role != "company" or author_id < 1 or recorded_at.tzinfo is None:
        return None, "author_not_authorized"
    motive = reason.strip()
    if not motive:
        return None, "reason_required"
    return (
        CashCorrectionEvent(
            original_key=entry.idempotency_key,
            reason=motive,
            author_role=author_role,
            author_id=author_id,
            recorded_at=recorded_at,
            persists=False,
        ),
        "correction_prepared",
    )


def cash_receipt_draft(entry: CashDeclarationRecord) -> CashReceiptDraft:
    """Le patient ne reçoit pas encore de justificatif."""
    return CashReceiptDraft(
        booking_id=entry.booking_id,
        company_id=entry.company_id,
        amount_received_chf=entry.amounts.net_collected_chf,
        currency="CHF",
        issued=False,
        bank_transfer=False,
        platform_fee=False,
    )
