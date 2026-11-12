"""Espèces insuffisantes ou retenues. Détenir n'est pas encaisser.

La politique de paiement partiel ne change pas. Aucune ligne n'est écrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_CENT = Decimal("0.01")
_HELD = "held"
_RETURNED = "returned"


@dataclass(frozen=True, slots=True)
class CashCustody:
    """Trace d'espèces. Le solde dû n'est pas réduit."""

    amount_due_chf: Decimal
    tendered_chf: Decimal
    disposition: str
    physically_held_chf: Decimal
    officially_accepted_chf: Decimal
    balance_still_due_chf: Decimal
    payment_accepted: bool
    partial_policy_changed: bool
    receipt_issued: bool
    custody_traced: bool
    custody_reference: str | None
    administrative: bool
    mission_close_blocked: bool
    restitution_possible: bool
    persists: bool
    reason: str


@dataclass(frozen=True, slots=True)
class CashReceiptDemo:
    """Rendu de démonstration. Aucun reçu n'est émis."""

    banner: str
    issuer_name: str
    proposed_reference: str | None
    reference_allocated: bool
    service_date: str
    scope_label: str
    service_label: str
    amount_due_chf: Decimal
    amount_tendered_chf: Decimal | None
    change_returned_chf: Decimal | None
    net_collected_chf: Decimal | None
    till_reference: str | None
    cash_mention: str
    fiscal_identifier: None
    issued: bool
    transmitted: bool
    duplicate: bool
    collection_writes: int
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ReconnectDecision:
    """Une déclaration locale ne devient pas un paiement au retour du réseau."""

    payment_confirmed: bool
    queued_collection: bool
    administrative: bool
    mission_preserved: bool
    persists: bool


def trace_insufficient_cash(
    *,
    amount_due_chf: Decimal,
    tendered_chf: Decimal,
    disposition: str | None,
    idempotency_key: str,
) -> CashCustody:
    """30 CHF gardés restent une détention. Ils ne soldent pas 45 CHF."""
    due = amount_due_chf.quantize(_CENT)
    tendered = tendered_chf.quantize(_CENT)
    if due <= 0 or tendered < 0:
        raise ValueError("Montant insuffisant invalide.")
    if tendered >= due:
        raise ValueError("Ce n'est pas un montant insuffisant.")
    key = idempotency_key.strip()
    if tendered == 0:
        return _custody(
            due=due,
            tendered=tendered,
            disposition="none",
            held=Decimal("0.00"),
            traced=False,
            reference=None,
            restitution=False,
            reason="nothing_remitted",
        )
    if disposition not in {_RETURNED, _HELD}:
        return _custody(
            due=due,
            tendered=tendered,
            disposition="undeclared",
            held=Decimal("0.00"),
            traced=False,
            reference=None,
            restitution=False,
            reason="disposition_required",
        )
    if disposition == _RETURNED:
        return _custody(
            due=due,
            tendered=tendered,
            disposition=_RETURNED,
            held=Decimal("0.00"),
            traced=False,
            reference=None,
            restitution=False,
            reason="cash_returned",
        )
    if not key:
        raise ValueError("Référence de détention requise.")
    return _custody(
        due=due,
        tendered=tendered,
        disposition=_HELD,
        held=tendered,
        traced=True,
        reference=key,
        restitution=True,
        reason="cash_held_not_settlement",
    )


def _custody(
    *,
    due: Decimal,
    tendered: Decimal,
    disposition: str,
    held: Decimal,
    traced: bool,
    reference: str | None,
    restitution: bool,
    reason: str,
) -> CashCustody:
    return CashCustody(
        amount_due_chf=due,
        tendered_chf=tendered,
        disposition=disposition,
        physically_held_chf=held,
        officially_accepted_chf=Decimal("0.00"),
        balance_still_due_chf=due,
        payment_accepted=False,
        partial_policy_changed=False,
        receipt_issued=False,
        custody_traced=traced,
        custody_reference=reference,
        administrative=True,
        mission_close_blocked=False,
        restitution_possible=restitution,
        persists=False,
        reason=reason,
    )


def demonstrate_cash_receipt(
    *,
    situation: str,
    issuer_name: str,
    idempotency_key: str,
    scope_label: str,
    service_label: str,
    service_date: str,
    amount_due_chf: Decimal,
    amount_tendered_chf: Decimal | None,
    change_returned_chf: Decimal | None,
    net_collected_chf: Decimal | None,
) -> CashReceiptDemo:
    """Montre le reçu. Il n'est pas numéroté pour de bon et n'est pas envoyé."""
    key = idempotency_key.strip()
    qualified = situation in {"exact", "full_with_explicit_change"} and bool(key)
    return CashReceiptDemo(
        banner="DRAFT — PENDING APPROVAL",
        issuer_name=issuer_name,
        proposed_reference=f"DEMO-{key}" if qualified else None,
        reference_allocated=False,
        service_date=service_date,
        scope_label=scope_label,
        service_label=service_label,
        amount_due_chf=amount_due_chf.quantize(_CENT),
        amount_tendered_chf=amount_tendered_chf,
        change_returned_chf=change_returned_chf,
        net_collected_chf=net_collected_chf if qualified else None,
        till_reference=None,
        cash_mention="Reçu d'espèces — document non émis",
        fiscal_identifier=None,
        issued=False,
        transmitted=False,
        duplicate=False,
        collection_writes=0,
        idempotency_key=key,
    )


def reprint_demo(demo: CashReceiptDemo) -> CashReceiptDemo:
    """Même référence proposée. Aucune seconde écriture."""
    return CashReceiptDemo(
        banner=demo.banner,
        issuer_name=demo.issuer_name,
        proposed_reference=demo.proposed_reference,
        reference_allocated=False,
        service_date=demo.service_date,
        scope_label=demo.scope_label,
        service_label=demo.service_label,
        amount_due_chf=demo.amount_due_chf,
        amount_tendered_chf=demo.amount_tendered_chf,
        change_returned_chf=demo.change_returned_chf,
        net_collected_chf=demo.net_collected_chf,
        till_reference=None,
        cash_mention=demo.cash_mention,
        fiscal_identifier=None,
        issued=False,
        transmitted=False,
        duplicate=True,
        collection_writes=0,
        idempotency_key=demo.idempotency_key,
    )


def reconnect_local_declaration(
    *,
    local_marked_confirmed: bool,
    server_payment_exists: bool,
    invoice_exists: bool,
    amount_changed: bool,
) -> ReconnectDecision:
    """Le doute reste côté entreprise. Rien n'est mis en file d'encaissement."""
    doubtful = (
        local_marked_confirmed
        or server_payment_exists
        or invoice_exists
        or amount_changed
    )
    return ReconnectDecision(
        payment_confirmed=False,
        queued_collection=False,
        administrative=doubtful,
        mission_preserved=True,
        persists=False,
    )


@dataclass(frozen=True, slots=True)
class HeldResolution:
    """Sort proposé d'une détention. Rien n'est écrit en comptabilité."""

    action: str
    physically_held_chf: Decimal
    officially_accepted_chf: Decimal
    balance_still_due_chf: Decimal
    proposed_balance_if_approved_chf: Decimal | None
    payment_accepted: bool
    custody_open: bool
    custody_reference: str
    holder: str
    returned: bool
    imputation_requested: bool
    imputation_applied: bool
    disputed: bool
    remitted_to_company: bool
    company_id: int
    patient_settlement: bool
    reconciliation_counts_as_payment: bool
    persists: bool
    reason: str
    policy_status: str
    idempotency_key: str


def resolve_held_cash(
    custody: CashCustody,
    prior: tuple[HeldResolution, ...],
    *,
    action: str,
    actor_role: str,
    actor_company_id: int,
    beneficiary_company_id: int,
    idempotency_key: str,
) -> tuple[tuple[HeldResolution, ...], HeldResolution]:
    """Propose un sort. L'imputation n'est pas appliquée : elle attend une règle."""
    key = idempotency_key.strip()
    replay = next(
        (
            item
            for item in prior
            if item.idempotency_key == key and key and item.action == action
        ),
        None,
    )
    if replay is not None:
        return prior, replay
    if actor_company_id != beneficiary_company_id:
        return prior, _refused(custody, action, beneficiary_company_id, "company_isolation_refused")
    if custody.reason != "cash_held_not_settlement" or not custody.custody_reference:
        return prior, _refused(custody, action, beneficiary_company_id, "custody_missing")
    if actor_role not in {"company", "driver"}:
        return prior, _refused(custody, action, beneficiary_company_id, "author_not_authorized")
    returned = any(item.returned for item in prior)
    requested = any(item.imputation_requested for item in prior)
    if action == "return":
        if actor_role not in {"company", "driver"}:
            return prior, _refused(custody, action, beneficiary_company_id, "author_not_authorized")
        if returned:
            return prior, _refused(custody, action, beneficiary_company_id, "double_return_refused")
        if requested:
            return prior, _refused(custody, action, beneficiary_company_id, "return_and_imputation_conflict")
        decision = _decision(
            custody,
            action="return",
            held=Decimal("0.00"),
            proposed=None,
            custody_open=False,
            holder="patient",
            returned=True,
            reason="cash_returned_to_patient",
            company_id=beneficiary_company_id,
            idempotency_key=key,
        )
        return (*prior, decision), decision
    if action == "impute":
        if actor_role != "company":
            return prior, _refused(custody, action, beneficiary_company_id, "author_not_authorized")
        if returned:
            return prior, _refused(custody, action, beneficiary_company_id, "return_and_imputation_conflict")
        if requested:
            return prior, _refused(custody, action, beneficiary_company_id, "double_imputation_refused")
        decision = _decision(
            custody,
            action="impute",
            held=custody.physically_held_chf,
            proposed=(custody.amount_due_chf - custody.physically_held_chf).quantize(_CENT),
            custody_open=True,
            holder="driver",
            imputation_requested=True,
            reason="imputation_pending_approval",
            company_id=beneficiary_company_id,
            idempotency_key=key,
        )
        return (*prior, decision), decision
    if action == "dispute":
        if actor_role != "company":
            return prior, _refused(custody, action, beneficiary_company_id, "author_not_authorized")
        decision = _decision(
            custody,
            action="dispute",
            held=custody.physically_held_chf,
            proposed=None,
            custody_open=True,
            holder="driver",
            disputed=True,
            reason="patient_disputes_custody",
            company_id=beneficiary_company_id,
            idempotency_key=key,
        )
        return (*prior, decision), decision
    if action == "remit":
        if actor_role != "driver":
            return prior, _refused(custody, action, beneficiary_company_id, "author_not_authorized")
        if any(item.remitted_to_company for item in prior):
            return prior, _refused(custody, action, beneficiary_company_id, "double_remittance_refused")
        decision = _decision(
            custody,
            action="remit",
            held=custody.physically_held_chf,
            proposed=None,
            custody_open=True,
            holder="company",
            remitted=True,
            reason="remitted_not_a_settlement",
            company_id=beneficiary_company_id,
            idempotency_key=key,
        )
        return (*prior, decision), decision
    if action == "close":
        return prior, _refused(custody, action, beneficiary_company_id, "misleading_close_refused")
    return prior, _refused(custody, action, beneficiary_company_id, "action_refused")


def _decision(
    custody: CashCustody,
    *,
    action: str,
    held: Decimal,
    proposed: Decimal | None,
    custody_open: bool,
    holder: str,
    reason: str,
    company_id: int,
    idempotency_key: str = "",
    returned: bool = False,
    imputation_requested: bool = False,
    disputed: bool = False,
    remitted: bool = False,
) -> HeldResolution:
    return HeldResolution(
        action=action,
        physically_held_chf=held,
        officially_accepted_chf=Decimal("0.00"),
        balance_still_due_chf=custody.amount_due_chf,
        proposed_balance_if_approved_chf=proposed,
        payment_accepted=False,
        custody_open=custody_open,
        custody_reference=custody.custody_reference or "",
        holder=holder,
        returned=returned,
        imputation_requested=imputation_requested,
        imputation_applied=False,
        disputed=disputed,
        remitted_to_company=remitted,
        company_id=company_id,
        patient_settlement=False,
        reconciliation_counts_as_payment=False,
        persists=False,
        reason=reason,
        policy_status="pending_approval",
        idempotency_key=idempotency_key,
    )


def _refused(
    custody: CashCustody,
    action: str,
    company_id: int,
    reason: str,
) -> HeldResolution:
    return _decision(
        custody,
        action=action,
        held=custody.physically_held_chf,
        proposed=None,
        custody_open=custody.reason == "cash_held_not_settlement",
        holder="driver",
        reason=reason,
        company_id=company_id,
    )
