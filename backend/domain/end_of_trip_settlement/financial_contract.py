"""Contrat financier pur.

Aucune base, aucun réseau, aucun effet de bord. Les centimes sont des
``Decimal``. Ce module ne crée pas de paiement négatif pour un avoir.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from domain.end_of_trip_settlement.cash_settlement import money

_ZERO = Decimal("0.00")
_CENT = Decimal("0.01")

KIND_COLLECTION = "collection"
KIND_COMPENSATION = "compensation"
KIND_ALLOCATION = "allocation"
KIND_CREDIT_NOTE = "credit_note"
KIND_REFUND = "refund"
KIND_HANDOVER = "handover"

OPERATION_KINDS = (
    KIND_COLLECTION,
    KIND_COMPENSATION,
    KIND_ALLOCATION,
    KIND_CREDIT_NOTE,
    KIND_REFUND,
    KIND_HANDOVER,
)

_NOT_IMPUTABLE_DISPOSITIONS = frozenset({"contested", "held", "returned"})


@dataclass(frozen=True, slots=True)
class Settlement:
    """Créance, encaissements nets, solde dû et trop-perçu."""

    claim_chf: Decimal
    net_paid_chf: Decimal
    balance_due_chf: Decimal
    refundable_chf: Decimal


@dataclass(frozen=True, slots=True)
class CashLine:
    """Ligne d'espèces déjà qualifiée. Le net d'une compensation est négatif."""

    kind: str
    state: str
    disposition: str | None
    net_chf: Decimal
    due_chf: Decimal


@dataclass(frozen=True, slots=True)
class CashOperation:
    """Opération identifiée. Le même identifiant ne compte qu'une fois."""

    operation_id: str
    kind: str
    state: str
    disposition: str | None
    net_chf: Decimal
    due_chf: Decimal
    leg_id: str


@dataclass(frozen=True, slots=True)
class LegAmount:
    """Montant contractuel d'un trajet. Il n'est pas une moitié d'un total."""

    leg_id: str
    contractual_chf: Decimal


@dataclass(frozen=True, slots=True)
class CashCustody:
    """Détention des espèces. La remise ne change pas le dû patient."""

    contractual_chf: Decimal
    patient_paid_chf: Decimal
    driver_held_chf: Decimal
    company_held_chf: Decimal


def imputable(
    *,
    state: str,
    disposition: str | None,
    net: Decimal,
    due: Decimal,
) -> bool:
    """Vrai seulement pour un encaissement confirmé qui peut réduire le solde.

    ``received`` avec un net égal au dû de l'opération, ou ``partial`` avec
    un net positif. Incertain, contesté, retenu ou restitué : non imputable.
    """
    net_q = money(net)
    due_q = money(due)
    if state in {"uncertain", "returned"}:
        return False
    if disposition in _NOT_IMPUTABLE_DISPOSITIONS:
        return False
    if disposition == "partial":
        return net_q > _ZERO
    if state == "received":
        return net_q == due_q and net_q > _ZERO
    return False


def recognized_net(lines: Sequence[CashLine]) -> Decimal:
    """Somme des nets imputables et des compensations. N'écrit rien."""
    total = _ZERO
    for line in lines:
        net = money(line.net_chf)
        if line.kind == KIND_COMPENSATION:
            if net >= _ZERO:
                raise ValueError("montant_incoherent")
            total += net
            continue
        if line.kind != KIND_COLLECTION:
            continue
        if imputable(
            state=line.state,
            disposition=line.disposition,
            net=net,
            due=money(line.due_chf),
        ):
            if net < _ZERO:
                raise ValueError("montant_incoherent")
            total += net
    return total.quantize(_CENT)


def within_cap(recognized: Decimal, due: Decimal) -> bool:
    """Le net reconnu reste entre zéro et le dû contractuel du trajet."""
    recognized_q = money(recognized)
    due_q = money(due)
    if due_q <= _ZERO:
        raise ValueError("montant_incoherent")
    return _ZERO <= recognized_q <= due_q


def _require_cap(recognized: Decimal, due: Decimal) -> None:
    if not within_cap(recognized, due):
        raise ValueError("depassement_autorise")


def accumulate_cash(
    due_chf: Decimal,
    operations: Sequence[CashOperation],
) -> Decimal:
    """Applique les opérations d'un trajet. Un identifiant déjà vu est un rejeu.

    Une remise ne change pas le net patient. Une collection non imputable
    ne le change pas non plus. Un dépassement ou un net négatif injustifié
    est refusé.
    """
    due = money(due_chf)
    seen: dict[str, Decimal] = {}
    recognized = _ZERO
    for operation in operations:
        if operation.kind not in OPERATION_KINDS:
            raise ValueError("montant_incoherent")
        net = money(operation.net_chf)
        previous = seen.get(operation.operation_id)
        if previous is not None:
            if previous != net:
                raise ValueError("montant_incoherent")
            continue
        seen[operation.operation_id] = net
        if operation.kind == KIND_HANDOVER:
            if net <= _ZERO:
                raise ValueError("montant_incoherent")
            continue
        if operation.kind == KIND_COMPENSATION:
            if net >= _ZERO:
                raise ValueError("montant_incoherent")
            candidate = (recognized + net).quantize(_CENT)
        elif operation.kind == KIND_COLLECTION:
            if net < _ZERO:
                raise ValueError("montant_incoherent")
            if not imputable(
                state=operation.state,
                disposition=operation.disposition,
                net=net,
                due=money(operation.due_chf),
            ):
                continue
            candidate = (recognized + net).quantize(_CENT)
        else:
            raise ValueError("montant_incoherent")
        _require_cap(candidate, due)
        recognized = candidate
    return recognized


def credit_room(total: Decimal, credit_notes: Sequence[Decimal]) -> Decimal:
    """Facturé encore annulable. Le plafond est le total, pas le payé."""
    total_q = money(total)
    if total_q <= _ZERO:
        raise ValueError("montant_incoherent")
    used = _ZERO
    for note in credit_notes:
        amount = money(note)
        if amount <= _ZERO:
            raise ValueError("montant_incoherent")
        used += amount
    used = used.quantize(_CENT)
    if used > total_q:
        raise ValueError("avoir_au_dela_du_facture")
    return (total_q - used).quantize(_CENT)


def _positive_sum(amounts: Sequence[Decimal], *, label: str) -> Decimal:
    total = _ZERO
    for raw in amounts:
        amount = money(raw)
        if amount < _ZERO:
            raise ValueError("montant_incoherent")
        if amount == _ZERO and label == "remboursement":
            raise ValueError("montant_incoherent")
        total += amount
    return total.quantize(_CENT)


def settle(
    total: Decimal,
    credit_notes: Sequence[Decimal],
    payments: Sequence[Decimal],
    refunds: Sequence[Decimal],
) -> Settlement:
    """Calcule la créance, le solde et le trop-perçu.

    Un avoir réduit la créance. Il n'est pas un paiement. Un remboursement
    ne peut pas dépasser le trop-perçu admissible.
    """
    room_before = credit_room(total, ())
    used_credits = (room_before - credit_room(total, credit_notes)).quantize(_CENT)
    claim = (room_before - used_credits).quantize(_CENT)
    paid = _positive_sum(payments, label="paiement")
    returned = _positive_sum(refunds, label="remboursement")
    admissible = max(_ZERO, (paid - claim).quantize(_CENT))
    if returned > admissible:
        raise ValueError("depassement_autorise")
    net_paid = (paid - returned).quantize(_CENT)
    if net_paid >= claim:
        balance = _ZERO
        refundable = (net_paid - claim).quantize(_CENT)
    else:
        balance = (claim - net_paid).quantize(_CENT)
        refundable = _ZERO
    return Settlement(
        claim_chf=claim,
        net_paid_chf=net_paid,
        balance_due_chf=balance,
        refundable_chf=refundable,
    )


def apply_handover(custody: CashCustody, amount: Decimal) -> CashCustody:
    """Transfère des espèces du chauffeur vers l'entreprise.

    Le montant contractuel et le paiement patient restent identiques.
    """
    moved = money(amount)
    if moved <= _ZERO:
        raise ValueError("montant_incoherent")
    contractual = money(custody.contractual_chf)
    patient_paid = money(custody.patient_paid_chf)
    driver_held = money(custody.driver_held_chf)
    company_held = money(custody.company_held_chf)
    if moved > driver_held:
        raise ValueError("depassement_autorise")
    return CashCustody(
        contractual_chf=contractual,
        patient_paid_chf=patient_paid,
        driver_held_chf=(driver_held - moved).quantize(_CENT),
        company_held_chf=(company_held + moved).quantize(_CENT),
    )


def settle_legs(
    legs: Sequence[LegAmount],
    payments_by_leg: Mapping[str, Sequence[Decimal]],
) -> dict[str, Settlement]:
    """Règle chaque trajet avec son propre montant. Aucun partage par moitié."""
    if not legs:
        raise ValueError("montant_incoherent")
    known = {leg.leg_id for leg in legs}
    if len(known) != len(legs):
        raise ValueError("montant_incoherent")
    unknown = set(payments_by_leg) - known
    if unknown:
        raise ValueError("montant_incoherent")
    settled: dict[str, Settlement] = {}
    for leg in legs:
        settled[leg.leg_id] = settle(
            leg.contractual_chf,
            (),
            payments_by_leg.get(leg.leg_id, ()),
            (),
        )
    return settled
