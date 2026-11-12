"""Règlement d'un aller-retour. Une seule demande couvre le total.

Le paiement par trajet et le partage entre entreprises ne sont pas activés.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class GroupCollectionPlan:
    total_chf: Decimal
    already_paid_chf: Decimal
    amount_still_due_chf: Decimal
    collection_count: int
    second_request: bool
    partial_leg_allowed: bool
    split_across_companies: bool
    activated: bool
    reason: str
    receipt_scope: str
    invoice_scope: str
    driver_may_split: bool
    decision_stale: bool
    collection_moment_status: str


def plan_round_trip(
    *,
    outbound_chf: Decimal,
    return_chf: Decimal,
    already_paid_chf: Decimal,
    separate_leg_approved: bool,
    group_revision: int = 1,
    decision_revision: int = 1,
) -> GroupCollectionPlan:
    """Prépare un règlement groupé. Il n'est pas exécuté."""
    if outbound_chf <= 0 or return_chf <= 0 or already_paid_chf < 0:
        raise ValueError("Montants d'aller-retour invalides.")
    total = (outbound_chf + return_chf).quantize(_CENT)
    paid = already_paid_chf.quantize(_CENT)
    stale = group_revision != decision_revision
    if paid >= total:
        return GroupCollectionPlan(
            total_chf=total,
            already_paid_chf=paid,
            amount_still_due_chf=Decimal("0.00"),
            collection_count=0,
            second_request=False,
            partial_leg_allowed=False,
            split_across_companies=False,
            activated=False,
            reason="already_paid",
            receipt_scope="group",
            invoice_scope="outbound_and_return",
            driver_may_split=False,
            decision_stale=stale,
            collection_moment_status="pending_approval",
        )
    if stale:
        return GroupCollectionPlan(
            total_chf=total,
            already_paid_chf=paid,
            amount_still_due_chf=(total - paid).quantize(_CENT),
            collection_count=0,
            second_request=False,
            partial_leg_allowed=False,
            split_across_companies=False,
            activated=False,
            reason="group_decision_stale",
            receipt_scope="group",
            invoice_scope="outbound_and_return",
            driver_may_split=False,
            decision_stale=True,
            collection_moment_status="pending_approval",
        )
    if separate_leg_approved:
        return GroupCollectionPlan(
            total_chf=total,
            already_paid_chf=paid,
            amount_still_due_chf=(total - paid).quantize(_CENT),
            collection_count=1,
            second_request=False,
            partial_leg_allowed=False,
            split_across_companies=False,
            activated=False,
            reason="separate_leg_not_activated",
            receipt_scope="group",
            invoice_scope="outbound_and_return",
            driver_may_split=False,
            decision_stale=False,
            collection_moment_status="pending_approval",
        )
    return GroupCollectionPlan(
        total_chf=total,
        already_paid_chf=paid,
        amount_still_due_chf=(total - paid).quantize(_CENT),
        collection_count=1,
        second_request=False,
        partial_leg_allowed=False,
        split_across_companies=False,
        activated=False,
        reason="single_group_collection",
        receipt_scope="group",
        invoice_scope="outbound_and_return",
        driver_may_split=False,
        decision_stale=False,
        collection_moment_status="pending_approval",
    )
