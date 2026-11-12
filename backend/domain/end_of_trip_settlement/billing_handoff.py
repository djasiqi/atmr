"""Raccord conceptuel à la facturation existante. Aucune facture n'est émise.

Les états ne se franchissent pas tout seuls. La signature n'émet pas
la facture et ne la marque pas payée. Les QR-factures déjà produites
restent celles du moteur de facturation en place.
"""

from __future__ import annotations

from dataclasses import dataclass

STATES = (
    "billing_request",
    "signature_recorded",
    "invoice_prepared",
    "invoice_issued",
    "invoice_paid",
)

_ORDER = {name: index for index, name in enumerate(STATES)}


@dataclass(frozen=True, slots=True)
class BillingHandoff:
    """Demande préparée pour le moteur existant. Elle n'écrit rien."""

    state: str
    scope_key: str
    invoice_issued: bool
    invoice_paid: bool
    signature_means_issued: bool
    signature_means_paid: bool
    second_invoice: bool
    persists: bool
    reason: str


def open_billing_request(*, scope_key: str, already_issued: bool) -> BillingHandoff:
    """Ouvre une demande. Un périmètre déjà facturé n'en reçoit pas une seconde."""
    if already_issued:
        return BillingHandoff(
            state="invoice_issued",
            scope_key=scope_key,
            invoice_issued=True,
            invoice_paid=False,
            signature_means_issued=False,
            signature_means_paid=False,
            second_invoice=False,
            persists=False,
            reason="invoice_already_issued",
        )
    return BillingHandoff(
        state="billing_request",
        scope_key=scope_key,
        invoice_issued=False,
        invoice_paid=False,
        signature_means_issued=False,
        signature_means_paid=False,
        second_invoice=False,
        persists=False,
        reason="billing_request_prepared",
    )


def advance_billing_state(
    current: BillingHandoff, target: str, *, explicit: bool
) -> BillingHandoff:
    """Un changement d'état exige une décision explicite."""
    if target not in _ORDER or not explicit:
        return _copy(current, reason="implicit_transition_refused")
    if current.reason == "invoice_already_issued":
        return _copy(current, reason="invoice_already_issued")
    if _ORDER[target] != _ORDER[current.state] + 1:
        return _copy(current, reason="state_skip_refused")
    issued = target in {"invoice_issued", "invoice_paid"}
    paid = target == "invoice_paid"
    return BillingHandoff(
        state=target,
        scope_key=current.scope_key,
        invoice_issued=issued,
        invoice_paid=paid,
        signature_means_issued=False,
        signature_means_paid=False,
        second_invoice=False,
        persists=False,
        reason="state_prepared",
    )


def _copy(current: BillingHandoff, *, reason: str) -> BillingHandoff:
    return BillingHandoff(
        state=current.state,
        scope_key=current.scope_key,
        invoice_issued=current.invoice_issued,
        invoice_paid=current.invoice_paid,
        signature_means_issued=False,
        signature_means_paid=False,
        second_invoice=False,
        persists=False,
        reason=reason,
    )
