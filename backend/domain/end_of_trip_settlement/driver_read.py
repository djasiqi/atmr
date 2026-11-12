"""Lecture chauffeur. Aucun débit, aucune espèce, aucune facture.

Le téléphone affiche ce contrat. Il ne recalcule ni le payeur, ni le
montant, ni les moyens. Tant que l'encaissement produit est coupé, une
décision favorable du moteur n'est pas une autorisation d'exécution.
"""

from __future__ import annotations

from decimal import Decimal

from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from domain.end_of_trip_settlement.eligibility import (
    SettlementSnapshot,
    decision_contract,
    evaluate,
)

_COLLECTABLE = frozenset({"collection_authorized", "invoice_authorized"})
_KINDS = frozenset(
    {
        "already_paid",
        "collection_authorized",
        "invoice_authorized",
        "no_collection",
        "collection_forbidden",
        "administrative_review",
    }
)
_MODES = ("tap_to_pay", "cash", "invoice")
_PREVIEW_SCREENS = frozenset(
    {
        "summary",
        "tap-prepare",
        "tap-progress",
        "tap-accepted",
        "tap-refused",
        "tap-uncertain",
        "tap-interrupted",
        "tap-already",
        "cash",
        "cash-difference",
        "invoice",
        "invoice-unable",
        "invoice-refused",
        "invoice-representative",
        "invoice-technical",
    }
)


def read_driver_settlement(
    snapshot: SettlementSnapshot,
    *,
    attestation_recorded: bool = False,
) -> dict[str, object]:
    """Construit la lecture d'une réservation. N'écrit rien."""
    decision = evaluate(snapshot)
    contract = decision_contract(decision, snapshot)
    if not END_OF_TRIP_COLLECTION_ENABLED and contract["kind"] in _COLLECTABLE:
        contract["kind"] = "administrative_review"
        contract["reason"] = "collection_disabled"
        contract["modes"] = []
        contract["administrative_review"] = True
        contract["signature_required"] = False
    contract["modes"] = [] if not END_OF_TRIP_COLLECTION_ENABLED else contract["modes"]
    contract.update(_safety(simulation=False))
    contract["attestation_recorded"] = attestation_recorded
    contract["debtor_label"] = snapshot.debtor_display_name
    contract["document_approved"] = False
    contract["document_version"] = None
    return contract


def read_for_assigned_driver(
    snapshot: SettlementSnapshot,
    *,
    assigned: bool,
    same_company: bool,
    attestation_recorded: bool = False,
) -> dict[str, object]:
    """Refuse un chauffeur non affecté ou d'une autre entreprise."""
    if not assigned:
        return _denied("driver_unassigned")
    if not same_company:
        return _denied("other_company")
    return read_driver_settlement(
        snapshot, attestation_recorded=attestation_recorded
    )


def invalidate_displayed_amount(
    contract: dict[str, object],
    *,
    displayed_amount: str,
    current_amount: str,
) -> dict[str, object]:
    """Un montant changé retire l'écran. Ce n'est pas une autorisation durable."""
    if displayed_amount == current_amount:
        return contract
    fresh = dict(contract)
    fresh["kind"] = "administrative_review"
    fresh["reason"] = "amount_changed"
    fresh["modes"] = []
    fresh["administrative_review"] = True
    fresh["stale"] = True
    fresh["payment_executable"] = False
    fresh["amount_chf"] = current_amount
    return fresh


def simulated_driver_contract(
    *,
    kind: str,
    modes: tuple[str, ...] = (),
    amount_chf: str | None = None,
    reason: str = "simulation",
    payment_scope: str | None = "booking",
    debtor_label: str | None = "Patient fictif",
    company_id: int = 10,
    document_approved: bool = False,
    document_version: str | None = None,
    signature_required: bool = False,
    preview_screen: str | None = None,
) -> dict[str, object]:
    """Fixture d'écran. Jamais une réservation réelle, jamais un paiement."""
    if kind not in _KINDS:
        raise ValueError("Situation simulée inconnue.")
    unknown = set(modes) - set(_MODES)
    if unknown:
        raise ValueError("Mode simulé inconnu.")
    ordered = [mode for mode in _MODES if mode in modes]
    if amount_chf is not None:
        parsed = Decimal(amount_chf)
        if parsed <= 0:
            raise ValueError("Montant simulé invalide.")
        amount_chf = format(parsed.quantize(Decimal("0.01")), "f")
    if preview_screen is not None and preview_screen not in _PREVIEW_SCREENS:
        raise ValueError("Écran simulé inconnu.")
    contract = {
        "kind": kind,
        "reason": reason,
        "modes": ordered,
        "amount_chf": amount_chf,
        "currency": "CHF",
        "payment_scope": payment_scope,
        "collection_moment": None,
        "company_id": company_id,
        "signature_required": signature_required,
        "administrative_review": kind == "administrative_review",
        "payment_received": kind == "already_paid",
        "invoice_means_paid": False,
        "price_computed_locally": False,
        "payer_computed_locally": False,
        "source": "simulation",
        "reconstruct_locally": False,
        "debtor_label": debtor_label,
        "document_approved": document_approved,
        "document_version": document_version,
        "attestation_recorded": False,
        **_safety(simulation=True),
    }
    contract["preview_screen"] = preview_screen
    return contract


def _safety(*, simulation: bool) -> dict[str, object]:
    return {
        "simulation": simulation,
        "payment_executable": False,
        "persists_payment": False,
        "provider": "simulation" if simulation else "none",
        "stale": False,
        "authorization_durable": False,
        "recheck_required": True,
        "attestation_separate": True,
        "completion_independent": True,
        "driver_may_edit_amount": False,
        "driver_may_edit_payer": False,
        "writes_completed_payment": False,
        "writes_cash_receipt": False,
        "writes_invoice": False,
        "writes_financial_signature": False,
        "preview_screen": None,
    }


def _denied(reason: str) -> dict[str, object]:
    return {
        "kind": "collection_forbidden",
        "reason": reason,
        "modes": [],
        "amount_chf": None,
        "currency": "CHF",
        "payment_scope": None,
        "collection_moment": None,
        "company_id": None,
        "signature_required": False,
        "administrative_review": False,
        "payment_received": False,
        "invoice_means_paid": False,
        "price_computed_locally": False,
        "payer_computed_locally": False,
        "source": "backend",
        "reconstruct_locally": False,
        "debtor_label": None,
        "document_approved": False,
        "document_version": None,
        "attestation_recorded": False,
        **_safety(simulation=False),
    }
