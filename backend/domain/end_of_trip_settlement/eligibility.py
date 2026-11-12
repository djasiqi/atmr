"""Décision d'encaissement en fin de transport.

Fonction pure. Elle ne lit pas la base, ne modifie pas Saferpay et ne
déduit jamais le payeur de la destination du trajet.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from domain.end_of_trip_settlement.activation import (
    END_OF_TRIP_COLLECTION_ENABLED,
)

OUTCOME_NO_COLLECTION = "no_collection"
OUTCOME_ALREADY_PAID = "already_paid"
OUTCOME_ON_SITE = "on_site_authorized"
OUTCOME_PATIENT_INVOICE = "patient_invoice_authorized"
OUTCOME_INSTITUTION = "institution_billed"
OUTCOME_ADMIN = "administrative_decision_required"
OUTCOME_AMOUNT = "amount_unvalidated_blocked"

_COLLECT = frozenset({OUTCOME_ON_SITE, OUTCOME_PATIENT_INVOICE})
_MODES = frozenset({"tap_to_pay", "cash", "invoice"})
_MODE_ORDER = ("tap_to_pay", "cash", "invoice")
_MODALITIES = frozenset(
    {
        "on_site",
        "patient_invoice_after",
        "established_account",
        "prepaid",
        "institution_account",
        "third_party",
    }
)
_PRIVATE_ORIGINS = frozenset(
    {
        "client_app",
        "public_guest",
        "dispatcher",
        "legacy",
        "api_partner",
    }
)
_THIRD_INTENTS = frozenset({"curator", "spc", "other", "insurance"})
_READY_STATUSES = frozenset({"IN_PROGRESS", "COMPLETED", "RETURN_COMPLETED"})
_CENT = Decimal("0.01")

INVOICE_SIGNATURE_DISCLAIMER = (
    "Ce document est distinct de l'attestation de réalisation du trajet. "
    "Il reprend uniquement le texte de facturation approuvé, sa version, "
    "le montant et le débiteur désigné."
)


@dataclass(frozen=True, slots=True)
class SettlementSnapshot:
    """Faits validés avant la course. Aucun lieu de destination."""

    booking_id: int
    company_id: int
    executing_company_id: int
    created_via: str
    billed_to_type: str
    billing_intent: str | None
    billing_party_id: int | None
    settlement_modality: str | None
    reservation_modes: frozenset[str]
    company_modes: frozenset[str]
    invoice_signature_required: bool | None
    invoice_signature_text: str | None
    invoice_signature_text_version: str | None
    leg_amount_chf: Decimal | None
    group_amount_chf: Decimal | None
    amount_validated: bool
    currency: str
    online_payment_status: str | None
    online_payment_covers_scope: bool
    invoice_already_issued: bool
    operational_status: str
    is_round_trip: bool
    is_return_leg: bool
    priced_leg_count: int
    payment_scope: str | None
    collection_moment: str | None
    group_already_settled: bool
    debtor_display_name: str | None
    policy_recorded: bool = False
    financial_responsibility_locked: bool = False
    separate_leg_collection_approved: bool = False
    price_matches_policy: bool = True
    group_collection_approved: bool = False

    def __post_init__(self) -> None:
        if self.currency != "CHF":
            raise ValueError("Devise attendue : CHF.")
        unknown = (self.reservation_modes | self.company_modes) - _MODES
        if unknown:
            raise ValueError("Mode de règlement inconnu.")
        if self.settlement_modality not in (None, *_MODALITIES):
            raise ValueError("Modalité de règlement inconnue.")
        if self.payment_scope not in (None, "booking", "group"):
            raise ValueError("Périmètre de paiement inconnu.")
        if self.collection_moment not in (None, "outbound", "return"):
            raise ValueError("Moment d'encaissement inconnu.")
        if self.priced_leg_count < 1:
            raise ValueError("Nombre de trajets invalide.")
        if self.company_id < 1 or self.executing_company_id < 1:
            raise ValueError("Entreprise invalide.")


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """Résultat conceptuel. L'écran chauffeur reste séparé."""

    outcome: str
    collection_required: bool
    modes: tuple[str, ...]
    signature_required: bool
    signature_document: str
    attestation_separate: bool
    amount_chf: Decimal | None
    amount_scope: str
    currency: str
    driver_may_edit_amount: bool
    driver_may_edit_payer: bool
    ready_for_driver_step: bool
    reasons: tuple[str, ...]


def evaluate(snapshot: SettlementSnapshot) -> EligibilityDecision:
    """Répond : cette mission exige-t-elle un règlement, et par quels modes ?"""
    ready = snapshot.operational_status in _READY_STATUSES
    guards = (
        _guard_company,
        _guard_canceled,
        _guard_already_paid,
        _guard_invoice_issued,
        _guard_guest,
        _guard_payer,
        _guard_policy_recorded,
        _guard_modality,
        _guard_responsibility_lock,
        _guard_price,
        _guard_online,
        _guard_group,
        _guard_category,
    )
    for guard in guards:
        decision = guard(snapshot, ready)
        if decision is not None:
            return decision
    return _collect(snapshot, ready)


def driver_collection_view(
    snapshot: SettlementSnapshot, *, enabled: bool | None = None
) -> dict[str, object] | None:
    """Vue chauffeur. Absente tant que l'activation produit est coupée."""
    active = END_OF_TRIP_COLLECTION_ENABLED if enabled is None else enabled
    if not active:
        return None
    decision = evaluate(snapshot)
    if not decision.collection_required or not decision.ready_for_driver_step:
        return None
    show_text = decision.signature_required
    amount = decision.amount_chf
    return {
        "outcome": decision.outcome,
        "modes": list(decision.modes),
        "amount_chf": format(amount, "f") if amount is not None else None,
        "currency": "CHF",
        "amount_scope": decision.amount_scope,
        "debtor_display_name": snapshot.debtor_display_name,
        "signature_required": decision.signature_required,
        "signature_document": decision.signature_document,
        "signature_text": snapshot.invoice_signature_text if show_text else None,
        "signature_text_version": (
            snapshot.invoice_signature_text_version if show_text else None
        ),
        "disclaimer": INVOICE_SIGNATURE_DISCLAIMER if show_text else None,
        "attestation_separate": True,
        "driver_may_edit_amount": False,
        "driver_may_edit_payer": False,
    }


def decision_contract(
    decision: EligibilityDecision,
    snapshot: SettlementSnapshot | None = None,
) -> dict[str, object]:
    """Contrat backend. Le téléphone ne reconstruit pas cette décision."""
    if decision.outcome == OUTCOME_ALREADY_PAID:
        kind = "already_paid"
    elif decision.outcome == OUTCOME_ON_SITE:
        kind = "collection_authorized"
    elif decision.outcome == OUTCOME_PATIENT_INVOICE:
        kind = "invoice_authorized"
    elif decision.outcome == OUTCOME_NO_COLLECTION:
        kind = "no_collection"
    elif decision.outcome == OUTCOME_INSTITUTION or decision.reasons in {
        ("payer_still_mutable",),
        ("company_not_owner",),
        ("guest_flow_unchanged",),
    }:
        kind = "collection_forbidden"
    else:
        kind = "administrative_review"
    amount = decision.amount_chf
    scope = decision.amount_scope if decision.amount_scope != "none" else None
    contract: dict[str, object] = {
        "kind": kind,
        "reason": decision.reasons[0] if decision.reasons else None,
        "modes": list(decision.modes),
        "amount_chf": format(amount, "f") if amount is not None else None,
        "currency": decision.currency,
        "payment_scope": scope,
        "collection_moment": None,
        "company_id": None,
        "signature_required": decision.signature_required,
        "administrative_review": kind == "administrative_review",
        "payment_received": decision.outcome == OUTCOME_ALREADY_PAID,
        "invoice_means_paid": False,
        "price_computed_locally": False,
        "payer_computed_locally": False,
        "source": "backend",
        "reconstruct_locally": False,
    }
    if snapshot is not None:
        contract["company_id"] = snapshot.company_id
        contract["collection_moment"] = snapshot.collection_moment
        if snapshot.payment_scope in {"booking", "group"}:
            contract["payment_scope"] = snapshot.payment_scope
    return contract


def _finish(
    outcome: str,
    *,
    ready: bool,
    reasons: tuple[str, ...],
    modes: tuple[str, ...] = (),
    signature_required: bool = False,
    amount: Decimal | None = None,
    amount_scope: str = "none",
) -> EligibilityDecision:
    document = "invoice_acknowledgement" if signature_required else "none"
    return EligibilityDecision(
        outcome=outcome,
        collection_required=outcome in _COLLECT,
        modes=modes,
        signature_required=signature_required,
        signature_document=document,
        attestation_separate=True,
        amount_chf=amount,
        amount_scope=amount_scope,
        currency="CHF",
        driver_may_edit_amount=False,
        driver_may_edit_payer=False,
        ready_for_driver_step=ready,
        reasons=reasons,
    )


def _guard_company(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    if snapshot.executing_company_id == snapshot.company_id:
        return None
    return _finish(OUTCOME_ADMIN, ready=ready, reasons=("company_not_owner",))


def _guard_canceled(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    if snapshot.operational_status != "CANCELED":
        return None
    return _finish(OUTCOME_NO_COLLECTION, ready=ready, reasons=("canceled",))


def _guard_already_paid(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    covered = (
        snapshot.online_payment_status or ""
    ).lower() == "completed" and snapshot.online_payment_covers_scope
    if not covered and not snapshot.group_already_settled:
        return None
    return _finish(OUTCOME_ALREADY_PAID, ready=ready, reasons=("payment_confirmed",))


def _guard_invoice_issued(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    if not snapshot.invoice_already_issued:
        return None
    return _finish(
        OUTCOME_NO_COLLECTION, ready=ready, reasons=("invoice_already_issued",)
    )


def _payer_class(snapshot: SettlementSnapshot) -> str:
    billed = (snapshot.billed_to_type or "").lower()
    intent = (snapshot.billing_intent or "").lower()
    institution = billed == "clinic" or intent == "institution"
    patient = billed == "patient" or intent == "patient"
    third = billed == "insurance" or intent in _THIRD_INTENTS
    if patient and third and intent in _THIRD_INTENTS and billed == "patient":
        return "third_party"
    if (institution and patient) or (institution and third) or (patient and third):
        return "conflict"
    if institution:
        return "institution"
    if third:
        return "third_party"
    if snapshot.created_via == "institution_portal" and intent != "patient":
        return "unconfirmed"
    if billed == "patient" and intent in {"", "patient"}:
        return "patient"
    return "unconfirmed"


def _guard_guest(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    """Le parcours invité certifié ne gagne pas d'encaissement sur place."""
    if snapshot.created_via != "public_guest":
        return None
    return _finish(OUTCOME_ADMIN, ready=ready, reasons=("guest_flow_unchanged",))


def _guard_policy_recorded(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    if snapshot.policy_recorded:
        return None
    return _finish(OUTCOME_ADMIN, ready=ready, reasons=("settlement_policy_missing",))


def _guard_responsibility_lock(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    """Une validation de contrôle ne suffit pas : le payeur doit être verrouillé.

    Le verrou de facturation et le verrou de responsabilité financière sont
    distincts. Le passage au mois suivant n'est pas consulté.
    """
    if snapshot.created_via != "institution_portal":
        return None
    if snapshot.settlement_modality not in {"on_site", "patient_invoice_after"}:
        return None
    if snapshot.financial_responsibility_locked:
        return None
    return _finish(OUTCOME_ADMIN, ready=ready, reasons=("payer_still_mutable",))


def _guard_price(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    if snapshot.price_matches_policy:
        return None
    return _finish(OUTCOME_ADMIN, ready=ready, reasons=("price_policy_mismatch",))


def _guard_payer(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    payer = _payer_class(snapshot)
    if payer == "institution":
        return _finish(OUTCOME_INSTITUTION, ready=ready, reasons=("institution_payer",))
    if payer == "third_party":
        return _finish(
            OUTCOME_NO_COLLECTION, ready=ready, reasons=("third_party_convention",)
        )
    if payer == "conflict":
        return _finish(OUTCOME_ADMIN, ready=ready, reasons=("payer_conflict",))
    if payer == "unconfirmed":
        return _finish(OUTCOME_ADMIN, ready=ready, reasons=("payer_not_explicit",))
    return None


def _guard_modality(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    modality = snapshot.settlement_modality
    if modality in (None, "", "unspecified"):
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("settlement_modality_missing",)
        )
    if modality == "established_account":
        return _finish(
            OUTCOME_NO_COLLECTION, ready=ready, reasons=("established_account",)
        )
    if modality == "institution_account":
        return _finish(
            OUTCOME_INSTITUTION, ready=ready, reasons=("institution_account",)
        )
    if modality == "third_party":
        return _finish(
            OUTCOME_NO_COLLECTION, ready=ready, reasons=("third_party_convention",)
        )
    if modality == "prepaid":
        return _finish(OUTCOME_ADMIN, ready=ready, reasons=("prepaid_not_confirmed",))
    if modality in {"on_site", "patient_invoice_after"}:
        return None
    return _finish(OUTCOME_ADMIN, ready=ready, reasons=("unknown_modality",))


def _guard_online(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    status = (snapshot.online_payment_status or "").lower()
    if status == "completed" and not snapshot.online_payment_covers_scope:
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("online_payment_scope_mismatch",)
        )
    if status in {"pending", "uncertain", "failed"}:
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("online_payment_unresolved",)
        )
    return None


def _effective_scope(snapshot: SettlementSnapshot) -> str | None:
    if snapshot.payment_scope in {"booking", "group"}:
        return snapshot.payment_scope
    single = (
        snapshot.priced_leg_count <= 1
        and not snapshot.is_round_trip
        and not snapshot.is_return_leg
    )
    if single:
        return "booking"
    return None


def _guard_group(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    scope = _effective_scope(snapshot)
    if scope is None:
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("collection_scope_missing",)
        )
    if scope == "booking":
        multi = (
            snapshot.priced_leg_count >= 2
            or snapshot.is_round_trip
            or snapshot.is_return_leg
        )
        if multi and not snapshot.separate_leg_collection_approved:
            return _finish(
                OUTCOME_ADMIN,
                ready=ready,
                reasons=("separate_leg_collection_unapproved",),
            )
        return None
    if not snapshot.group_collection_approved:
        return _finish(
            OUTCOME_ADMIN,
            ready=ready,
            reasons=("round_trip_collection_unapproved",),
        )
    if snapshot.priced_leg_count < 2:
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("inconsistent_group_scope",)
        )
    if snapshot.collection_moment not in {"outbound", "return"}:
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("collection_moment_missing",)
        )
    if snapshot.leg_amount_chf is None or snapshot.group_amount_chf is None:
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("incomplete_group_amounts",)
        )
    leg = snapshot.leg_amount_chf.quantize(_CENT)
    group = snapshot.group_amount_chf.quantize(_CENT)
    if group == leg:
        return _finish(
            OUTCOME_ADMIN, ready=ready, reasons=("group_amount_equals_single_leg",)
        )
    this_leg = "return" if snapshot.is_return_leg else "outbound"
    if this_leg == snapshot.collection_moment:
        return None
    if this_leg == "return":
        return _finish(OUTCOME_ADMIN, ready=ready, reasons=("missed_group_collection",))
    return _finish(
        OUTCOME_NO_COLLECTION, ready=ready, reasons=("outside_collection_moment",)
    )


def _guard_category(
    snapshot: SettlementSnapshot, ready: bool
) -> EligibilityDecision | None:
    modality = snapshot.settlement_modality
    private = snapshot.created_via in _PRIVATE_ORIGINS
    institution = snapshot.created_via == "institution_portal"
    allowed = modality in {"on_site", "patient_invoice_after"}
    if (private or institution) and allowed:
        return None
    return _finish(OUTCOME_ADMIN, ready=ready, reasons=("category_out_of_scope",))


def _signature_withhold_reason(snapshot: SettlementSnapshot) -> str:
    text = (snapshot.invoice_signature_text or "").strip()
    version = (snapshot.invoice_signature_text_version or "").strip()
    if snapshot.invoice_signature_required is True and not (text and version):
        return "signature_text_unapproved"
    return "invoice_signature_mandatory"


def _invoice_allowed(snapshot: SettlementSnapshot) -> bool:
    """La facture directe en fin de course exige une signature au texte approuvé."""
    if snapshot.invoice_signature_required is not True:
        return False
    text = (snapshot.invoice_signature_text or "").strip()
    version = (snapshot.invoice_signature_text_version or "").strip()
    return bool(text and version)


def _collect(snapshot: SettlementSnapshot, ready: bool) -> EligibilityDecision:
    modality = snapshot.settlement_modality
    outcome = OUTCOME_ON_SITE if modality == "on_site" else OUTCOME_PATIENT_INVOICE
    raw = snapshot.reservation_modes & snapshot.company_modes
    if modality == "patient_invoice_after":
        raw = raw & frozenset({"invoice"})
    if not raw:
        return _finish(OUTCOME_ADMIN, ready=ready, reasons=("no_authorized_mode",))
    extra: list[str] = []
    modes = set(raw)
    if "invoice" in modes and not _invoice_allowed(snapshot):
        modes.discard("invoice")
        extra.append(_signature_withhold_reason(snapshot))
    if "invoice" in modes and snapshot.billing_party_id is None:
        modes.discard("invoice")
        extra.append("missing_billing_party")
    ordered = tuple(mode for mode in _MODE_ORDER if mode in modes)
    if not ordered:
        reason = extra[0] if extra else "no_authorized_mode"
        return _finish(OUTCOME_ADMIN, ready=ready, reasons=(reason,))
    scope = _effective_scope(snapshot) or "booking"
    amount = snapshot.group_amount_chf if scope == "group" else snapshot.leg_amount_chf
    if amount is not None:
        amount = amount.quantize(_CENT)
    if not snapshot.amount_validated or amount is None or amount <= 0:
        return _finish(OUTCOME_AMOUNT, ready=ready, reasons=("amount_unvalidated",))
    signature = "invoice" in ordered and snapshot.invoice_signature_required is True
    reason = (
        "patient_invoice_authorized"
        if outcome == OUTCOME_PATIENT_INVOICE
        else "on_site_authorized"
    )
    return _finish(
        outcome,
        ready=ready,
        reasons=(reason, *extra),
        modes=ordered,
        signature_required=signature,
        amount=amount,
        amount_scope=scope,
    )
