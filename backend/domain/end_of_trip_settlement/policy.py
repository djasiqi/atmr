"""Politique de règlement enregistrée sur une réservation.

Le payeur reste `billed_to_type`, `billing_party_id` et `billing_intent`.
Cette politique dit seulement comment et quand encaisser.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from domain.end_of_trip_settlement.eligibility import SettlementSnapshot

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
_MODES = frozenset({"tap_to_pay", "cash", "invoice"})
_ORIGINS = frozenset({"client", "institution", "company"})

# Table prévue `booking_settlement_policy`. Aucune colonne de payeur :
# l'absence de ligne vaut « pas de politique », donc pas d'encaissement.
SETTLEMENT_POLICY_TABLE = (
    ("booking_id", False),
    ("revision", False),
    ("modality", False),
    ("authorized_modes", False),
    ("payment_scope", True),
    ("collection_moment", True),
    ("amount_chf", True),
    ("currency", False),
    ("decision_origin", False),
    ("policy_version", False),
    ("financial_responsibility_locked", False),
    ("invoice_signature_text", True),
    ("invoice_signature_text_version", True),
    ("validated_at", False),
    ("validated_by_user_id", False),
    ("status", False),
    ("supersedes_revision", True),
)


@dataclass(frozen=True, slots=True)
class SettlementPolicy:
    """Une révision. L'historique ne s'efface pas."""

    booking_id: int
    revision: int
    modality: str
    modes: frozenset[str]
    payment_scope: str | None
    collection_moment: str | None
    amount_chf: Decimal | None
    currency: str
    decision_origin: str
    policy_version: str
    financial_responsibility_locked: bool
    invoice_signature_text: str | None
    invoice_signature_text_version: str | None
    validated_at: datetime
    validated_by_user_id: int
    status: str = "active"
    supersedes_revision: int | None = None
    round_trip_collection_approved: bool = False

    def __post_init__(self) -> None:
        if self.modality not in _MODALITIES:
            raise ValueError("Modalité de règlement inconnue.")
        if self.modes - _MODES:
            raise ValueError("Mode de règlement inconnu.")
        if self.decision_origin not in _ORIGINS:
            raise ValueError("Origine de décision inconnue.")
        if self.currency != "CHF":
            raise ValueError("Devise attendue : CHF.")
        if self.validated_at.tzinfo is None:
            raise ValueError("Horodatage de validation avec fuseau requis.")
        if self.validated_by_user_id < 1 or self.revision < 1:
            raise ValueError("Révision de politique invalide.")
        if not self.policy_version.strip():
            raise ValueError("Version de politique requise.")


@dataclass(frozen=True, slots=True)
class PolicyRevisionResult:
    accepted: bool
    active: SettlementPolicy | None
    history: tuple[SettlementPolicy, ...]
    reason: str


def propose_policy(
    history: tuple[SettlementPolicy, ...],
    proposal: SettlementPolicy,
    *,
    payment_confirmed: bool,
    invoice_issued: bool,
    cash_recorded: bool,
) -> PolicyRevisionResult:
    """Ajoute une révision tant qu'aucun règlement n'a été enregistré."""
    active = _active(history)
    if active is not None and active.financial_responsibility_locked:
        return PolicyRevisionResult(
            accepted=False,
            active=active,
            history=history,
            reason="policy_locked",
        )
    if payment_confirmed or invoice_issued or cash_recorded:
        return PolicyRevisionResult(
            accepted=False,
            active=_active(history),
            history=history,
            reason="policy_frozen_after_settlement",
        )
    if history and proposal.booking_id != history[0].booking_id:
        raise ValueError("La politique ne change pas de réservation.")
    revision = (history[-1].revision + 1) if history else 1
    superseded = tuple(replace(item, status="superseded") for item in history)
    active = replace(
        proposal,
        revision=revision,
        status="active",
        supersedes_revision=(history[-1].revision if history else None),
    )
    return PolicyRevisionResult(
        accepted=True,
        active=active,
        history=(*superseded, active),
        reason="policy_recorded",
    )


def snapshot_from_policy(
    policy: SettlementPolicy | None,
    *,
    booking_id: int,
    company_id: int,
    executing_company_id: int,
    created_via: str,
    billed_to_type: str,
    billing_intent: str | None,
    billing_party_id: int | None,
    company_modes: frozenset[str],
    leg_amount_chf: Decimal | None,
    group_amount_chf: Decimal | None,
    amount_validated: bool,
    online_payment_status: str | None,
    online_payment_covers_scope: bool,
    invoice_already_issued: bool,
    operational_status: str,
    is_round_trip: bool,
    is_return_leg: bool,
    priced_leg_count: int,
    group_already_settled: bool,
    debtor_display_name: str | None,
    separate_leg_collection_approved: bool = False,
) -> SettlementSnapshot:
    """Relie la politique au payeur existant. Sans ligne : rien à encaisser."""
    if policy is None or policy.status != "active":
        return SettlementSnapshot(
            booking_id=booking_id,
            company_id=company_id,
            executing_company_id=executing_company_id,
            created_via=created_via,
            billed_to_type=billed_to_type,
            billing_intent=billing_intent,
            billing_party_id=billing_party_id,
            settlement_modality=None,
            reservation_modes=frozenset(),
            company_modes=company_modes,
            invoice_signature_required=None,
            invoice_signature_text=None,
            invoice_signature_text_version=None,
            leg_amount_chf=leg_amount_chf,
            group_amount_chf=group_amount_chf,
            amount_validated=amount_validated,
            currency="CHF",
            online_payment_status=online_payment_status,
            online_payment_covers_scope=online_payment_covers_scope,
            invoice_already_issued=invoice_already_issued,
            operational_status=operational_status,
            is_round_trip=is_round_trip,
            is_return_leg=is_return_leg,
            priced_leg_count=priced_leg_count,
            payment_scope=None,
            collection_moment=None,
            group_already_settled=group_already_settled,
            debtor_display_name=debtor_display_name,
            policy_recorded=False,
        )
    text = (policy.invoice_signature_text or "").strip()
    version = (policy.invoice_signature_text_version or "").strip()
    runtime_amount = (
        group_amount_chf if policy.payment_scope == "group" else leg_amount_chf
    )
    if policy.amount_chf is None or runtime_amount is None:
        priced = runtime_amount if runtime_amount is not None else policy.amount_chf
        amount_ok = False
        price_matches = True
    elif policy.amount_chf != runtime_amount:
        priced = None
        amount_ok = False
        price_matches = False
    else:
        priced = policy.amount_chf
        amount_ok = amount_validated
        price_matches = True
    leg = priced if policy.payment_scope != "group" else leg_amount_chf
    group = priced if policy.payment_scope == "group" else group_amount_chf
    return SettlementSnapshot(
        booking_id=booking_id,
        company_id=company_id,
        executing_company_id=executing_company_id,
        created_via=created_via,
        billed_to_type=billed_to_type,
        billing_intent=billing_intent,
        billing_party_id=billing_party_id,
        settlement_modality=policy.modality,
        reservation_modes=policy.modes,
        company_modes=company_modes,
        invoice_signature_required=bool(text and version),
        invoice_signature_text=policy.invoice_signature_text,
        invoice_signature_text_version=policy.invoice_signature_text_version,
        leg_amount_chf=leg,
        group_amount_chf=group,
        amount_validated=amount_ok,
        currency="CHF",
        online_payment_status=online_payment_status,
        online_payment_covers_scope=online_payment_covers_scope,
        invoice_already_issued=invoice_already_issued,
        operational_status=operational_status,
        is_round_trip=is_round_trip,
        is_return_leg=is_return_leg,
        priced_leg_count=priced_leg_count,
        payment_scope=policy.payment_scope,
        collection_moment=policy.collection_moment,
        group_already_settled=group_already_settled,
        debtor_display_name=debtor_display_name,
        policy_recorded=True,
        financial_responsibility_locked=policy.financial_responsibility_locked,
        separate_leg_collection_approved=separate_leg_collection_approved,
        price_matches_policy=price_matches,
        group_collection_approved=policy.round_trip_collection_approved,
    )


@dataclass(frozen=True, slots=True)
class CollectionClaim:
    driver_id: int
    state: str


def claim_collection(
    existing: CollectionClaim | None, driver_id: int
) -> tuple[CollectionClaim, str]:
    """Un second chauffeur ne reprend pas un règlement déjà en cours."""
    if driver_id < 1:
        raise ValueError("Chauffeur invalide.")
    if existing is not None and existing.state == "in_progress":
        if existing.driver_id != driver_id:
            return existing, "concurrent_claim"
        return existing, "same_claim"
    return CollectionClaim(driver_id=driver_id, state="in_progress"), "granted"


@dataclass(frozen=True, slots=True)
class CashLedgerEntry:
    entry_id: int
    company_id: int
    driver_id: int
    expected_amount_chf: Decimal
    received_amount_chf: Decimal
    receipt_reference: str
    status: str
    supersedes_entry_id: int | None
    electronic_confirmation: bool


def record_cash_ledger(
    existing: tuple[CashLedgerEntry, ...],
    *,
    booking_company_id: int,
    receiving_company_id: int,
    driver_id: int,
    expected_amount_chf: Decimal,
    received_amount_chf: Decimal | None,
    receipt_reference: str,
    invoice_already_issued: bool,
    corrects_entry_id: int | None = None,
) -> tuple[tuple[CashLedgerEntry, ...], str]:
    """Ajoute une ligne de caisse. Une correction ne retire pas la ligne d'origine."""
    if receiving_company_id != booking_company_id:
        return existing, "wrong_company"
    if invoice_already_issued:
        return existing, "invoice_already_issued"
    reference = receipt_reference.strip()
    if driver_id < 1 or received_amount_chf is None or not reference:
        return existing, "cash_declaration_incomplete"
    if corrects_entry_id is not None and not any(
        item.entry_id == corrects_entry_id for item in existing
    ):
        return existing, "missing_original_entry"
    status = "declared" if received_amount_chf == expected_amount_chf else "difference"
    entry = CashLedgerEntry(
        entry_id=(existing[-1].entry_id + 1) if existing else 1,
        company_id=receiving_company_id,
        driver_id=driver_id,
        expected_amount_chf=expected_amount_chf,
        received_amount_chf=received_amount_chf,
        receipt_reference=reference,
        status=status,
        supersedes_entry_id=corrects_entry_id,
        electronic_confirmation=False,
    )
    return (*existing, entry), "cash_declared"


def _active(history: tuple[SettlementPolicy, ...]) -> SettlementPolicy | None:
    for item in reversed(history):
        if item.status == "active":
            return item
    return None
