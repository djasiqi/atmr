"""Faits d'une mission et fixtures de prévisualisation.

Les préférences d'entreprise ne deviennent pas des moyens autorisés.
Les fixtures ne lisent aucune réservation persistante.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from domain.end_of_trip_settlement.driver_read import (
    read_for_assigned_driver,
    simulated_driver_contract,
)
from domain.end_of_trip_settlement.policy import SettlementPolicy, snapshot_from_policy

_CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class MissionFacts:
    """Faits déjà lus. Le téléphone ne les reconstitue pas."""

    booking_id: int
    company_id: int
    executing_company_id: int
    reader_driver_id: int
    reader_company_id: int
    assigned_driver_id: int | None
    created_via: str
    billed_to_type: str
    billing_intent: str | None
    billing_party_id: int | None
    operational_status: str
    leg_amount_chf: Decimal
    group_amount_chf: Decimal | None
    is_round_trip: bool
    is_return_leg: bool
    priced_leg_count: int
    invoice_already_issued: bool
    completed_payment_amount_chf: Decimal
    debtor_display_name: str | None
    policy: SettlementPolicy | None
    tap_to_pay_enabled: bool
    cash_enabled: bool
    direct_patient_invoice_enabled: bool
    tap_to_pay_preferred: bool = False
    cash_preferred: bool = False
    direct_patient_invoice_preferred: bool = False
    attestation_recorded: bool = False
    separate_leg_collection_approved: bool = False


def enabled_company_modes(facts: MissionFacts) -> frozenset[str]:
    """Seules les colonnes d'autorisation comptent."""
    modes: set[str] = set()
    if facts.tap_to_pay_enabled:
        modes.add("tap_to_pay")
    if facts.cash_enabled:
        modes.add("cash")
    if facts.direct_patient_invoice_enabled:
        modes.add("invoice")
    return frozenset(modes)


def decide_mission(facts: MissionFacts) -> dict[str, object]:
    """Décision serveur d'une mission. N'écrit rien et n'encaisse rien."""
    paid = facts.completed_payment_amount_chf.quantize(_CENT)
    group = facts.group_amount_chf
    leg = facts.leg_amount_chf.quantize(_CENT)
    covers_group = bool(facts.is_round_trip and group is not None and paid >= group)
    covers_leg = paid >= leg and paid > 0
    online_status = "COMPLETED" if paid > 0 else None
    snapshot = snapshot_from_policy(
        facts.policy,
        booking_id=facts.booking_id,
        company_id=facts.company_id,
        executing_company_id=facts.executing_company_id,
        created_via=facts.created_via,
        billed_to_type=facts.billed_to_type,
        billing_intent=facts.billing_intent,
        billing_party_id=facts.billing_party_id,
        company_modes=enabled_company_modes(facts),
        leg_amount_chf=leg,
        group_amount_chf=group,
        amount_validated=leg > 0,
        online_payment_status=online_status,
        online_payment_covers_scope=covers_group or (covers_leg and not facts.is_round_trip),
        invoice_already_issued=facts.invoice_already_issued,
        operational_status=facts.operational_status,
        is_round_trip=facts.is_round_trip,
        is_return_leg=facts.is_return_leg,
        priced_leg_count=facts.priced_leg_count,
        group_already_settled=covers_group,
        debtor_display_name=facts.debtor_display_name,
        separate_leg_collection_approved=facts.separate_leg_collection_approved,
    )
    reading = read_for_assigned_driver(
        snapshot,
        assigned=facts.assigned_driver_id == facts.reader_driver_id
        and facts.assigned_driver_id is not None,
        same_company=facts.reader_company_id == facts.company_id,
        attestation_recorded=facts.attestation_recorded,
    )
    reading["simulation"] = False
    reading["payment_executable"] = False
    reading["modes"] = []
    reading["preview_screen"] = None
    return reading


def simulation_catalog() -> list[dict[str, str]]:
    return [{"id": key, "label": label} for key, label, _builder in _SCENARIOS]


def simulation_scenario(scenario_id: str) -> dict[str, object]:
    """Fixture nommée. Aucune réservation réelle n'est consultée."""
    for key, _label, builder in _SCENARIOS:
        if key == scenario_id:
            contract = builder()
            contract["simulation"] = True
            contract["payment_executable"] = False
            contract["persists_payment"] = False
            return contract
    raise KeyError(scenario_id)


def _money(amount: str, **kwargs: object) -> dict[str, object]:
    return simulated_driver_contract(amount_chf=amount, **kwargs)  # type: ignore[arg-type]


def _tap() -> dict[str, object]:
    return _money(
        "45.00",
        kind="collection_authorized",
        modes=("tap_to_pay",),
        reason="simulation",
    )


def _cash() -> dict[str, object]:
    return _money(
        "45.00",
        kind="collection_authorized",
        modes=("cash",),
        reason="simulation",
    )


def _invoice() -> dict[str, object]:
    return _money(
        "45.00",
        kind="invoice_authorized",
        modes=("invoice",),
        reason="simulation",
        signature_required=True,
        document_approved=False,
    )


def _several() -> dict[str, object]:
    return _money(
        "45.00",
        kind="collection_authorized",
        modes=("tap_to_pay", "cash", "invoice"),
        reason="simulation",
    )


def _refused() -> dict[str, object]:
    contract = _tap()
    contract["preview_screen"] = "tap-refused"
    return contract


def _uncertain() -> dict[str, object]:
    contract = _tap()
    contract["preview_screen"] = "tap-uncertain"
    return contract


def _interrupted() -> dict[str, object]:
    contract = _tap()
    contract["preview_screen"] = "tap-interrupted"
    return contract


def _cash_missing() -> dict[str, object]:
    contract = _cash()
    contract["preview_screen"] = "cash-difference"
    return contract


def _signature_refused() -> dict[str, object]:
    contract = _invoice()
    contract["preview_screen"] = "invoice-refused"
    return contract


def _signature_unable() -> dict[str, object]:
    contract = _invoice()
    contract["preview_screen"] = "invoice-unable"
    return contract


def _already_paid() -> dict[str, object]:
    return simulated_driver_contract(
        kind="already_paid",
        reason="payment_confirmed",
        payment_scope="booking",
    )


def _institution() -> dict[str, object]:
    return simulated_driver_contract(
        kind="collection_forbidden",
        reason="institution_payer",
    )


def _round_trip_paid() -> dict[str, object]:
    return simulated_driver_contract(
        kind="already_paid",
        reason="payment_confirmed",
        payment_scope="group",
        amount_chf="90.00",
    )


_SCENARIOS: tuple[tuple[str, str, object], ...] = (
    ("tap-to-pay", "Patient éligible à Tap to Pay", _tap),
    ("cash", "Patient éligible aux espèces", _cash),
    ("invoice", "Patient éligible à la facture", _invoice),
    ("several-means", "Plusieurs moyens disponibles", _several),
    ("card-refused", "Carte refusée", _refused),
    ("card-uncertain", "Carte avec réponse incertaine", _uncertain),
    ("card-interrupted", "Carte interrompue", _interrupted),
    ("cash-missing", "Espèces non remises", _cash_missing),
    ("signature-refused", "Patient refusant la signature", _signature_refused),
    ("signature-unable", "Patient incapable de signer", _signature_unable),
    ("already-paid", "Réservation déjà payée", _already_paid),
    ("institution", "Institution payeuse", _institution),
    ("round-trip-paid", "Aller-retour déjà réglé", _round_trip_paid),
)


def policy_at(
    *,
    modality: str,
    modes: frozenset[str],
    locked: bool,
    payment_scope: str | None = "booking",
    amount_chf: str | None = "45.00",
    round_trip_collection_approved: bool = False,
    validated_at: datetime | None = None,
) -> SettlementPolicy:
    """Aide de test et de lecture. N'enregistre pas la politique."""
    from datetime import UTC

    moment = validated_at or datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
    return SettlementPolicy(
        booking_id=1,
        revision=1,
        modality=modality,
        modes=modes,
        payment_scope=payment_scope,
        collection_moment="outbound" if payment_scope == "group" else None,
        amount_chf=Decimal(amount_chf) if amount_chf is not None else None,
        currency="CHF",
        decision_origin="client",
        policy_version="recette",
        financial_responsibility_locked=locked,
        invoice_signature_text=None,
        invoice_signature_text_version=None,
        validated_at=moment,
        validated_by_user_id=1,
        round_trip_collection_approved=round_trip_collection_approved,
    )
