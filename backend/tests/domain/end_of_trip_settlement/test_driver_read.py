"""Lecture chauffeur et fixtures d'écran. Aucune base, aucun prestataire."""

from __future__ import annotations

from decimal import Decimal

from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from domain.end_of_trip_settlement.driver_read import (
    invalidate_displayed_amount,
    read_driver_settlement,
    read_for_assigned_driver,
    simulated_driver_contract,
)
from domain.end_of_trip_settlement.eligibility import (
    SettlementSnapshot,
    driver_collection_view,
)
from domain.end_of_trip_settlement.prototypes import (
    TAP_IN_PROGRESS,
    TAP_UNCERTAIN,
    open_tap_attempt,
    resolve_tap_simulation,
)


def snap(**overrides: object) -> SettlementSnapshot:
    base: dict[str, object] = {
        "booking_id": 1,
        "company_id": 10,
        "executing_company_id": 10,
        "created_via": "client_app",
        "billed_to_type": "patient",
        "billing_intent": None,
        "billing_party_id": 5,
        "settlement_modality": "on_site",
        "reservation_modes": frozenset({"tap_to_pay", "cash", "invoice"}),
        "company_modes": frozenset(),
        "invoice_signature_required": False,
        "invoice_signature_text": None,
        "invoice_signature_text_version": None,
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": None,
        "amount_validated": True,
        "currency": "CHF",
        "online_payment_status": None,
        "online_payment_covers_scope": False,
        "invoice_already_issued": False,
        "operational_status": "COMPLETED",
        "is_round_trip": False,
        "is_return_leg": False,
        "priced_leg_count": 1,
        "payment_scope": "booking",
        "collection_moment": None,
        "group_already_settled": False,
        "debtor_display_name": "Patient fictif",
        "policy_recorded": True,
    }
    base.update(overrides)
    return SettlementSnapshot(**base)  # type: ignore[arg-type]


def _assigned(snapshot: SettlementSnapshot, **kwargs: object) -> dict[str, object]:
    return read_for_assigned_driver(
        snapshot,
        assigned=True,
        same_company=True,
        **kwargs,  # type: ignore[arg-type]
    )


def test_encaissement_produit_reste_coupe() -> None:
    assert END_OF_TRIP_COLLECTION_ENABLED is False
    assert driver_collection_view(snap()) is None


def test_course_terminee_sans_moyen_autorise_n_encaisse_pas() -> None:
    capable = _assigned(snap(company_modes=frozenset({"tap_to_pay", "cash"})))
    disabled = _assigned(snap(company_modes=frozenset()))
    assert capable["kind"] == "administrative_review"
    assert capable["reason"] == "collection_disabled"
    assert disabled["reason"] == "no_authorized_mode"
    reading = capable
    assert reading["modes"] == []
    assert reading["payment_executable"] is False
    assert reading["persists_payment"] is False
    assert reading["completion_independent"] is True
    assert reading["attestation_separate"] is True
    assert reading["authorization_durable"] is False


def test_institution_compte_et_invite_deja_paye() -> None:
    institution = _assigned(
        snap(
            created_via="institution_portal",
            billed_to_type="clinic",
            billing_intent="institution",
            settlement_modality="institution_account",
        )
    )
    account = _assigned(snap(settlement_modality="established_account"))
    guest = _assigned(
        snap(
            created_via="public_guest",
            settlement_modality="prepaid",
            online_payment_status="COMPLETED",
            online_payment_covers_scope=True,
        )
    )
    assert institution["kind"] == "collection_forbidden"
    assert institution["reason"] == "institution_payer"
    assert account["kind"] == "no_collection"
    assert account["reason"] == "established_account"
    assert guest["kind"] == "already_paid"
    assert guest["payment_received"] is True
    assert guest["modes"] == []


def test_patient_institutionnel_non_verrouille() -> None:
    reading = _assigned(
        snap(
            created_via="institution_portal",
            billed_to_type="patient",
            billing_intent="patient",
            financial_responsibility_locked=False,
        )
    )
    assert reading["kind"] == "collection_forbidden"
    assert reading["reason"] == "payer_still_mutable"
    assert reading["modes"] == []


def test_acces_chauffeur() -> None:
    other = read_for_assigned_driver(snap(), assigned=True, same_company=False)
    gone = read_for_assigned_driver(snap(), assigned=False, same_company=True)
    assert other["reason"] == "other_company"
    assert gone["reason"] == "driver_unassigned"
    assert other["amount_chf"] is None
    assert other["modes"] == []


def test_politique_absente_et_obsolete() -> None:
    missing = _assigned(snap(policy_recorded=False))
    stale_policy = _assigned(snap(price_matches_policy=False))
    assert missing["reason"] == "settlement_policy_missing"
    assert stale_policy["reason"] == "price_policy_mismatch"
    assert missing["payment_executable"] is False


def test_montant_modifie_retire_l_ecran() -> None:
    opened = simulated_driver_contract(
        kind="collection_authorized",
        modes=("tap_to_pay",),
        amount_chf="45.00",
    )
    refreshed = invalidate_displayed_amount(
        opened, displayed_amount="45.00", current_amount="50.00"
    )
    assert refreshed["stale"] is True
    assert refreshed["modes"] == []
    assert refreshed["reason"] == "amount_changed"
    assert refreshed["payment_executable"] is False


def test_aller_retour_ne_demande_pas_deux_encaissements() -> None:
    simple = _assigned(snap(leg_amount_chf=Decimal("45.00")))
    prepaid = _assigned(
        snap(
            is_round_trip=True,
            priced_leg_count=2,
            leg_amount_chf=Decimal("45.00"),
            group_amount_chf=Decimal("90.00"),
            payment_scope="group",
            online_payment_status="COMPLETED",
            online_payment_covers_scope=True,
        )
    )
    unapproved = _assigned(
        snap(
            is_round_trip=True,
            priced_leg_count=2,
            leg_amount_chf=Decimal("45.00"),
            group_amount_chf=Decimal("90.00"),
            payment_scope="group",
            group_collection_approved=False,
        )
    )
    return_open = _assigned(
        snap(
            is_round_trip=True,
            is_return_leg=True,
            operational_status="ACCEPTED",
            priced_leg_count=2,
            payment_scope="group",
            group_amount_chf=Decimal("90.00"),
        )
    )
    return_unscheduled = _assigned(
        snap(
            is_round_trip=True,
            is_return_leg=True,
            operational_status="ACCEPTED",
            priced_leg_count=2,
            collection_moment="return",
            payment_scope="group",
            group_amount_chf=Decimal("90.00"),
        )
    )
    other_company = _assigned(
        snap(
            is_round_trip=True,
            executing_company_id=99,
            priced_leg_count=2,
            group_amount_chf=Decimal("90.00"),
            payment_scope="group",
            operational_status="COMPLETED",
        )
    )
    assert simple["payment_executable"] is False
    assert prepaid["kind"] == "already_paid"
    assert prepaid["amount_chf"] is None
    assert unapproved["reason"] == "round_trip_collection_unapproved"
    assert return_open["payment_executable"] is False
    assert return_unscheduled["modes"] == []
    assert other_company["reason"] == "company_not_owner"
    for reading in (
        simple,
        prepaid,
        unapproved,
        return_open,
        return_unscheduled,
        other_company,
    ):
        assert reading["modes"] == []
        assert reading["writes_completed_payment"] is False


def test_attestation_deja_enregistree_ne_change_pas_la_course() -> None:
    reading = _assigned(snap(), attestation_recorded=True)
    assert reading["attestation_recorded"] is True
    assert reading["attestation_separate"] is True
    assert reading["completion_independent"] is True
    assert reading["kind"] != "already_paid"


def test_simulation_n_ecrit_aucun_paiement() -> None:
    screen = simulated_driver_contract(
        kind="collection_authorized",
        modes=("tap_to_pay", "cash", "invoice"),
        amount_chf="45.00",
        document_approved=False,
    )
    assert screen["simulation"] is True
    assert screen["source"] == "simulation"
    assert screen["payment_executable"] is False
    assert screen["persists_payment"] is False
    assert screen["writes_cash_receipt"] is False
    assert screen["writes_invoice"] is False
    assert screen["writes_financial_signature"] is False
    assert screen["invoice_means_paid"] is False
    attempt = open_tap_attempt(
        current_state="ready",
        current_attempt_id=None,
        amount_chf=Decimal("45.00"),
        currency="CHF",
        merchant_company_id=10,
        new_attempt_id="sim-1",
    )
    lost = resolve_tap_simulation(attempt, TAP_UNCERTAIN)
    assert attempt.state == TAP_IN_PROGRESS
    assert lost.may_start_another_debit is False
    assert lost.state == TAP_UNCERTAIN
