"""Matrice d'encaissement en fin de transport. Aucun prestataire, aucune base."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from domain.end_of_trip_settlement.activation import (
    END_OF_TRIP_COLLECTION_ENABLED,
)
from domain.end_of_trip_settlement.eligibility import (
    OUTCOME_ADMIN,
    OUTCOME_ALREADY_PAID,
    OUTCOME_AMOUNT,
    OUTCOME_INSTITUTION,
    OUTCOME_NO_COLLECTION,
    OUTCOME_ON_SITE,
    OUTCOME_PATIENT_INVOICE,
    SettlementSnapshot,
    driver_collection_view,
    evaluate,
)
from domain.end_of_trip_settlement.prototypes import (
    TAP_ALREADY_DONE,
    TAP_CONFIRMED,
    TAP_IN_PROGRESS,
    TAP_REFUSED,
    TAP_UNCERTAIN,
    CashDeclarationError,
    declare_cash,
    open_tap_attempt,
    record_exception,
    resolve_tap_simulation,
)

_ROOT = Path(__file__).resolve().parents[3]


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
        "reservation_modes": frozenset({"tap_to_pay", "cash"}),
        "company_modes": frozenset({"tap_to_pay", "cash", "invoice"}),
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
        "operational_status": "IN_PROGRESS",
        "is_round_trip": False,
        "is_return_leg": False,
        "priced_leg_count": 1,
        "payment_scope": None,
        "collection_moment": None,
        "group_already_settled": False,
        "debtor_display_name": "Client",
        "policy_recorded": True,
    }
    base.update(overrides)
    return SettlementSnapshot(**base)  # type: ignore[arg-type]


def test_patient_institution_non_verrouille_ne_s_encaisse_pas() -> None:
    decision = evaluate(
        snap(
            created_via="institution_portal",
            billing_intent="patient",
            settlement_modality="on_site",
            financial_responsibility_locked=False,
        )
    )
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("payer_still_mutable",)
    assert decision.modes == ()


def test_invite_impaye_ne_quitte_pas_le_parcours_certifie() -> None:
    decision = evaluate(
        snap(
            created_via="public_guest",
            settlement_modality="on_site",
            online_payment_status=None,
        )
    )
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("guest_flow_unchanged",)


def test_reservation_sans_politique_ne_declenche_rien() -> None:
    decision = evaluate(snap(policy_recorded=False, settlement_modality="on_site"))
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("settlement_policy_missing",)


def test_retour_sans_politique_explicite() -> None:
    decision = evaluate(
        snap(
            policy_recorded=False,
            is_round_trip=True,
            is_return_leg=True,
            priced_leg_count=2,
            payment_scope="group",
        )
    )
    assert decision.reasons == ("settlement_policy_missing",)


def test_entreprise_sans_moyen_approuve() -> None:
    decision = evaluate(snap(company_modes=frozenset()))
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("no_authorized_mode",)


def test_activation_produit_coupee_par_defaut() -> None:
    assert END_OF_TRIP_COLLECTION_ENABLED is False
    assert driver_collection_view(snap()) is None


def test_cloture_chauffeur_n_importe_pas_le_prototype() -> None:
    source = (
        _ROOT / "application" / "drivers" / "update_driver_booking_status.py"
    ).read_text(encoding="utf-8")
    assert "end_of_trip_settlement" not in source


def test_prototype_sans_prestataire() -> None:
    package = _ROOT / "domain" / "end_of_trip_settlement"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )
    for forbidden in (
        "services.saferpay",
        "services.worldline",
        "import stripe",
        "import adyen",
        "import sumup",
        "import requests",
        "httpx",
    ):
        assert forbidden not in source


def test_client_prive_sur_place_tap_et_especes() -> None:
    decision = evaluate(snap())
    assert decision.outcome == OUTCOME_ON_SITE
    assert decision.collection_required is True
    assert decision.modes == ("tap_to_pay", "cash")
    assert decision.amount_chf == Decimal("45.00")
    assert decision.driver_may_edit_amount is False
    assert decision.driver_may_edit_payer is False
    assert decision.signature_required is False
    assert decision.signature_document == "none"
    assert decision.attestation_separate is True


def test_facture_privee_sans_texte_approuve_est_retenue() -> None:
    decision = evaluate(
        snap(
            reservation_modes=frozenset({"tap_to_pay", "cash", "invoice"}),
            invoice_signature_required=None,
        )
    )
    assert decision.outcome == OUTCOME_ON_SITE
    assert decision.modes == ("tap_to_pay", "cash")
    assert "invoice_signature_mandatory" in decision.reasons


def test_facture_avec_texte_approuve_demande_une_signature_distincte() -> None:
    decision = evaluate(
        snap(
            reservation_modes=frozenset({"invoice"}),
            settlement_modality="patient_invoice_after",
            invoice_signature_required=True,
            invoice_signature_text="Texte approuvé, version d'essai.",
            invoice_signature_text_version="2026-10-essai",
        )
    )
    assert decision.outcome == OUTCOME_PATIENT_INVOICE
    assert decision.modes == ("invoice",)
    assert decision.signature_required is True
    assert decision.signature_document == "invoice_acknowledgement"
    assert decision.signature_document != "transport_attestation"


def test_invite_deja_paye_ne_redemande_rien() -> None:
    decision = evaluate(
        snap(
            created_via="public_guest",
            settlement_modality="on_site",
            online_payment_status="completed",
            online_payment_covers_scope=True,
        )
    )
    assert decision.outcome == OUTCOME_ALREADY_PAID
    assert decision.collection_required is False
    assert decision.modes == ()


def test_paiement_en_ligne_incertain_bloque_un_second_debit() -> None:
    decision = evaluate(snap(online_payment_status="pending"))
    uncertain = evaluate(snap(online_payment_status="uncertain"))
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("online_payment_unresolved",)
    assert uncertain.reasons == ("online_payment_unresolved",)
    assert decision.modes == ()
    assert uncertain.modes == ()


def test_invite_en_attente_ne_bascule_pas_sur_place() -> None:
    decision = evaluate(
        snap(
            created_via="public_guest",
            online_payment_status="pending",
            settlement_modality="on_site",
        )
    )
    assert decision.reasons == ("guest_flow_unchanged",)


def test_clinique_payeuse_meme_vers_le_domicile() -> None:
    decision = evaluate(
        snap(
            created_via="institution_portal",
            billed_to_type="clinic",
            billing_intent="institution",
            settlement_modality="on_site",
        )
    )
    assert decision.outcome == OUTCOME_INSTITUTION
    assert decision.collection_required is False
    assert decision.signature_required is False


def test_patient_institution_sans_modalite_ne_declenche_pas_tap_to_pay() -> None:
    decision = evaluate(
        snap(
            created_via="institution_portal",
            billing_intent="patient",
            settlement_modality=None,
        )
    )
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("settlement_modality_missing",)


def test_patient_institution_sur_place_selon_modes_autorises() -> None:
    decision = evaluate(
        snap(
            created_via="institution_portal",
            billing_intent="patient",
            settlement_modality="on_site",
            financial_responsibility_locked=True,
            reservation_modes=frozenset({"cash"}),
        )
    )
    assert decision.outcome == OUTCOME_ON_SITE
    assert decision.modes == ("cash",)


def test_compte_mensuel_sans_encaissement() -> None:
    decision = evaluate(
        snap(settlement_modality="established_account", billing_intent="patient")
    )
    assert decision.outcome == OUTCOME_NO_COLLECTION
    assert decision.reasons == ("established_account",)


@pytest.mark.parametrize("intent", ["curator", "spc", "other", "insurance"])
def test_tiers_sans_encaissement_patient(intent: str) -> None:
    decision = evaluate(
        snap(
            created_via="institution_portal",
            billed_to_type="patient",
            billing_intent=intent,
            settlement_modality="on_site",
        )
    )
    assert decision.outcome == OUTCOME_NO_COLLECTION
    assert decision.reasons == ("third_party_convention",)


def test_payeur_contredit_reste_administratif() -> None:
    decision = evaluate(
        snap(
            created_via="institution_portal",
            billed_to_type="patient",
            billing_intent="institution",
            settlement_modality="on_site",
        )
    )
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("payer_conflict",)


def test_montant_non_valide_bloque_l_encaissement() -> None:
    decision = evaluate(snap(amount_validated=False))
    assert decision.outcome == OUTCOME_AMOUNT
    assert decision.modes == ()


def test_facture_deja_emise_n_est_pas_dupliquee() -> None:
    decision = evaluate(snap(invoice_already_issued=True))
    assert decision.outcome == OUTCOME_NO_COLLECTION
    assert decision.reasons == ("invoice_already_issued",)


def test_autre_entreprise_ne_peut_pas_encaisser() -> None:
    decision = evaluate(snap(executing_company_id=11))
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("company_not_owner",)


def test_aller_retour_prepaye() -> None:
    common = {
        "is_round_trip": True,
        "priced_leg_count": 2,
        "payment_scope": "group",
        "collection_moment": "outbound",
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": Decimal("90.00"),
        "online_payment_status": "completed",
        "online_payment_covers_scope": True,
    }
    outbound = evaluate(snap(**common, is_return_leg=False))
    inbound = evaluate(snap(**common, is_return_leg=True))
    assert outbound.outcome == OUTCOME_ALREADY_PAID
    assert inbound.outcome == OUTCOME_ALREADY_PAID


def test_aller_retour_institution() -> None:
    decision = evaluate(
        snap(
            created_via="institution_portal",
            billed_to_type="clinic",
            billing_intent="institution",
            is_round_trip=True,
            priced_leg_count=2,
            is_return_leg=True,
        )
    )
    assert decision.outcome == OUTCOME_INSTITUTION
    assert decision.modes == ()


def test_aller_retour_sans_approbation_n_encaisse_pas() -> None:
    common = {
        "created_via": "institution_portal",
        "billing_intent": "patient",
        "settlement_modality": "patient_invoice_after",
        "reservation_modes": frozenset({"invoice"}),
        "company_modes": frozenset({"invoice"}),
        "invoice_signature_required": True,
        "invoice_signature_text": "Texte approuvé, version d'essai.",
        "invoice_signature_text_version": "2026-10-essai",
        "financial_responsibility_locked": True,
        "is_round_trip": True,
        "priced_leg_count": 2,
        "payment_scope": "group",
        "collection_moment": "return",
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": Decimal("90.00"),
    }
    outbound = evaluate(snap(**common, is_return_leg=False))
    inbound = evaluate(snap(**common, is_return_leg=True))
    assert outbound.reasons == ("round_trip_collection_unapproved",)
    assert inbound.reasons == ("round_trip_collection_unapproved",)
    approved = evaluate(
        snap(**common, is_return_leg=True, group_collection_approved=True)
    )
    assert approved.outcome == OUTCOME_PATIENT_INVOICE
    assert approved.amount_chf == Decimal("90.00")
    assert approved.signature_required is True


def test_groupe_90_n_est_pas_redemande_en_45() -> None:
    common = {
        "is_round_trip": True,
        "priced_leg_count": 2,
        "payment_scope": "group",
        "collection_moment": "outbound",
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": Decimal("90.00"),
    }
    blocked = evaluate(snap(**common, is_return_leg=False))
    settled = evaluate(snap(**common, is_return_leg=True, group_already_settled=True))
    approved = evaluate(
        snap(**common, is_return_leg=False, group_collection_approved=True)
    )
    second = evaluate(
        snap(
            **common,
            is_return_leg=True,
            group_collection_approved=True,
            group_already_settled=False,
        )
    )
    assert blocked.reasons == ("round_trip_collection_unapproved",)
    assert settled.outcome == OUTCOME_ALREADY_PAID
    assert approved.amount_chf == Decimal("90.00")
    assert second.reasons == ("missed_group_collection",)
    assert second.amount_chf is None


def test_deux_trajets_separes_exigent_une_approbation() -> None:
    common = {
        "is_round_trip": True,
        "priced_leg_count": 2,
        "payment_scope": "booking",
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": Decimal("90.00"),
    }
    blocked = evaluate(snap(**common, is_return_leg=False))
    approved = evaluate(
        snap(**common, is_return_leg=True, separate_leg_collection_approved=True)
    )
    assert blocked.outcome == OUTCOME_ADMIN
    assert blocked.reasons == ("separate_leg_collection_unapproved",)
    assert approved.amount_chf == Decimal("45.00")
    assert approved.amount_scope == "booking"


def test_montant_groupe_egal_a_un_trajet_est_bloque() -> None:
    common = {
        "is_round_trip": True,
        "priced_leg_count": 2,
        "payment_scope": "group",
        "collection_moment": "outbound",
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": Decimal("45.00"),
    }
    withheld = evaluate(snap(**common))
    approved = evaluate(snap(**common, group_collection_approved=True))
    assert withheld.reasons == ("round_trip_collection_unapproved",)
    assert approved.reasons == ("group_amount_equals_single_leg",)


def test_vue_chauffeur_reste_masquee_si_on_force_seulement_le_calcul() -> None:
    shown = driver_collection_view(snap(), enabled=True)
    assert shown is not None
    assert shown["modes"] == ["tap_to_pay", "cash"]
    assert shown["driver_may_edit_amount"] is False
    hidden = driver_collection_view(snap(operational_status="PENDING"), enabled=True)
    assert hidden is None


def test_tap_incertain_n_ouvre_pas_un_second_debit() -> None:
    started = open_tap_attempt(
        current_state="ready",
        current_attempt_id=None,
        amount_chf=Decimal("45.00"),
        currency="CHF",
        merchant_company_id=10,
        new_attempt_id="tentative-1",
    )
    assert started.state == TAP_IN_PROGRESS
    assert started.may_start_another_debit is False
    lost = resolve_tap_simulation(started, TAP_UNCERTAIN)
    again = open_tap_attempt(
        current_state=lost.state,
        current_attempt_id=lost.attempt_id,
        amount_chf=Decimal("45.00"),
        currency="CHF",
        merchant_company_id=10,
        new_attempt_id="tentative-2",
    )
    assert again.state == TAP_UNCERTAIN
    assert again.attempt_id == "tentative-1"
    assert again.may_start_another_debit is False
    refused = resolve_tap_simulation(started, TAP_REFUSED)
    assert refused.state == TAP_REFUSED
    done = open_tap_attempt(
        current_state=TAP_CONFIRMED,
        current_attempt_id="tentative-1",
        amount_chf=Decimal("45.00"),
        currency="CHF",
        merchant_company_id=10,
        new_attempt_id="tentative-3",
    )
    assert done.state == TAP_ALREADY_DONE


def test_especes_exigent_un_montant_recu() -> None:
    with pytest.raises(CashDeclarationError):
        declare_cash(
            booking_id=1,
            payment_scope="booking",
            expected_amount_chf=Decimal("45.00"),
            received_amount_chf=None,
            company_id=10,
            driver_id=7,
            recorded_at=datetime(2026, 10, 25, tzinfo=UTC),
            cash_reference="caisse-1",
        )
    declared = declare_cash(
        booking_id=1,
        payment_scope="booking",
        expected_amount_chf=Decimal("45.00"),
        received_amount_chf=Decimal("45.00"),
        company_id=10,
        driver_id=7,
        recorded_at=datetime(2026, 10, 25, tzinfo=UTC),
        cash_reference="caisse-1",
    )
    assert declared.status == "declared"
    assert declared.electronic_confirmation is False
    short = declare_cash(
        booking_id=1,
        payment_scope="group",
        expected_amount_chf=Decimal("90.00"),
        received_amount_chf=Decimal("50.00"),
        company_id=10,
        driver_id=7,
        recorded_at=datetime(2026, 10, 25, tzinfo=UTC),
        cash_reference="caisse-2",
    )
    assert short.status == "difference"


@pytest.mark.parametrize(
    "kind",
    [
        "cannot_pay",
        "signature_refused",
        "signature_unable",
        "authorized_representative",
        "technical",
    ],
)
def test_exception_ne_fabrique_ni_paiement_ni_signature(kind: str) -> None:
    record = record_exception(kind)
    assert record.payment_marked_received is False
    assert record.signature_recorded is False
    assert record.service_cancelled is False
    assert record.invoice_auto_issued is False
