"""Politique de règlement : révisions, verrou, caisse. Aucune base."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from domain.end_of_trip_settlement.access import actor_may
from domain.end_of_trip_settlement.eligibility import (
    OUTCOME_ADMIN,
    OUTCOME_NO_COLLECTION,
    OUTCOME_ON_SITE,
    OUTCOME_PATIENT_INVOICE,
    decision_contract,
    evaluate,
)
from domain.end_of_trip_settlement.policy import (
    SETTLEMENT_POLICY_TABLE,
    SettlementPolicy,
    claim_collection,
    propose_policy,
    record_cash_ledger,
    snapshot_from_policy,
)
from domain.end_of_trip_settlement.signature import (
    payer_change_conflict,
    review_billing_signature,
)

WHEN = datetime(2026, 10, 26, tzinfo=UTC)


def policy(**overrides: object) -> SettlementPolicy:
    base: dict[str, object] = {
        "booking_id": 1,
        "revision": 1,
        "modality": "on_site",
        "modes": frozenset({"tap_to_pay", "cash"}),
        "payment_scope": "booking",
        "collection_moment": None,
        "amount_chf": Decimal("45.00"),
        "currency": "CHF",
        "decision_origin": "client",
        "policy_version": "projet-2026-10",
        "financial_responsibility_locked": False,
        "invoice_signature_text": None,
        "invoice_signature_text_version": None,
        "validated_at": WHEN,
        "validated_by_user_id": 4,
    }
    base.update(overrides)
    return SettlementPolicy(**base)  # type: ignore[arg-type]


def _snap_kwargs() -> dict[str, object]:
    return {
        "booking_id": 1,
        "company_id": 10,
        "executing_company_id": 10,
        "created_via": "client_app",
        "billed_to_type": "patient",
        "billing_intent": None,
        "billing_party_id": 5,
        "company_modes": frozenset({"tap_to_pay", "cash", "invoice"}),
        "leg_amount_chf": Decimal("45.00"),
        "group_amount_chf": None,
        "amount_validated": True,
        "online_payment_status": None,
        "online_payment_covers_scope": False,
        "invoice_already_issued": False,
        "operational_status": "IN_PROGRESS",
        "is_round_trip": False,
        "is_return_leg": False,
        "priced_leg_count": 1,
        "group_already_settled": False,
        "debtor_display_name": "Client",
    }


def test_table_ne_duplique_pas_le_payeur() -> None:
    names = {name for name, _nullable in SETTLEMENT_POLICY_TABLE}
    assert "billed_to_type" not in names
    assert "billing_party_id" not in names
    assert "billing_intent" not in names
    assert "booking_id" in names
    assert "financial_responsibility_locked" in names


def test_historique_sans_politique_reste_sans_encaissement() -> None:
    snapshot = snapshot_from_policy(None, **_snap_kwargs())  # type: ignore[arg-type]
    decision = evaluate(snapshot)
    assert snapshot.policy_recorded is False
    assert snapshot.billed_to_type == "patient"
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("settlement_policy_missing",)


def test_revision_avant_execution_conserve_l_historique() -> None:
    first = propose_policy(
        (),
        policy(),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    second = propose_policy(
        first.history,
        policy(modality="established_account", modes=frozenset()),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    assert second.accepted is True
    assert second.active is not None
    assert second.active.revision == 2
    assert second.active.supersedes_revision == 1
    assert [item.status for item in second.history] == ["superseded", "active"]
    assert second.history[0].modality == "on_site"


def test_politique_figee_apres_paiement() -> None:
    current = propose_policy(
        (),
        policy(),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    frozen = propose_policy(
        current.history,
        policy(modality="on_site"),
        payment_confirmed=True,
        invoice_issued=False,
        cash_recorded=False,
    )
    assert frozen.accepted is False
    assert frozen.reason == "policy_frozen_after_settlement"
    assert frozen.history == current.history
    assert frozen.active is not None
    assert frozen.active.modality == "on_site"


def test_politique_verrouillee_refuse_une_revision_contradictoire() -> None:
    locked = propose_policy(
        (),
        policy(financial_responsibility_locked=True),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    revised = propose_policy(
        locked.history,
        policy(modality="patient_invoice_after", modes=frozenset({"invoice"})),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    assert revised.accepted is False
    assert revised.reason == "policy_locked"
    assert revised.history == locked.history


def test_client_prive_sur_place_depuis_la_politique() -> None:
    recorded = propose_policy(
        (),
        policy(),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    decision = evaluate(snapshot_from_policy(recorded.active, **_snap_kwargs()))  # type: ignore[arg-type]
    assert decision.outcome == OUTCOME_ON_SITE
    assert decision.modes == ("tap_to_pay", "cash")


def test_facture_patient_exige_le_texte_approuve() -> None:
    recorded = propose_policy(
        (),
        policy(
            modality="patient_invoice_after",
            modes=frozenset({"invoice"}),
            invoice_signature_text="Texte approuvé, version d'essai.",
            invoice_signature_text_version="2026-10-essai",
        ),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    decision = evaluate(snapshot_from_policy(recorded.active, **_snap_kwargs()))  # type: ignore[arg-type]
    assert decision.outcome == OUTCOME_PATIENT_INVOICE
    assert decision.signature_required is True
    assert decision.signature_document == "invoice_acknowledgement"


def test_institution_verrouillee_encaisse_le_mode_prevu() -> None:
    recorded = propose_policy(
        (),
        policy(
            decision_origin="institution",
            financial_responsibility_locked=True,
            modes=frozenset({"cash"}),
        ),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    fields = _snap_kwargs()
    fields["created_via"] = "institution_portal"
    fields["billing_intent"] = "patient"
    decision = evaluate(snapshot_from_policy(recorded.active, **fields))  # type: ignore[arg-type]
    assert decision.outcome == OUTCOME_ON_SITE
    assert decision.modes == ("cash",)


def test_institution_non_verrouillee_meme_si_controle_valide() -> None:
    """Le statut validated du contrôle ne verrouille pas le payeur."""
    recorded = propose_policy(
        (),
        policy(decision_origin="institution", financial_responsibility_locked=False),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    fields = _snap_kwargs()
    fields["created_via"] = "institution_portal"
    fields["billing_intent"] = "patient"
    decision = evaluate(snapshot_from_policy(recorded.active, **fields))  # type: ignore[arg-type]
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("payer_still_mutable",)


def test_compte_mensuel_et_clinique_sans_ecran() -> None:
    monthly = propose_policy(
        (),
        policy(modality="established_account", modes=frozenset()),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    clinic_fields = _snap_kwargs()
    clinic_fields["created_via"] = "institution_portal"
    clinic_fields["billed_to_type"] = "clinic"
    clinic_fields["billing_intent"] = "institution"
    clinic = snapshot_from_policy(None, **clinic_fields)  # type: ignore[arg-type]
    monthly_decision = evaluate(
        snapshot_from_policy(monthly.active, **_snap_kwargs())  # type: ignore[arg-type]
    )
    assert monthly_decision.outcome == OUTCOME_NO_COLLECTION
    assert evaluate(clinic).reasons == ("institution_payer",)


def test_prix_divergent_bloque() -> None:
    recorded = propose_policy(
        (),
        policy(amount_chf=Decimal("90.00")),
        payment_confirmed=False,
        invoice_issued=False,
        cash_recorded=False,
    )
    decision = evaluate(snapshot_from_policy(recorded.active, **_snap_kwargs()))  # type: ignore[arg-type]
    assert decision.outcome == OUTCOME_ADMIN
    assert decision.reasons == ("price_policy_mismatch",)


def test_deux_chauffeurs_ne_traitent_pas_le_meme_reglement() -> None:
    first, granted = claim_collection(None, 7)
    again, same = claim_collection(first, 7)
    other, blocked = claim_collection(first, 8)
    assert granted == "granted"
    assert same == "same_claim"
    assert again.driver_id == 7
    assert blocked == "concurrent_claim"
    assert other.driver_id == 7


def test_caisse_refuse_une_autre_entreprise_et_une_facture_existante() -> None:
    wrong, reason = record_cash_ledger(
        (),
        booking_company_id=10,
        receiving_company_id=11,
        driver_id=7,
        expected_amount_chf=Decimal("45.00"),
        received_amount_chf=Decimal("45.00"),
        receipt_reference="recu-1",
        invoice_already_issued=False,
    )
    duplicate, duplicate_reason = record_cash_ledger(
        (),
        booking_company_id=10,
        receiving_company_id=10,
        driver_id=7,
        expected_amount_chf=Decimal("45.00"),
        received_amount_chf=Decimal("45.00"),
        receipt_reference="recu-1",
        invoice_already_issued=True,
    )
    assert wrong == ()
    assert reason == "wrong_company"
    assert duplicate == ()
    assert duplicate_reason == "invoice_already_issued"


def test_correction_d_especes_conserve_la_ligne_initiale() -> None:
    first, _reason = record_cash_ledger(
        (),
        booking_company_id=10,
        receiving_company_id=10,
        driver_id=7,
        expected_amount_chf=Decimal("45.00"),
        received_amount_chf=Decimal("40.00"),
        receipt_reference="recu-1",
        invoice_already_issued=False,
    )
    corrected, reason = record_cash_ledger(
        first,
        booking_company_id=10,
        receiving_company_id=10,
        driver_id=7,
        expected_amount_chf=Decimal("45.00"),
        received_amount_chf=Decimal("45.00"),
        receipt_reference="recu-1-corrige",
        invoice_already_issued=False,
        corrects_entry_id=1,
    )
    assert reason == "cash_declared"
    assert len(corrected) == 2
    assert corrected[0].received_amount_chf == Decimal("40.00")
    assert corrected[1].supersedes_entry_id == 1
    assert corrected[1].electronic_confirmation is False


def test_chauffeur_ne_configure_pas_le_reglement() -> None:
    assert actor_may("driver", "change_payer") is False
    assert actor_may("driver", "widen_modes") is False
    assert actor_may("driver", "sign_for_patient") is False
    assert actor_may("guest", "choose_modality") is False
    assert actor_may("private_client", "choose_modality") is False
    assert actor_may("institution", "choose_modality") is False
    assert actor_may("institution", "designate_payer") is True
    assert actor_may("company", "configure_capabilities") is True
    assert actor_may("company", "activate_collection") is False
    assert actor_may("lirie_operator", "lock_payer_in_live_portal") is False
    assert actor_may("driver", "consult_decision") is True
    assert actor_may("driver", "execute_collection") is False
    assert actor_may("driver", "activate_collection") is False
    assert actor_may("lirie_operator", "prepare_payer_lock") is True
    assert actor_may("lirie_operator", "exceptional_correction") is False
    assert actor_may("institution", "exceptional_correction") is False


def test_signature_patient_n_est_pas_un_paiement() -> None:
    recorded = review_billing_signature(
        signatory_role="patient",
        signatory_user_id=3,
        driver_user_id=9,
        text_present=True,
        version_present=True,
    )
    missing = review_billing_signature(
        signatory_role="patient",
        signatory_user_id=3,
        driver_user_id=9,
        text_present=False,
        version_present=False,
    )
    driver = review_billing_signature(
        signatory_role="driver",
        signatory_user_id=9,
        driver_user_id=9,
        text_present=True,
        version_present=True,
    )
    other = review_billing_signature(
        signatory_role="clerk",
        signatory_user_id=4,
        driver_user_id=9,
        text_present=True,
        version_present=True,
    )
    assert recorded.signature_recorded is True
    assert recorded.payment_received is False
    assert recorded.invoice_accepted is False
    assert missing.reason == "signature_text_required"
    assert driver.reason == "driver_cannot_sign"
    assert other.reason == "signatory_not_allowed"


def test_changement_de_payeur_conflit_avec_un_verrou_ou_un_encaissement() -> None:
    assert (
        payer_change_conflict(
            financial_responsibility_locked=True, collection_in_progress=False
        )
        == "payer_locked"
    )
    assert (
        payer_change_conflict(
            financial_responsibility_locked=False, collection_in_progress=True
        )
        == "collection_in_progress"
    )
    source = (
        Path(__file__).resolve().parents[3]
        / "application"
        / "institutions"
        / "change_booking_payer.py"
    ).read_text(encoding="utf-8")
    assert "refuse_ordinary_payer_change" in source
    assert "end_of_trip_settlement" not in source


def test_contrat_backend_ne_se_reconstruit_pas_sur_le_telephone() -> None:
    paid = evaluate(
        snapshot_from_policy(
            None,
            **{
                **_snap_kwargs(),
                "online_payment_status": "completed",
                "online_payment_covers_scope": True,
                "created_via": "public_guest",
            },
        )  # type: ignore[arg-type]
    )
    contract = decision_contract(paid)
    assert contract["kind"] == "already_paid"
    assert contract["payment_received"] is True
    assert contract["invoice_means_paid"] is False
    assert contract["reconstruct_locally"] is False
    assert contract["price_computed_locally"] is False
    assert contract["payer_computed_locally"] is False
    assert contract["source"] == "backend"
    assert contract["reason"] == "payment_confirmed"
