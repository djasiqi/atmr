"""Validation métier simulée. Aucune somme n'est encaissée."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from domain.end_of_trip_settlement.activation import (
    END_OF_TRIP_COLLECTION_ENABLED,
)
from domain.end_of_trip_settlement.cash_exception import (
    demonstrate_cash_receipt,
    reconnect_local_declaration,
    reprint_demo,
    trace_insufficient_cash,
)
from domain.end_of_trip_settlement.cash_procedure import assess_cash_settlement
from domain.end_of_trip_settlement.company_configuration import (
    CompanySettlementSettings,
)
from domain.end_of_trip_settlement.eligibility import (
    OUTCOME_ADMIN,
    OUTCOME_ALREADY_PAID,
    OUTCOME_INSTITUTION,
    OUTCOME_NO_COLLECTION,
    SettlementSnapshot,
    evaluate,
)
from domain.end_of_trip_settlement.group_collection import plan_round_trip
from domain.end_of_trip_settlement.patient_invoice_procedure import (
    DOCUMENT_VERSION_REVISED,
    REVISED_TEXT,
    draft_signature_document,
    present_billing_document,
)

_ROOT = Path(__file__).resolve().parents[3]


def snap(**overrides: object) -> SettlementSnapshot:
    base: dict[str, object] = {
        "booking_id": 8,
        "company_id": 10,
        "executing_company_id": 10,
        "created_via": "client_app",
        "billed_to_type": "patient",
        "billing_intent": None,
        "billing_party_id": 5,
        "settlement_modality": "on_site",
        "reservation_modes": frozenset({"cash"}),
        "company_modes": frozenset({"cash"}),
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
        "payment_scope": None,
        "collection_moment": None,
        "group_already_settled": False,
        "debtor_display_name": "Patient fictif",
        "policy_recorded": True,
    }
    base.update(overrides)
    return SettlementSnapshot(**base)  # type: ignore[arg-type]


def test_trente_francs_rendus_ne_laissent_aucune_detention() -> None:
    returned = trace_insufficient_cash(
        amount_due_chf=Decimal("45.00"),
        tendered_chf=Decimal("30.00"),
        disposition="returned",
        idempotency_key="court-1",
    )
    assert returned.reason == "cash_returned"
    assert returned.physically_held_chf == Decimal("0.00")
    assert returned.officially_accepted_chf == Decimal("0.00")
    assert returned.balance_still_due_chf == Decimal("45.00")
    assert returned.payment_accepted is False
    assert returned.receipt_issued is False
    assert returned.custody_traced is False
    assert returned.mission_close_blocked is False
    assert returned.partial_policy_changed is False
    assert returned.persists is False


def test_trente_francs_detenus_ne_soldent_pas_la_course() -> None:
    held = trace_insufficient_cash(
        amount_due_chf=Decimal("45.00"),
        tendered_chf=Decimal("30.00"),
        disposition="held",
        idempotency_key="detention-1",
    )
    unknown = trace_insufficient_cash(
        amount_due_chf=Decimal("45.00"),
        tendered_chf=Decimal("30.00"),
        disposition=None,
        idempotency_key="detention-1",
    )
    nothing = trace_insufficient_cash(
        amount_due_chf=Decimal("45.00"),
        tendered_chf=Decimal("0.00"),
        disposition=None,
        idempotency_key="rien",
    )
    assert held.reason == "cash_held_not_settlement"
    assert held.physically_held_chf == Decimal("30.00")
    assert held.officially_accepted_chf == Decimal("0.00")
    assert held.balance_still_due_chf == Decimal("45.00")
    assert held.custody_traced is True
    assert held.custody_reference == "detention-1"
    assert held.restitution_possible is True
    assert held.payment_accepted is False
    assert held.receipt_issued is False
    assert held.administrative is True
    assert held.partial_policy_changed is False
    assert unknown.reason == "disposition_required"
    assert unknown.physically_held_chf == Decimal("0.00")
    assert nothing.reason == "nothing_remitted"
    assert nothing.custody_traced is False


def test_monnaie_non_confirmee_n_est_pas_un_pourboire() -> None:
    pending = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("50.00"),
        change_returned_chf=Decimal("5.00"),
        change_handed_back=False,
    )
    confirmed = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("50.00"),
        change_returned_chf=Decimal("5.00"),
        change_handed_back=True,
    )
    assert pending.situation == "change_not_confirmed"
    assert pending.net_collected_chf is None
    assert pending.payment_confirmed is False
    assert confirmed.situation == "full_with_explicit_change"
    assert confirmed.net_collected_chf == Decimal("45.00")
    assert confirmed.payment_confirmed is False


def test_recu_de_demonstration_n_est_pas_emis() -> None:
    demo = demonstrate_cash_receipt(
        situation="exact",
        issuer_name="Société de transport fictive",
        idempotency_key="recu-1",
        scope_label="Réservation fictive 8",
        service_label="Transport",
        service_date="26.10.2026",
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("45.00"),
        change_returned_chf=Decimal("0.00"),
        net_collected_chf=Decimal("45.00"),
    )
    blocked = demonstrate_cash_receipt(
        situation="partial_forbidden",
        issuer_name="Société de transport fictive",
        idempotency_key="recu-2",
        scope_label="Réservation fictive 8",
        service_label="Transport",
        service_date="26.10.2026",
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("30.00"),
        change_returned_chf=Decimal("0.00"),
        net_collected_chf=None,
    )
    copy = reprint_demo(demo)
    assert demo.banner == "DRAFT — PENDING APPROVAL"
    assert demo.proposed_reference == "DEMO-recu-1"
    assert demo.reference_allocated is False
    assert demo.fiscal_identifier is None
    assert demo.issued is False
    assert demo.transmitted is False
    assert demo.collection_writes == 0
    assert "Reçu d'espèces" in demo.cash_mention
    assert blocked.proposed_reference is None
    assert blocked.net_collected_chf is None
    assert copy.duplicate is True
    assert copy.proposed_reference == demo.proposed_reference
    assert copy.collection_writes == 0


def test_document_facture_reste_une_prise_de_connaissance() -> None:
    document = draft_signature_document(
        version=DOCUMENT_VERSION_REVISED, text=REVISED_TEXT
    )
    shown = present_billing_document(
        debtor_label="Patient fictif",
        issuer_name="Société de transport fictive",
        service_label="Transport",
        amount_chf=Decimal("45.00"),
        document=document,
    )
    assert shown.version == DOCUMENT_VERSION_REVISED
    assert shown.awareness_only is True
    assert shown.contractual_acceptance_activated is False
    assert shown.mandatory_signature_enabled is False
    assert shown.means_payment is False
    assert shown.means_attestation is False
    assert shown.debt_acknowledgment is False
    assert shown.recovery_guarantee is False
    assert shown.full_conditions_available is True
    assert shown.banner == "DRAFT — PENDING APPROVAL"


def test_reconnexion_ne_confirme_pas_la_declaration_locale() -> None:
    lost = reconnect_local_declaration(
        local_marked_confirmed=True,
        server_payment_exists=False,
        invoice_exists=False,
        amount_changed=False,
    )
    conflict = reconnect_local_declaration(
        local_marked_confirmed=False,
        server_payment_exists=True,
        invoice_exists=True,
        amount_changed=True,
    )
    assert lost.payment_confirmed is False
    assert lost.queued_collection is False
    assert lost.administrative is True
    assert lost.mission_preserved is True
    assert conflict.payment_confirmed is False
    assert conflict.persists is False


def test_limites_financieres_restent_fermees() -> None:
    settings = CompanySettlementSettings()
    institution = evaluate(snap(billed_to_type="clinic", billing_party_id=None))
    unlocked = evaluate(
        snap(
            created_via="institution_portal",
            billing_intent="patient",
            financial_responsibility_locked=False,
        )
    )
    monthly = evaluate(snap(settlement_modality="established_account"))
    guest = evaluate(
        snap(
            created_via="public_guest",
            online_payment_status="completed",
            online_payment_covers_scope=True,
        )
    )
    group = plan_round_trip(
        outbound_chf=Decimal("45.00"),
        return_chf=Decimal("45.00"),
        already_paid_chf=Decimal("90.00"),
        separate_leg_approved=False,
    )
    invoiced = evaluate(snap(invoice_already_issued=True))
    missing = evaluate(snap(policy_recorded=False))
    other_company = evaluate(snap(executing_company_id=99))
    assert END_OF_TRIP_COLLECTION_ENABLED is False
    assert settings.cash_enabled is False
    assert settings.direct_patient_invoice_enabled is False
    assert institution.outcome == OUTCOME_INSTITUTION
    assert unlocked.reasons == ("payer_still_mutable",)
    assert monthly.outcome == OUTCOME_NO_COLLECTION
    assert guest.outcome == OUTCOME_ALREADY_PAID
    assert group.second_request is False
    assert group.collection_count == 0
    assert invoiced.reasons == ("invoice_already_issued",)
    assert missing.outcome == OUTCOME_ADMIN
    assert other_company.reasons == ("company_not_owner",)


def test_parite_des_tests_historiques_est_documentee() -> None:
    historical = {
        "test_changement_ordinaire_refuse_apres_verrou_puis_annule": (
            "test_verrou_refuse_le_changement_puis_disparait"
        ),
        "test_deux_transactions_se_serialisent_sur_la_reservation": (
            "test_deux_connexions_se_serialisent"
        ),
        "test_politique_inseree_puis_annulee_sans_toucher_la_reservation": (
            "test_politique_active_unique_et_verrou_complet"
        ),
        "test_preferences_n_activent_pas_les_moyens": (
            "test_preferences_n_activent_pas_les_moyens"
        ),
        "test_declaration_patient_n_est_pas_une_preuve": (
            "test_declaration_patient_n_est_pas_une_preuve"
        ),
        "test_acceptation_ne_cree_pas_de_politique_active": (
            "test_acceptation_ne_cree_pas_de_politique"
        ),
        "test_politique_confirmee_n_est_pas_ecrasee": (
            "test_politique_confirmee_n_est_pas_ecrasee"
        ),
    }
    isolated = (_ROOT / "tests/e2e/test_stage14_persistence_isolated.py").read_text(
        encoding="utf-8"
    )
    originals = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            _ROOT / "tests/domain/end_of_trip_settlement/test_payer_lock_db.py",
            _ROOT / "tests/domain/end_of_trip_settlement/test_policy_persistence.py",
            _ROOT
            / "tests/domain/end_of_trip_settlement/test_settlement_settings_db.py",
        )
    )
    assert len(historical) == 7
    for old, new in historical.items():
        assert f"def {old}" in originals
        assert "atmr_test" in originals
        assert f"def {new}" in isolated
    assert "lirie_e2e_" in isolated
    assert "create_app" not in isolated
