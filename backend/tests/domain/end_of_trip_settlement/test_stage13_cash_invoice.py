"""Espèces et facture patient, en simulation. Aucune écriture financière."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from domain.end_of_trip_settlement.access import actor_may
from domain.end_of_trip_settlement.activation import (
    END_OF_TRIP_COLLECTION_ENABLED,
    driver_panel_enabled,
    simulation_preview_allowed,
)
from domain.end_of_trip_settlement.cash_procedure import (
    audit_cash_correction,
    cash_receipt_draft,
    consult_cash,
    describe_cash_amounts,
    reconcile_cash,
    submit_cash_declaration,
    visible_cash_declarations,
)
from domain.end_of_trip_settlement.company_configuration import (
    CompanySettlementSettings,
)
from domain.end_of_trip_settlement.destination_event import (
    DESTINATION_ARRIVAL_EVENT_ENABLED,
    propose_destination_arrival,
    record_destination_arrival,
)
from domain.end_of_trip_settlement.eligibility import (
    OUTCOME_ADMIN,
    OUTCOME_ALREADY_PAID,
    OUTCOME_INSTITUTION,
    OUTCOME_NO_COLLECTION,
    OUTCOME_ON_SITE,
    OUTCOME_PATIENT_INVOICE,
    SettlementSnapshot,
    driver_collection_view,
    evaluate,
)
from domain.end_of_trip_settlement.group_collection import plan_round_trip
from domain.end_of_trip_settlement.patient_invoice_procedure import (
    DOCUMENT_STATUS,
    DOCUMENT_TEXT,
    DOCUMENT_VERSION,
    SignatureDocument,
    attempt_financial_signature,
    draft_signature_document,
    prepare_patient_invoice,
)

_ROOT = Path(__file__).resolve().parents[3]
_NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
_LIVE_PATHS = (
    "application/drivers/update_driver_booking_status.py",
    "routes/driver_settlement_read.py",
    "services/booking/status_transitions.py",
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
        "reservation_modes": frozenset({"cash", "tap_to_pay"}),
        "company_modes": frozenset({"cash", "tap_to_pay", "invoice"}),
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


def _declare(tendered: str, *, key: str = "cash-1", prior: tuple = ()):
    consultation = consult_cash(
        booking_id=8,
        payment_scope="booking",
        amount_due_chf=Decimal("45.00"),
        beneficiary_company_id=10,
    )
    amounts = describe_cash_amounts(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal(tendered),
    )
    return submit_cash_declaration(
        prior,
        consultation=consultation,
        amounts=amounts,
        driver_id=3,
        idempotency_key=key,
        recorded_at=_NOW,
    )


def test_choix_especes_n_est_pas_un_paiement() -> None:
    shown = consult_cash(
        booking_id=8,
        payment_scope="booking",
        amount_due_chf=Decimal("45.00"),
        beneficiary_company_id=10,
    )
    assert shown.mode == "cash"
    assert shown.currency == "CHF"
    assert shown.amount_due_chf == Decimal("45.00")
    assert shown.payment_confirmed is False
    assert shown.persists is False
    assert shown.confirmation_required is True


def test_montants_especes_restent_distincts() -> None:
    exact = describe_cash_amounts(
        amount_due_chf=Decimal("45.00"), amount_tendered_chf=Decimal("45.00")
    )
    change = describe_cash_amounts(
        amount_due_chf=Decimal("45.00"), amount_tendered_chf=Decimal("50.00")
    )
    short = describe_cash_amounts(
        amount_due_chf=Decimal("45.00"), amount_tendered_chf=Decimal("20.00")
    )
    absent = describe_cash_amounts(
        amount_due_chf=Decimal("45.00"), amount_tendered_chf=Decimal("0.00")
    )
    error = describe_cash_amounts(
        amount_due_chf=Decimal("45.00"), amount_tendered_chf=None
    )
    assert exact.situation == "exact"
    assert exact.net_collected_chf == Decimal("45.00")
    assert exact.change_returned_chf == Decimal("0.00")
    assert exact.reconciled_amount_chf is None
    assert exact.payment_confirmed is False
    assert change.situation == "surplus_not_returned"
    assert change.change_returned_chf is None
    assert change.net_collected_chf is None
    assert change.change_policy == "pending_approval"
    assert short.situation == "partial_forbidden"
    assert short.net_collected_chf is None
    assert short.partial_policy == "pending_approval"
    assert absent.situation == "absent"
    assert error.situation == "declaration_error"
    for item in (exact, change, short, absent, error):
        assert item.persists is False


def test_declaration_confirmee_ne_se_corrige_pas_en_silence() -> None:
    first, reason = _declare("45.00")
    assert reason == "declared"
    assert first[0].frozen is True
    assert first[0].persists is False
    replay, replay_reason = _declare("45.00", prior=first)
    assert replay_reason == "replay"
    assert replay == first
    silent, silent_reason = _declare("45.00", key="cash-2", prior=first)
    assert silent_reason == "silent_correction_refused"
    assert silent == first
    surplus, surplus_reason = submit_cash_declaration(
        first,
        consultation=consult_cash(
            booking_id=8,
            payment_scope="booking",
            amount_due_chf=Decimal("45.00"),
            beneficiary_company_id=10,
        ),
        amounts=describe_cash_amounts(
            amount_due_chf=Decimal("45.00"),
            amount_tendered_chf=Decimal("50.00"),
        ),
        driver_id=3,
        idempotency_key="cash-surplus",
        recorded_at=_NOW,
    )
    assert surplus_reason == "surplus_not_returned"
    assert surplus == first
    event, event_reason = audit_cash_correction(
        first[0],
        reason="Montant remis relu avec le patient.",
        author_role="company",
        author_id=12,
        recorded_at=_NOW,
    )
    assert event_reason == "correction_prepared"
    assert event is not None
    assert event.original_key == "cash-1"
    assert event.persists is False
    assert first[0].amounts.net_collected_chf == Decimal("45.00")
    silent_author, silent_author_reason = audit_cash_correction(
        first[0],
        reason="Correction du chauffeur.",
        author_role="driver",
        author_id=3,
        recorded_at=_NOW,
    )
    assert silent_author is None
    assert silent_author_reason == "author_not_authorized"


def test_caisse_isolee_par_entreprise_et_chauffeur() -> None:
    company_a, _reason = _declare("45.00", key="a")
    other = consult_cash(
        booking_id=9,
        payment_scope="booking",
        amount_due_chf=Decimal("30.00"),
        beneficiary_company_id=20,
    )
    amounts = describe_cash_amounts(
        amount_due_chf=Decimal("30.00"), amount_tendered_chf=Decimal("30.00")
    )
    both, _other_reason = submit_cash_declaration(
        company_a,
        consultation=other,
        amounts=amounts,
        driver_id=4,
        idempotency_key="b",
        recorded_at=_NOW,
    )
    company_view = visible_cash_declarations(
        both,
        reader_company_id=10,
        reader_role="company",
        reader_driver_id=None,
        driver_may_see_company_register=False,
    )
    driver_view = visible_cash_declarations(
        both,
        reader_company_id=10,
        reader_role="driver",
        reader_driver_id=3,
        driver_may_see_company_register=False,
    )
    foreign = visible_cash_declarations(
        both,
        reader_company_id=99,
        reader_role="company",
        reader_driver_id=None,
        driver_may_see_company_register=True,
    )
    assert [item.company_id for item in company_view] == [10]
    assert [item.driver_id for item in driver_view] == [3]
    assert foreign == ()
    assert (
        reconcile_cash(
            company_view[0],
            actor_role="driver",
            actor_company_id=10,
            reconciled_amount_chf=Decimal("45.00"),
        )
        == "reconciliation_refused"
    )
    assert (
        reconcile_cash(
            company_view[0],
            actor_role="company",
            actor_company_id=20,
            reconciled_amount_chf=Decimal("45.00"),
        )
        == "reconciliation_refused"
    )
    assert (
        reconcile_cash(
            company_view[0],
            actor_role="company",
            actor_company_id=10,
            reconciled_amount_chf=Decimal("45.00"),
        )
        == "reconciliation_prepared"
    )
    assert company_view[0].persists is False
    receipt = cash_receipt_draft(company_view[0])
    assert receipt.issued is False
    assert receipt.bank_transfer is False
    assert receipt.platform_fee is False
    assert receipt.currency == "CHF"


def test_document_de_signature_reste_non_approuve() -> None:
    document = draft_signature_document(version=DOCUMENT_VERSION, text=DOCUMENT_TEXT)
    assert DOCUMENT_STATUS == "pending_approval"
    assert document.approved is False
    assert document.debt_acknowledgment is False
    assert document.replaces_transport_attestation is False
    assert document.means_payment_received is False
    assert "reconnaissance de dette" in document.text
    prepared = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=True,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=False,
        invoice_already_issued=False,
        debtor_label="Patient",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=document,
    )
    assert prepared.allowed is False
    assert prepared.reason == "signature_text_unapproved"
    assert prepared.invoice_issued is False
    assert prepared.signature_persisted is False
    assert prepared.attestation_separate is True
    assert prepared.amount_chf == Decimal("45.00")


def test_signature_approuvee_n_est_pas_enregistree_dans_ce_lot() -> None:
    document = SignatureDocument(
        version="future",
        text="Texte qui resterait à valider hors de ce lot.",
        approved=True,
        legal_qualification="pending_approval",
        replaces_transport_attestation=False,
        means_payment_received=False,
        debt_acknowledgment=False,
        status="approved",
    )
    prepared = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=False,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=False,
        invoice_already_issued=False,
        debtor_label="Patient",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=document,
    )
    signed = attempt_financial_signature(
        prepared, signatory_role="patient", situation=None
    )
    assert prepared.allowed is True
    assert signed.signature_persisted is False
    assert signed.invoice_issued is False
    assert signed.invoice_means_paid is False
    assert signed.reason == "signature_not_persisted"


def test_patient_qui_ne_signe_pas_garde_le_montant() -> None:
    document = draft_signature_document(version=DOCUMENT_VERSION, text=DOCUMENT_TEXT)
    prepared = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=False,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=False,
        invoice_already_issued=False,
        debtor_label="Patient",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=document,
    )
    situations = (
        ("patient", "refused", "refused"),
        ("patient", "unable", "unable"),
        ("patient", "not_understood", "not_understood"),
        ("representative", None, "representative_needs_administrative_review"),
        ("patient", "phone_unavailable", "phone_unavailable"),
        ("patient", "offline", "offline"),
        ("patient", "document_unavailable", "document_unavailable"),
        ("driver", None, "driver_cannot_sign"),
    )
    for role, situation, reason in situations:
        result = attempt_financial_signature(
            prepared, signatory_role=role, situation=situation
        )
        assert result.reason == reason
        assert result.signature_persisted is False
        assert result.amount_preserved is True
        assert result.amount_chf == Decimal("45.00")
        assert result.attestation_separate is True
        assert result.invoice_issued is False
        assert result.administrative is True


def test_facture_refusee_si_deja_couverte_ou_payeur_ouvert() -> None:
    document = draft_signature_document(version=DOCUMENT_VERSION, text=DOCUMENT_TEXT)
    covered = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=False,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=True,
        invoice_already_issued=False,
        debtor_label="Patient",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=document,
    )
    existing = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=False,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=False,
        invoice_already_issued=True,
        debtor_label="Patient",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=document,
    )
    unlocked = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=False,
        institution_origin=True,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=False,
        invoice_already_issued=False,
        debtor_label="Patient",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=document,
    )
    assert covered.reason == "already_paid"
    assert existing.reason == "invoice_already_issued"
    assert unlocked.reason == "payer_still_mutable"
    assert covered.invoice_issued is False


def test_evenement_destination_reste_inactif() -> None:
    proposed = propose_destination_arrival(
        None,
        booking_id=8,
        driver_id=3,
        assigned_driver_id=3,
        server_time=_NOW,
    )
    assert DESTINATION_ARRIVAL_EVENT_ENABLED is False
    assert proposed.applied is False
    assert proposed.reason == "not_activated"
    assert proposed.completes_mission is False
    assert proposed.creates_payment is False
    studied = record_destination_arrival(
        None,
        enabled=True,
        booking_id=8,
        driver_id=3,
        assigned_driver_id=3,
        server_time=_NOW,
    )
    replay = record_destination_arrival(
        studied,
        enabled=True,
        booking_id=8,
        driver_id=3,
        assigned_driver_id=3,
        server_time=_NOW,
    )
    stranger = record_destination_arrival(
        None,
        enabled=True,
        booking_id=8,
        driver_id=9,
        assigned_driver_id=3,
        server_time=_NOW,
    )
    assert studied.applied is True
    assert studied.completes_mission is False
    assert studied.creates_payment is False
    assert studied.geo_proof is False
    assert studied.immobilization_proof is False
    assert replay.idempotent_replay is True
    assert replay.recorded_at == studied.recorded_at
    assert stranger.reason == "driver_not_assigned"
    for relative in _LIVE_PATHS:
        source = (_ROOT / relative).read_text(encoding="utf-8")
        assert "destination_event" not in source
        assert "arrived_destination" not in source
        assert "cash_procedure" not in source
        assert "patient_invoice_procedure" not in source


def test_aller_retour_ne_demande_pas_deux_fois_les_memes_90() -> None:
    paid = plan_round_trip(
        outbound_chf=Decimal("45.00"),
        return_chf=Decimal("45.00"),
        already_paid_chf=Decimal("90.00"),
        separate_leg_approved=False,
    )
    open_group = plan_round_trip(
        outbound_chf=Decimal("45.00"),
        return_chf=Decimal("45.00"),
        already_paid_chf=Decimal("0.00"),
        separate_leg_approved=False,
    )
    flagged = plan_round_trip(
        outbound_chf=Decimal("45.00"),
        return_chf=Decimal("45.00"),
        already_paid_chf=Decimal("0.00"),
        separate_leg_approved=True,
    )
    assert paid.reason == "already_paid"
    assert paid.collection_count == 0
    assert paid.second_request is False
    assert open_group.total_chf == Decimal("90.00")
    assert open_group.collection_count == 1
    assert open_group.activated is False
    assert open_group.partial_leg_allowed is False
    assert open_group.split_across_companies is False
    assert flagged.activated is False
    assert flagged.partial_leg_allowed is False
    assert flagged.reason == "separate_leg_not_activated"


def test_regles_par_type_de_client_sans_encaissement_actif() -> None:
    private = evaluate(snap())
    locked = evaluate(
        snap(
            created_via="institution_portal",
            billing_intent="patient",
            financial_responsibility_locked=True,
        )
    )
    unlocked = evaluate(
        snap(
            created_via="institution_portal",
            billing_intent="patient",
            financial_responsibility_locked=False,
        )
    )
    institution = evaluate(snap(billed_to_type="clinic", billing_party_id=None))
    monthly = evaluate(snap(settlement_modality="established_account"))
    guest = evaluate(
        snap(
            created_via="public_guest",
            online_payment_status="completed",
            online_payment_covers_scope=True,
        )
    )
    invoiced = evaluate(snap(invoice_already_issued=True))
    missing = evaluate(snap(policy_recorded=False))
    assert private.outcome == OUTCOME_ON_SITE
    assert locked.outcome == OUTCOME_ON_SITE
    assert unlocked.outcome == OUTCOME_ADMIN
    assert unlocked.reasons == ("payer_still_mutable",)
    assert institution.outcome == OUTCOME_INSTITUTION
    assert monthly.outcome == OUTCOME_NO_COLLECTION
    assert monthly.reasons == ("established_account",)
    assert guest.outcome == OUTCOME_ALREADY_PAID
    assert invoiced.outcome == OUTCOME_NO_COLLECTION
    assert invoiced.reasons == ("invoice_already_issued",)
    assert missing.outcome == OUTCOME_ADMIN
    assert missing.reasons == ("settlement_policy_missing",)
    assert driver_collection_view(private) is None
    assert driver_collection_view(locked) is None
    direct = evaluate(
        snap(
            settlement_modality="patient_invoice_after",
            reservation_modes=frozenset({"invoice"}),
            invoice_signature_required=True,
            invoice_signature_text=DOCUMENT_TEXT,
            invoice_signature_text_version=DOCUMENT_VERSION,
        )
    )
    assert direct.outcome == OUTCOME_PATIENT_INVOICE
    assert direct.attestation_separate is True
    assert direct.driver_may_edit_amount is False
    assert direct.driver_may_edit_payer is False
    assert driver_collection_view(direct) is None


def test_capacites_financieres_restent_eteintes() -> None:
    settings = CompanySettlementSettings()
    assert END_OF_TRIP_COLLECTION_ENABLED is False
    assert DESTINATION_ARRIVAL_EVENT_ENABLED is False
    assert driver_panel_enabled() is False
    assert settings.cash_enabled is False
    assert settings.direct_patient_invoice_enabled is False
    assert settings.invoice_signature_approved is False
    assert actor_may("driver", "execute_collection") is False
    assert actor_may("driver", "change_payer") is False
    assert actor_may("driver", "sign_for_patient") is False
    assert (
        simulation_preview_allowed(
            panel_enabled=True, production=True, recipe_database=True
        )
        is False
    )
    for relative in _LIVE_PATHS:
        source = (_ROOT / relative).read_text(encoding="utf-8")
        assert "submit_cash_declaration" not in source
        assert "prepare_patient_invoice" not in source
