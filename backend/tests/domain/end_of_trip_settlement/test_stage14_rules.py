"""Règles V1 des espèces et de la facture. Rien n'est approuvé ni émis."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from domain.end_of_trip_settlement.billing_handoff import (
    advance_billing_state,
    open_billing_request,
)
from domain.end_of_trip_settlement.cash_procedure import (
    RECEIPT_TEXT,
    RECEIPT_TEXT_STATUS,
    assess_cash_settlement,
    consult_cash,
    project_till,
    propose_cash_receipt,
    reprint_cash_receipt,
    submit_cash_declaration,
)
from domain.end_of_trip_settlement.group_collection import plan_round_trip
from domain.end_of_trip_settlement.patient_invoice_procedure import (
    DOCUMENT_TEXT,
    DOCUMENT_VERSION,
    DOCUMENT_VERSION_REVISED,
    REVISED_TEXT,
    SignatureDocument,
    attempt_financial_signature,
    draft_signature_document,
    prepare_patient_invoice,
    revise_signature_document,
)

_NOW = datetime(2026, 10, 26, 18, 0, tzinfo=UTC)


def _receipt(source: str, amounts):
    return propose_cash_receipt(
        source=source,
        issuer_company_id=10,
        issuer_name="Société de transport fictive",
        booking_reference="R-8",
        payment_scope="booking",
        debtor_label="Patient fictif",
        service_label="Transport",
        amounts=amounts,
        service_date="26.10.2026",
        received_at=_NOW,
        confirmed_recorded=source == "recorded",
    )


def test_monnaie_explicite_et_partiel_bloque() -> None:
    exact = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("45.00"),
        change_returned_chf=Decimal("0.00"),
    )
    change = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("50.00"),
        change_returned_chf=Decimal("5.00"),
        change_handed_back=True,
    )
    guessed = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("50.00"),
        change_returned_chf=None,
    )
    short = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("30.00"),
        change_returned_chf=Decimal("0.00"),
    )
    mismatch = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("50.00"),
        change_returned_chf=Decimal("1.00"),
    )
    assert exact.situation == "exact"
    assert exact.net_collected_chf == Decimal("45.00")
    assert change.situation == "full_with_explicit_change"
    assert change.change_returned_chf == Decimal("5.00")
    assert change.net_collected_chf == Decimal("45.00")
    assert change.payment_confirmed is False
    assert guessed.situation == "surplus_not_returned"
    assert guessed.net_collected_chf is None
    assert short.situation == "partial_forbidden"
    assert short.net_collected_chf is None
    assert mismatch.situation == "change_mismatch"
    for item in (exact, change, guessed, short, mismatch):
        assert item.persists is False
        assert item.change_policy == "pending_approval"
        assert item.partial_policy == "pending_approval"


def test_recu_non_emis_avant_reception_enregistree() -> None:
    exact = assess_cash_settlement(
        amount_due_chf=Decimal("45.00"),
        amount_tendered_chf=Decimal("45.00"),
        change_returned_chf=Decimal("0.00"),
    )
    selected = _receipt("mode_selected", exact)
    announced = _receipt("patient_announced", exact)
    simulated = _receipt("simulated_success", exact)
    pending = _receipt("declaration_pending", exact)
    short = _receipt(
        "recorded",
        assess_cash_settlement(
            amount_due_chf=Decimal("45.00"),
            amount_tendered_chf=Decimal("30.00"),
            change_returned_chf=Decimal("0.00"),
        ),
    )
    ready = _receipt("recorded", exact)
    assert selected.blocked_reason == "mode_selected"
    assert announced.blocked_reason == "patient_announced"
    assert simulated.blocked_reason == "simulated_success"
    assert pending.blocked_reason == "declaration_pending"
    assert short.blocked_reason == "partial_forbidden"
    assert ready.blocked_reason == "issuance_disabled"
    assert ready.issuer_name == "Société de transport fictive"
    assert ready.receipt_number is None
    assert ready.statement == RECEIPT_TEXT
    assert ready.statement_status == RECEIPT_TEXT_STATUS
    assert RECEIPT_TEXT_STATUS == "draft"
    assert ready.issued is False
    assert ready.transmitted is False
    assert ready.generation_allowed is False
    assert ready.replaces_attestation is False
    assert ready.replaces_invoice is False
    assert ready.medical_details_included is False
    assert ready.fiscal_lines == ()
    assert ready.net_collected_chf == Decimal("45.00")
    assert ready.balance_chf == Decimal("0.00")
    copy = reprint_cash_receipt(ready)
    assert copy.duplicate is True
    assert copy.receipt_number == ready.receipt_number
    assert copy.issued is False


def test_caisse_separe_encaissement_remise_et_rapprochement() -> None:
    prepared = project_till("declaration_prepared")
    received = project_till("cash_confirmed_received")
    waiting = project_till("awaiting_company_remittance")
    remitted = project_till("remittance_recorded")
    reconciled = project_till("reconciled")
    variance = project_till("variance_to_check")
    dispute = project_till("correction_or_dispute")
    assert prepared.patient_payment_count == 0
    assert received.patient_payment_count == 1
    assert waiting.patient_payment_count == 1
    assert remitted.patient_payment_count == 1
    assert reconciled.patient_payment_count == 1
    assert variance.state == "variance_to_check"
    assert dispute.state == "correction_or_dispute"
    for item in (prepared, received, waiting, remitted, reconciled, variance, dispute):
        assert item.persists is False


def test_texte_revise_reste_un_brouillon() -> None:
    original = draft_signature_document(
        version=DOCUMENT_VERSION, text=DOCUMENT_TEXT
    )
    kept, revised = revise_signature_document(
        original, version=DOCUMENT_VERSION_REVISED, text=REVISED_TEXT
    )
    assert kept.text == DOCUMENT_TEXT
    assert kept.version == DOCUMENT_VERSION
    assert revised.version == DOCUMENT_VERSION_REVISED
    assert revised.status == "draft"
    assert revised.approved is False
    assert revised.approved_at is None
    assert revised.approved_by is None
    assert revised.prefilled is False
    assert "paiement reçu" in revised.text
    assert "reconnaissance de dette" in revised.text
    prepared = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=False,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=False,
        invoice_already_issued=False,
        debtor_label="Patient fictif",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=revised,
    )
    assert prepared.reason == "signature_text_unapproved"
    assert prepared.signature_persisted is False


def test_exceptions_de_signature_ne_fabriquent_rien() -> None:
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
        debtor_label="Patient fictif",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=document,
    )
    cases = (
        ("patient", "refused"),
        ("patient", "unable"),
        ("patient", "not_understood"),
        ("representative", "representative"),
        ("patient", "phone_unavailable"),
        ("patient", "offline"),
        ("patient", "document_unavailable"),
        ("patient", "document_withdrawn"),
        ("patient", "amount_changed"),
        ("patient", "payer_changed"),
        ("patient", "already_invoiced"),
        ("patient", "already_paid"),
    )
    for role, situation in cases:
        result = attempt_financial_signature(
            prepared, signatory_role=role, situation=situation
        )
        assert result.signature_persisted is False
        assert result.invoice_issued is False
        assert result.invoice_means_paid is False
        assert result.amount_chf == Decimal("45.00")
        assert result.administrative is True
    prefilled = attempt_financial_signature(
        prepared, signatory_role="patient", situation=None, prefilled=True
    )
    assert prefilled.reason == "signature_prefilled"
    assert prefilled.signature_persisted is False
    withdrawn = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=False,
        company_invoice_enabled=True,
        policy_allows_invoice=True,
        already_paid=False,
        invoice_already_issued=False,
        debtor_label="Patient fictif",
        amount_chf=Decimal("45.00"),
        company_id=10,
        document=SignatureDocument(
            version=DOCUMENT_VERSION_REVISED,
            text=REVISED_TEXT,
            approved=False,
            legal_qualification="pending_approval",
            replaces_transport_attestation=False,
            means_payment_received=False,
            debt_acknowledgment=False,
            status="withdrawn",
        ),
    )
    assert withdrawn.reason == "document_withdrawn"
    assert withdrawn.amount_chf == Decimal("45.00")


def test_facture_ne_saute_pas_les_etats() -> None:
    request = open_billing_request(scope_key="booking-8", already_issued=False)
    skipped = advance_billing_state(request, "invoice_issued", explicit=True)
    implicit = advance_billing_state(request, "signature_recorded", explicit=False)
    signed = advance_billing_state(request, "signature_recorded", explicit=True)
    covered = open_billing_request(scope_key="booking-8", already_issued=True)
    assert request.state == "billing_request"
    assert request.persists is False
    assert skipped.reason == "state_skip_refused"
    assert skipped.invoice_issued is False
    assert implicit.reason == "implicit_transition_refused"
    assert signed.state == "signature_recorded"
    assert signed.signature_means_issued is False
    assert signed.signature_means_paid is False
    assert signed.invoice_issued is False
    assert covered.reason == "invoice_already_issued"
    assert covered.second_invoice is False


def test_aller_retour_perime_la_decision_si_le_groupe_change() -> None:
    current = plan_round_trip(
        outbound_chf=Decimal("45.00"),
        return_chf=Decimal("45.00"),
        already_paid_chf=Decimal("0.00"),
        separate_leg_approved=False,
    )
    stale = plan_round_trip(
        outbound_chf=Decimal("45.00"),
        return_chf=Decimal("45.00"),
        already_paid_chf=Decimal("0.00"),
        separate_leg_approved=False,
        group_revision=2,
        decision_revision=1,
    )
    paid = plan_round_trip(
        outbound_chf=Decimal("45.00"),
        return_chf=Decimal("45.00"),
        already_paid_chf=Decimal("90.00"),
        separate_leg_approved=False,
        group_revision=2,
        decision_revision=1,
    )
    assert current.receipt_scope == "group"
    assert current.invoice_scope == "outbound_and_return"
    assert current.driver_may_split is False
    assert current.collection_moment_status == "pending_approval"
    assert current.collection_count == 1
    assert stale.reason == "group_decision_stale"
    assert stale.collection_count == 0
    assert stale.amount_still_due_chf == Decimal("90.00")
    assert paid.reason == "already_paid"
    assert paid.second_request is False
    consultation = consult_cash(
        booking_id=8,
        payment_scope="group",
        amount_due_chf=Decimal("90.00"),
        beneficiary_company_id=10,
    )
    stored, reason = submit_cash_declaration(
        (),
        consultation=consultation,
        amounts=assess_cash_settlement(
            amount_due_chf=Decimal("90.00"),
            amount_tendered_chf=Decimal("90.00"),
            change_returned_chf=Decimal("0.00"),
        ),
        driver_id=3,
        idempotency_key="groupe-90",
        recorded_at=_NOW,
    )
    assert reason == "declared"
    assert stored[0].persists is False
    assert stored[0].amounts.payment_confirmed is False
