"""Sort fictif des espèces détenues. Aucune écriture comptable."""

from decimal import Decimal

from domain.end_of_trip_settlement.cash_exception import (
    resolve_held_cash,
    trace_insufficient_cash,
)
from domain.end_of_trip_settlement.patient_invoice_procedure import (
    DOCUMENT_VERSION_REVISED,
    REVISED_TEXT,
    attempt_financial_signature,
    draft_signature_document,
    prepare_patient_invoice,
    present_billing_document,
)

DUE = Decimal("45.00")
HELD = Decimal("30.00")


def _held():
    return trace_insufficient_cash(
        amount_due_chf=DUE,
        tendered_chf=HELD,
        disposition="held",
        idempotency_key="det-30",
    )


def test_restitution_rend_les_30_sans_reconnaitre_de_paiement() -> None:
    custody = _held()
    prior, decision = resolve_held_cash(
        custody,
        (),
        action="return",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="retour-1",
    )
    assert decision.returned is True
    assert decision.physically_held_chf == Decimal("0.00")
    assert decision.officially_accepted_chf == Decimal("0.00")
    assert decision.balance_still_due_chf == DUE
    assert decision.payment_accepted is False
    assert decision.custody_open is False
    assert decision.custody_reference == "det-30"
    assert decision.persists is False
    assert len(prior) == 1


def test_deuxieme_restitution_et_imputation_simultanee_sont_refusees() -> None:
    custody = _held()
    prior, first = resolve_held_cash(
        custody,
        (),
        action="return",
        actor_role="driver",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="retour-1",
    )
    prior, second = resolve_held_cash(
        custody,
        prior,
        action="return",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="retour-2",
    )
    assert second.reason == "double_return_refused"
    assert len(prior) == 1
    assert first.reason == "cash_returned_to_patient"
    prior, clash = resolve_held_cash(
        custody,
        prior,
        action="impute",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="impute-1",
    )
    assert clash.reason == "return_and_imputation_conflict"
    assert len(prior) == 1


def test_imputation_est_tracee_mais_pas_appliquee() -> None:
    custody = _held()
    prior, decision = resolve_held_cash(
        custody,
        (),
        action="impute",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="impute-1",
    )
    assert decision.imputation_requested is True
    assert decision.imputation_applied is False
    assert decision.proposed_balance_if_approved_chf == Decimal("15.00")
    assert decision.balance_still_due_chf == DUE
    assert decision.officially_accepted_chf == Decimal("0.00")
    assert decision.payment_accepted is False
    assert decision.policy_status == "pending_approval"
    assert decision.persists is False
    prior, again = resolve_held_cash(
        custody,
        prior,
        action="impute",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="impute-2",
    )
    assert again.reason == "double_imputation_refused"
    assert len(prior) == 1
    same, replay = resolve_held_cash(
        custody,
        prior,
        action="impute",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="impute-1",
    )
    assert replay is decision
    assert len(same) == 1


def test_litige_ne_transfere_pas_les_especes_a_l_entreprise() -> None:
    custody = _held()
    _prior, decision = resolve_held_cash(
        custody,
        (),
        action="dispute",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="litige-1",
    )
    assert decision.disputed is True
    assert decision.payment_accepted is False
    assert decision.officially_accepted_chf == Decimal("0.00")
    assert decision.balance_still_due_chf == DUE
    assert decision.custody_open is True
    assert decision.patient_settlement is False


def test_remise_a_l_entreprise_n_est_pas_un_reglement() -> None:
    custody = _held()
    prior, decision = resolve_held_cash(
        custody,
        (),
        action="remit",
        actor_role="driver",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="remise-1",
    )
    assert decision.remitted_to_company is True
    assert decision.holder == "company"
    assert decision.physically_held_chf == HELD
    assert decision.payment_accepted is False
    assert decision.patient_settlement is False
    assert decision.reconciliation_counts_as_payment is False
    assert decision.balance_still_due_chf == DUE
    _prior, second = resolve_held_cash(
        custody,
        prior,
        action="remit",
        actor_role="driver",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="remise-2",
    )
    assert second.reason == "double_remittance_refused"


def test_cloture_autre_entreprise_et_chauffeur_sont_bloques() -> None:
    custody = _held()
    prior, closed = resolve_held_cash(
        custody,
        (),
        action="close",
        actor_role="company",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="cloture",
    )
    assert closed.reason == "misleading_close_refused"
    assert closed.custody_reference == "det-30"
    assert prior == ()
    _prior, foreign = resolve_held_cash(
        custody,
        (),
        action="return",
        actor_role="company",
        actor_company_id=8,
        beneficiary_company_id=7,
        idempotency_key="autre",
    )
    assert foreign.reason == "company_isolation_refused"
    _prior, driver = resolve_held_cash(
        custody,
        (),
        action="impute",
        actor_role="driver",
        actor_company_id=7,
        beneficiary_company_id=7,
        idempotency_key="chauffeur",
    )
    assert driver.reason == "author_not_authorized"


def test_brouillon_de_facture_reste_non_approuve() -> None:
    document = draft_signature_document(
        version=DOCUMENT_VERSION_REVISED, text=REVISED_TEXT
    )
    assert document.version == "2026-10-26.draft-2"
    assert document.approved is False
    assert document.debt_acknowledgment is False
    presented = present_billing_document(
        debtor_label="Patient fictif",
        issuer_name="Entreprise fictive",
        service_label="Transport",
        amount_chf=DUE,
        document=document,
    )
    assert presented.awareness_only is True
    assert presented.debt_acknowledgment is False
    assert presented.contractual_acceptance_activated is False
    assert presented.version == "2026-10-26.draft-2"
    preparation = prepare_patient_invoice(
        patient_is_payer=True,
        responsibility_locked=True,
        institution_origin=False,
        company_invoice_enabled=False,
        policy_allows_invoice=False,
        already_paid=False,
        invoice_already_issued=False,
        debtor_label="Patient fictif",
        amount_chf=DUE,
        company_id=7,
        document=document,
    )
    assert preparation.reason == "company_invoice_disabled"
    assert preparation.invoice_issued is False
    assert preparation.attestation_separate is True
    refused = attempt_financial_signature(
        preparation, signatory_role="patient", situation="refused"
    )
    assert refused.signature_persisted is False
    assert refused.invoice_issued is False
    assert refused.reason == "refused"
    driver = attempt_financial_signature(
        preparation, signatory_role="driver", situation=None
    )
    assert driver.reason == "driver_cannot_sign"
    assert driver.signature_persisted is False
