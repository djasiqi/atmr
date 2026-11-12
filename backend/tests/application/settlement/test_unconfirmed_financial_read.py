"""Le contrat non confirmé n'autorise aucune opération financière."""

from application.settlement.unconfirmed_financial_read import (
    unconfirmed_financial_contract,
)


def test_contrat_patient_sans_politique():
    contract = unconfirmed_financial_contract(billed_to_type="patient", amount="45")
    assert contract["kind"] == "financial_situation_unconfirmed"
    assert contract["verification_required"] is True
    assert contract["amount_chf"] == "45.00"
    assert contract["billed_to_type"] == "patient"
    assert contract["modes"] == []
    assert contract["simulation"] is False
    assert contract["payment_executable"] is False
    assert contract["writes_invoice"] is False
    assert contract["writes_cash_receipt"] is False
    assert contract["writes_financial_signature"] is False
    assert contract["driver_may_edit_amount"] is False
    assert contract["driver_may_edit_payer"] is False


def test_montant_absent_reste_absent():
    contract = unconfirmed_financial_contract(billed_to_type=None, amount=None)
    assert contract["amount_chf"] is None
    assert contract["billed_to_type"] is None
    assert contract["modes"] == []
