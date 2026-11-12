"""Règles d'espèces, sans base. La décision H reste un brouillon."""

from decimal import Decimal

import pytest

from domain.end_of_trip_settlement.cash_settlement import evaluate_cash, money
from domain.end_of_trip_settlement.production_prep_barrier import prep_open


def test_quarante_cinq_recus():
    decision = evaluate_cash(
        due=Decimal("45.00"),
        tendered=Decimal("45.00"),
        change=Decimal("0.00"),
        change_confirmed=False,
        disposition=None,
    )
    assert decision.outcome == "received"
    assert decision.net_chf == Decimal("45.00")
    assert decision.receipt_allowed is True
    assert decision.imputed is False


def test_cinquante_avec_cinq_rendus():
    decision = evaluate_cash(
        due=Decimal("45.00"),
        tendered=Decimal("50.00"),
        change=Decimal("5.00"),
        change_confirmed=True,
        disposition=None,
    )
    assert decision.net_chf == Decimal("45.00")
    assert decision.change_chf == Decimal("5.00")


def test_monnaie_non_confirmee():
    with pytest.raises(ValueError, match="monnaie_non_confirmee"):
        evaluate_cash(
            due=Decimal("45.00"),
            tendered=Decimal("50.00"),
            change=Decimal("5.00"),
            change_confirmed=False,
            disposition=None,
        )


def test_trente_restitues():
    decision = evaluate_cash(
        due=Decimal("45.00"),
        tendered=Decimal("30.00"),
        change=Decimal("0.00"),
        change_confirmed=False,
        disposition="returned",
    )
    assert decision.outcome == "returned"
    assert decision.net_chf == Decimal("0.00")
    assert decision.receipt_allowed is False


def test_trente_conserves_sans_imputation():
    decision = evaluate_cash(
        due=Decimal("45.00"),
        tendered=Decimal("30.00"),
        change=Decimal("0.00"),
        change_confirmed=False,
        disposition="held",
    )
    assert decision.outcome == "administrative"
    assert decision.imputed is False
    assert decision.full_receivable_created is False
    assert decision.held_is_not_void is True
    assert decision.decision_h == "draft"
    assert decision.receipt_allowed is False


def test_partiel_garde_le_solde_et_autorise_un_specimen():
    decision = evaluate_cash(
        due=Decimal("45.00"),
        tendered=Decimal("20.00"),
        change=Decimal("0.00"),
        change_confirmed=False,
        disposition="partial",
    )
    assert decision.outcome == "administrative"
    assert decision.net_chf == Decimal("20.00")
    assert decision.receipt_allowed is True
    assert decision.imputed is False
    assert decision.full_receivable_created is False


def test_imputation_refusee():
    with pytest.raises(ValueError, match="imputation_non_approuvee"):
        evaluate_cash(
            due=Decimal("45.00"),
            tendered=Decimal("30.00"),
            change=Decimal("0.00"),
            change_confirmed=False,
            disposition="imputed",
        )


def test_montant_a_trois_decimales_refuse():
    with pytest.raises(ValueError, match="montant_illisible"):
        money("45.001")


def test_barriere_fermee_hors_base_jetable():
    assert prep_open(database="atmr", switch="1", production=False) is False
    assert prep_open(database="atmr_test", switch="1", production=False) is False
    assert (
        prep_open(database="lirie_e2e_settlement_etape28", switch="0", production=False)
        is False
    )
    assert (
        prep_open(database="lirie_e2e_settlement_etape28", switch="1", production=True)
        is False
    )
    assert (
        prep_open(database="lirie_e2e_settlement_etape28", switch="1", production=False)
        is True
    )
