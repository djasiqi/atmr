"""Machine fictive : aucun prestataire, aucun second débit."""

from decimal import Decimal

import pytest

from domain.end_of_trip_settlement.simulation_barrier import ledger_open
from domain.end_of_trip_settlement.simulation_machine import (
    CONFIRMED,
    IN_PROGRESS,
    REFUSED,
    UNCERTAIN,
    SimulationRefusal,
    declare_cash,
    recover_tap,
    resolve_tap,
    shortfall_cash,
    start_tap,
)

AMOUNT = Decimal("45.00")


def test_barrier_fermee_par_defaut():
    assert ledger_open(database="atmr", switch="1", production=False) is False
    assert ledger_open(database="atmr_test", switch="1", production=False) is False
    assert ledger_open(database="atmr_etape24", switch="0", production=False) is False
    assert ledger_open(database="atmr_etape24", switch="1", production=True) is False
    assert ledger_open(database="atmr_etape24", switch="1", production=False) is True


def test_tap_accepte_puis_seconde_tentative_refusee():
    started = start_tap(
        state="preparing",
        mode=None,
        attempt_id=None,
        amount_chf=AMOUNT,
        merchant_company_id=3,
        new_attempt_id="tap-1",
    )
    assert started.state == IN_PROGRESS
    assert started.attempt_id == "tap-1"
    confirmed = resolve_tap(
        state=started.state,
        attempt_id=started.attempt_id,
        amount_chf=AMOUNT,
        merchant_company_id=3,
        simulated_result="confirmed",
    )
    assert confirmed.state == CONFIRMED
    with pytest.raises(SimulationRefusal) as refused:
        start_tap(
            state=confirmed.state,
            mode="tap_to_pay",
            attempt_id=started.attempt_id,
            amount_chf=AMOUNT,
            merchant_company_id=3,
            new_attempt_id="tap-2",
        )
    assert refused.value.code == "deja_simule"


def test_reponse_perdue_ne_relance_pas():
    started = start_tap(
        state="preparing",
        mode=None,
        attempt_id=None,
        amount_chf=AMOUNT,
        merchant_company_id=3,
        new_attempt_id="tap-perdu",
    )
    uncertain = resolve_tap(
        state=started.state,
        attempt_id=started.attempt_id,
        amount_chf=AMOUNT,
        merchant_company_id=3,
        simulated_result="uncertain",
    )
    assert uncertain.state == UNCERTAIN
    recovered = recover_tap(state=uncertain.state, attempt_id=uncertain.attempt_id)
    assert recovered.replay is True
    assert recovered.state == UNCERTAIN
    with pytest.raises(SimulationRefusal) as second:
        start_tap(
            state=uncertain.state,
            mode="tap_to_pay",
            attempt_id=uncertain.attempt_id,
            amount_chf=AMOUNT,
            merchant_company_id=3,
            new_attempt_id="tap-autre",
        )
    assert second.value.code == "tentative_deja_ouverte"


def test_deux_appuis_pendant_l_operation():
    started = start_tap(
        state="preparing",
        mode=None,
        attempt_id=None,
        amount_chf=AMOUNT,
        merchant_company_id=8,
        new_attempt_id="rapide",
    )
    with pytest.raises(SimulationRefusal):
        start_tap(
            state=started.state,
            mode="tap_to_pay",
            attempt_id=started.attempt_id,
            amount_chf=AMOUNT,
            merchant_company_id=8,
            new_attempt_id="rapide-2",
        )


def test_refus_simule():
    started = start_tap(
        state="preparing",
        mode=None,
        attempt_id=None,
        amount_chf=AMOUNT,
        merchant_company_id=3,
        new_attempt_id="refus",
    )
    refused = resolve_tap(
        state=started.state,
        attempt_id=started.attempt_id,
        amount_chf=AMOUNT,
        merchant_company_id=3,
        simulated_result="refused",
    )
    assert refused.state == REFUSED


def test_especes_45_avec_monnaie_confirmee():
    done = declare_cash(
        state="preparing",
        mode=None,
        expected_chf=AMOUNT,
        tendered_chf="50.00",
        change_chf="5.00",
        change_confirmed=True,
    )
    assert done.state == CONFIRMED
    assert done.payload["net_chf"] == "45.00"
    assert done.payload["real_cash_receipt"] is False


def test_monnaie_non_confirmee_n_ecrit_rien():
    with pytest.raises(SimulationRefusal) as error:
        declare_cash(
            state="preparing",
            mode=None,
            expected_chf=AMOUNT,
            tendered_chf="50.00",
            change_chf="5.00",
            change_confirmed=False,
        )
    assert error.value.code == "monnaie_non_confirmee"


def test_somme_conservee_ne_cree_pas_de_creance_integrale():
    held = shortfall_cash(
        state="preparing",
        mode=None,
        expected_chf=AMOUNT,
        tendered_chf="30.00",
        disposition="held",
    )
    assert held.payload["held_is_not_void"] is True
    assert held.payload["full_receivable_created"] is False
    assert held.payload["noted_gap_chf"] == "15.00"
    assert held.payload["decision_h"] == "draft"


def test_deux_modes_en_meme_temps_refuses():
    with pytest.raises(SimulationRefusal) as error:
        declare_cash(
            state="operation_in_progress",
            mode="tap_to_pay",
            expected_chf=AMOUNT,
            tendered_chf="45.00",
            change_chf="0",
            change_confirmed=True,
        )
    assert error.value.code == "especes_deja_enregistrees"
