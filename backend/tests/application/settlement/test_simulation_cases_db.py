"""Dossier fictif sur PostgreSQL jetable. Aucune écriture de paiement."""

from __future__ import annotations

from decimal import Decimal

import pytest

from application.settlement.simulation_cases import (
    LedgerClosed,
    open_case,
    transition_case,
)
from domain.end_of_trip_settlement.simulation_machine import SimulationRefusal
from models.enums import BookingStatus, PaymentStatus
from models.payment import Payment
from models.settlement_policy import CompanySettlementCapability
from models.settlement_simulation import (
    SettlementSimulationCase,
    SettlementSimulationEvent,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def ledger(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_SIMULATION_LEDGER", "1")


def _prepare(booking, driver):
    booking.status = BookingStatus.IN_PROGRESS
    booking.price_amount = Decimal("45.00")
    booking.driver_id = driver.id
    booking.billed_to_type = "patient"
    booking.is_round_trip = False
    booking.parent_booking_id = None
    return booking


def test_barriere_fermee_n_ecrit_rien(db, sample_booking, simple_driver, monkeypatch):
    monkeypatch.setenv("SETTLEMENT_SIMULATION_LEDGER", "0")
    booking = _prepare(sample_booking, simple_driver)
    db.session.commit()
    with pytest.raises(LedgerClosed):
        open_case(
            booking,
            driver_id=int(simple_driver.id),
            company_id=int(booking.company_id),
        )
    assert (
        db.session.query(SettlementSimulationCase)
        .filter_by(booking_id=booking.id)
        .count()
        == 0
    )


def test_especes_puis_rejeu_et_aucune_ecriture_reelle(
    db, sample_booking, simple_driver, ledger
):
    booking = _prepare(sample_booking, simple_driver)
    db.session.commit()
    payments_before = db.session.query(Payment).count()
    view = open_case(
        booking, driver_id=int(simple_driver.id), company_id=int(booking.company_id)
    )
    assert view["banner"] == "SIMULATION — AUCUN PAIEMENT RÉEL"
    assert view["real_capabilities_enabled"] is False
    assert view["amount_chf"] == "45.00"
    done = transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="declare_cash",
        idempotency_key="cash-45",
        payload={
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
        },
    )
    assert done["state"] == "simulated_confirmed"
    assert done["writes_cash_receipt"] is False
    replay = transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="declare_cash",
        idempotency_key="cash-45",
        payload={
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
        },
    )
    assert replay["replay"] is True
    assert (
        db.session.query(SettlementSimulationEvent)
        .filter_by(case_id=done["id"])
        .count()
        == 1
    )
    assert db.session.query(Payment).count() == payments_before
    assert db.session.get(CompanySettlementCapability, int(booking.company_id)) is None


def test_tap_incertain_puis_relecture(db, sample_booking, simple_driver, ledger):
    booking = _prepare(sample_booking, simple_driver)
    db.session.commit()
    open_case(
        booking, driver_id=int(simple_driver.id), company_id=int(booking.company_id)
    )
    transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="start_tap",
        idempotency_key="tap-a",
        payload={},
    )
    with pytest.raises(SimulationRefusal) as second:
        transition_case(
            booking,
            driver_id=int(simple_driver.id),
            company_id=int(booking.company_id),
            actor_role="driver",
            actor_id=int(simple_driver.id),
            action="start_tap",
            idempotency_key="tap-b",
            payload={},
        )
    assert second.value.code == "tentative_deja_ouverte"
    uncertain = transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="resolve_tap",
        idempotency_key="tap-resolve",
        payload={"simulated_result": "uncertain"},
    )
    assert uncertain["state"] == "uncertain"
    recovered = transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="recover_tap",
        idempotency_key="tap-recover",
        payload={},
    )
    assert recovered["replay"] is True
    assert recovered["state"] == "uncertain"
    assert db.session.query(Payment).filter_by(booking_id=booking.id).count() == 0


def test_aller_retour_deja_paye_et_autre_chauffeur(
    db, sample_booking, simple_driver, ledger
):
    booking = _prepare(sample_booking, simple_driver)
    booking.is_round_trip = True
    db.session.commit()
    with pytest.raises(SimulationRefusal) as round_trip:
        open_case(
            booking,
            driver_id=int(simple_driver.id),
            company_id=int(booking.company_id),
        )
    assert round_trip.value.code == "aller_retour_bloque"
    booking.is_round_trip = False
    db.session.add(
        Payment(
            amount=45.0,
            method="credit_card",
            status=PaymentStatus.COMPLETED,
            booking_id=booking.id,
        )
    )
    db.session.commit()
    with pytest.raises(SimulationRefusal) as paid:
        open_case(
            booking,
            driver_id=int(simple_driver.id),
            company_id=int(booking.company_id),
        )
    assert paid.value.code == "deja_paye"
    assert (
        db.session.query(SettlementSimulationCase)
        .filter_by(booking_id=booking.id)
        .count()
        == 0
    )


def test_facture_patient_sans_signature(db, sample_booking, simple_driver, ledger):
    booking = _prepare(sample_booking, simple_driver)
    db.session.commit()
    open_case(
        booking, driver_id=int(simple_driver.id), company_id=int(booking.company_id)
    )
    view = transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="invoice_event",
        idempotency_key="fac-1",
        payload={"kind": "simulated_acceptance"},
    )
    assert view["signature_recorded"] is False
    assert view["writes_invoice"] is False
    assert view["document_approved"] is False
    assert view["state"] == "administrative_simulated"


def test_chauffeur_ne_rapproche_pas(db, sample_booking, simple_driver, ledger):
    booking = _prepare(sample_booking, simple_driver)
    db.session.commit()
    open_case(
        booking, driver_id=int(simple_driver.id), company_id=int(booking.company_id)
    )
    transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="declare_cash",
        idempotency_key="cash-net",
        payload={
            "tendered_chf": "45.00",
            "change_chf": "0",
            "change_confirmed": False,
        },
    )
    with pytest.raises(SimulationRefusal) as denied:
        transition_case(
            booking,
            driver_id=int(simple_driver.id),
            company_id=int(booking.company_id),
            actor_role="driver",
            actor_id=int(simple_driver.id),
            action="company_cash",
            idempotency_key="rapproche",
            payload={"step": "reconcile"},
        )
    assert denied.value.code == "rapprochement_reserve_entreprise"
    followed = transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="company",
        actor_id=1,
        action="company_cash",
        idempotency_key="reception",
        payload={"step": "receive"},
    )
    assert followed["history"][-1]["action"] == "company_cash_receive"
    assert followed["writes_completed_payment"] is False


def test_montant_change_pendant_la_tentative(db, sample_booking, simple_driver, ledger):
    booking = _prepare(sample_booking, simple_driver)
    db.session.commit()
    open_case(
        booking, driver_id=int(simple_driver.id), company_id=int(booking.company_id)
    )
    transition_case(
        booking,
        driver_id=int(simple_driver.id),
        company_id=int(booking.company_id),
        actor_role="driver",
        actor_id=int(simple_driver.id),
        action="start_tap",
        idempotency_key="tap-montant",
        payload={},
    )
    booking.price_amount = Decimal("60.00")
    db.session.commit()
    with pytest.raises(SimulationRefusal) as stale:
        transition_case(
            booking,
            driver_id=int(simple_driver.id),
            company_id=int(booking.company_id),
            actor_role="driver",
            actor_id=int(simple_driver.id),
            action="resolve_tap",
            idempotency_key="tap-montant-resolve",
            payload={"simulated_result": "confirmed"},
        )
    assert stale.value.code == "decision_obsolete"
    case = (
        db.session.query(SettlementSimulationCase)
        .filter_by(booking_id=booking.id)
        .one()
    )
    assert case.state == "administrative_simulated"
    assert db.session.query(Payment).filter_by(booking_id=booking.id).count() == 0
