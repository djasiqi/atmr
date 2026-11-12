"""Contrat lu par le chauffeur. Les montants viennent du serveur."""

import os
from decimal import Decimal
from types import SimpleNamespace

import pytest
from flask import Flask
from sqlalchemy import text

from application.settlement import cash_ledger
from application.settlement.cash_ledger import (
    describe_cash,
    record_capped_collection,
    record_compensation,
)
from application.settlement.production_gates import refuse_booking
from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db
from models.enums import BookingStatus

pytestmark = pytest.mark.skipif(
    not os.getenv("P2L03_DATABASE_URL"),
    reason="P2L03_DATABASE_URL absente",
)


@pytest.fixture(scope="module")
def app():
    flask = Flask("p2l03")
    flask.config["SQLALCHEMY_DATABASE_URI"] = os.environ["P2L03_DATABASE_URL"]
    flask.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask)

    def _plain(row):
        return {"id": row.id, "state": row.state, "replay": False}

    cash_ledger._view = _plain
    cash_ledger._receipt = lambda _row: None
    from application.settlement import patient_billing_prep

    patient_billing_prep.reconcile_followup_balance = lambda _booking: False
    return flask


def _booking(booking_id, due="45.00"):
    return SimpleNamespace(
        id=booking_id,
        company_id=1,
        driver_id=7,
        price_amount=Decimal(due),
        customer_name="Essai",
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="patient",
        is_round_trip=False,
        parent_booking_id=None,
    )


def _collect(booking, key, amount):
    return record_capped_collection(
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": key,
            "tendered_chf": amount,
            "change_chf": "0.00",
            "change_confirmed": False,
            "disposition": "partial" if Decimal(amount) < Decimal("45.00") else None,
        },
        driver_id=7,
        company_id=1,
        actor_id=7,
        operation_key=key,
    )


def test_reponse_partielle_puis_deuxieme_encaissement(app):
    booking = _booking(60)
    second = _booking(61)
    with app.app_context():
        partial = _collect(booking, "enc-60", "20.00")
        assert partial["amount_due_chf"] == "45.00"
        assert partial["recognized_net_chf"] == "20.00"
        assert partial["remaining_due_chf"] == "25.00"
        assert partial["net_chf"] == "20.00"
        assert partial["imputable"] is True
        assert partial["allocated_invoice_payment_id"] is None
        assert partial["collect_allowed"] is True
        assert partial["actions"]["tap_to_pay"] is False
        assert partial["actions"]["real_receipt"] is False
        assert partial["replay"] is False
        assert partial["client_operation_key"] == "enc-60"
        assert partial["operations"][0]["display"] == "Encaissement 20.00 CHF"
        both = _collect(second, "enc-61-a", "20.00")
        both = _collect(second, "enc-61-b", "15.00")
        assert both["recognized_net_chf"] == "35.00"
        assert both["remaining_due_chf"] == "10.00"
        assert len(both["operations"]) == 2
        db.session.remove()


def test_compensation_solde_nul_et_depassement(app):
    booking = _booking(62)
    full = _booking(63)
    capped = _booking(64)
    with app.app_context():
        origin = _collect(booking, "enc-62", "20.00")
        corrected = record_compensation(
            booking,
            {
                "action": "compensate_cash",
                "idempotency_key": "comp-62",
                "net_chf": "-5.00",
                "compensates_collection_id": origin["id"],
                "reason": "erreur de saisie",
            },
            driver_id=7,
            company_id=1,
            actor_id=9,
            operation_key="comp-62",
        )
        assert corrected["recognized_net_chf"] == "15.00"
        assert corrected["remaining_due_chf"] == "30.00"
        assert corrected["operations"][-1]["nature"] == "correction"
        assert corrected["operations"][-1]["display"] == "Correction -5.00 CHF"
        assert corrected["operations"][0]["net_chf"] == "20.00"
        assert "compensate_cash" in [line["action"] for line in corrected["history"]]
        settled = _collect(full, "enc-63", "45.00")
        assert settled["recognized_net_chf"] == "45.00"
        assert settled["remaining_due_chf"] == "0.00"
        assert settled["collect_allowed"] is False
        _collect(capped, "enc-64", "20.00")
        with pytest.raises(PrepRefusal) as refused:
            _collect(capped, "enc-64-trop", "30.00")
        assert refused.value.code == "depassement_autorise"
        position = describe_cash(capped)
        assert position["recognized_net_chf"] == "20.00"
        assert position["remaining_due_chf"] == "25.00"
        assert len(position["operations"]) == 1
        replay = _collect(capped, "enc-64", "20.00")
        assert replay["replay"] is True
        assert replay["recognized_net_chf"] == "20.00"
        assert len(replay["operations"]) == 1
        payments = db.session.execute(
            text(
                "SELECT COUNT(*) FROM invoice_payments "
                "WHERE cash_collection_id IS NOT NULL"
            )
        ).scalar_one()
        db.session.remove()
    assert payments == 0


def test_autre_entreprise_ne_lit_pas_la_course():
    booking = SimpleNamespace(
        id=60,
        company_id=1,
        driver_id=7,
        status=BookingStatus.IN_PROGRESS,
        price_amount=Decimal("45.00"),
    )
    with pytest.raises(PrepRefusal) as refused:
        refuse_booking(booking, driver_id=7, company_id=2)
    assert refused.value.code == "autre_entreprise"
