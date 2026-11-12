"""Encaissements successifs plafonnés. Base jetable, aucun paiement de facture."""

import os
import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest
from flask import Flask
from sqlalchemy import text

from application.settlement import cash_ledger
from application.settlement.cash_ledger import record_capped_collection
from application.settlement.production_gates import refuse_booking
from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db
from models.enums import BookingStatus

pytestmark = pytest.mark.skipif(
    not os.getenv("P2L01_DATABASE_URL"),
    reason="P2L01_DATABASE_URL absente",
)


@pytest.fixture(scope="module")
def app():
    flask = Flask("p2l01")
    flask.config["SQLALCHEMY_DATABASE_URI"] = os.environ["P2L01_DATABASE_URL"]
    flask.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask)
    cash_ledger._receipt = lambda _row: None

    def _plain(row):
        return {"id": row.id, "state": row.state, "imputable": False}

    cash_ledger._view = _plain
    from application.settlement import patient_billing_prep

    patient_billing_prep.reconcile_followup_balance = lambda _booking: False
    return flask


def _booking(booking_id, due, *, company_id=1):
    return SimpleNamespace(
        id=booking_id,
        company_id=company_id,
        driver_id=7,
        price_amount=Decimal(due),
        customer_name="Essai",
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="patient",
        is_round_trip=False,
        parent_booking_id=None,
    )


def _payload(key, amount):
    return {
        "action": "declare_cash",
        "idempotency_key": key,
        "tendered_chf": amount,
        "change_chf": "0.00",
        "change_confirmed": False,
        "disposition": "partial",
    }


def _record(booking, key, amount):
    return record_capped_collection(
        booking,
        _payload(key, amount),
        driver_id=7,
        company_id=int(booking.company_id),
        actor_id=7,
        operation_key=key,
    )


def _recognized(connection, booking_id):
    return connection.execute(
        text(
            "SELECT COALESCE(SUM(net_chf), 0) FROM cash_collection "
            "WHERE booking_id = :booking_id AND disposition = 'partial'"
        ),
        {"booking_id": booking_id},
    ).scalar_one()


def test_vingt_puis_quinze_puis_plafond_puis_reste(app):
    booking = _booking(20, "45.00")
    with app.app_context():
        first = _record(booking, "enc-20", "20.00")
        assert first["recognized_net_chf"] == "20.00"
        assert first["remaining_due_chf"] == "25.00"
        assert first["allocated_invoice_payment_id"] is None
        second = _record(booking, "enc-15", "15.00")
        assert second["recognized_net_chf"] == "35.00"
        assert second["remaining_due_chf"] == "10.00"
        with pytest.raises(PrepRefusal) as refused:
            _record(booking, "enc-trop", "15.00")
        assert refused.value.code == "depassement_autorise"
        exact = _record(booking, "enc-10", "10.00")
        assert exact["recognized_net_chf"] == "45.00"
        assert exact["remaining_due_chf"] == "0.00"
        with pytest.raises(PrepRefusal) as closed:
            _record(booking, "enc-encore", "1.00")
        assert closed.value.code == "depassement_autorise"
        kept = db.session.execute(
            text("SELECT COUNT(*) FROM cash_collection WHERE booking_id = 20")
        ).scalar_one()
        assert kept == 3
        db.session.remove()


def test_rejeu_et_contenu_different(app):
    booking = _booking(23, "45.00")
    with app.app_context():
        _record(booking, "enc-rejeu", "20.00")
        replay = _record(booking, "enc-rejeu", "20.00")
        assert replay["replay"] is True
        count = db.session.execute(
            text(
                "SELECT COUNT(*) FROM cash_collection "
                "WHERE booking_id = 23 AND client_operation_key = 'enc-rejeu'"
            )
        ).scalar_one()
        assert count == 1
        with pytest.raises(PrepRefusal) as conflict:
            _record(booking, "enc-rejeu", "10.00")
        assert conflict.value.code == "conflit_idempotence"
        db.session.remove()


def test_concurrence_depassement_rejeu_et_compatible(app):
    over = _race(app, 24, ("30.00", "30.00"), ("course-a", "course-b"))
    assert sorted(code for _ok, code in over) == ["30.00", "depassement_autorise"]
    same = _race(app, 25, ("20.00", "20.00"), ("meme-cle", "meme-cle"))
    assert [code for ok, code in same if ok] == ["20.00", "20.00"]
    both = _race(app, 26, ("20.00", "15.00"), ("lot-a", "lot-b"))
    assert all(ok for ok, _code in both)
    assert "35.00" in [code for ok, code in both if ok]
    with app.app_context():
        totals = {
            24: _recognized(db.session, 24),
            25: _recognized(db.session, 25),
            26: _recognized(db.session, 26),
        }
        rows = db.session.execute(
            text(
                "SELECT booking_id, COUNT(*) FROM cash_collection "
                "WHERE booking_id IN (24, 25, 26) GROUP BY booking_id"
            )
        ).all()
        db.session.remove()
    assert totals[24] == Decimal("30.00")
    assert totals[25] == Decimal("20.00")
    assert totals[26] == Decimal("35.00")
    assert dict(rows)[25] == 1
    assert dict(rows)[26] == 2


def test_aller_retour_et_historique(app):
    outbound = _booking(27, "45.00")
    inbound = _booking(28, "30.00")
    with app.app_context():
        _record(outbound, "aller", "20.00")
        retour = _record(inbound, "retour", "10.00")
        assert retour["recognized_net_chf"] == "10.00"
        assert retour["remaining_due_chf"] == "20.00"
        origin = db.session.execute(
            text(
                "SELECT net_chf, client_operation_key FROM cash_collection WHERE id = 1"
            )
        ).one()
        flags = db.session.execute(
            text(
                "SELECT bool_or(production_activation) OR bool_or(legally_approved) "
                "FROM cash_collection"
            )
        ).scalar_one()
        payments = db.session.execute(
            text(
                "SELECT COUNT(*) FROM invoice_payments "
                "WHERE cash_collection_id IS NOT NULL"
            )
        ).scalar_one()
        notes = db.session.execute(
            text("SELECT COUNT(*) FROM invoice_credit_note")
        ).scalar_one()
        handovers = db.session.execute(
            text("SELECT COUNT(*) FROM cash_handover")
        ).scalar_one()
        db.session.remove()
    assert origin.net_chf == Decimal("20.00")
    assert origin.client_operation_key == "1"
    assert flags is False
    assert payments == 0
    assert notes == 0
    assert handovers == 0


def test_refus_metier_sans_ecriture():
    cancelled = SimpleNamespace(
        id=1,
        company_id=1,
        driver_id=7,
        status=BookingStatus.CANCELED,
        price_amount=Decimal("45.00"),
    )
    with pytest.raises(PrepRefusal) as cancelled_error:
        refuse_booking(cancelled, driver_id=7, company_id=1)
    assert cancelled_error.value.code == "transport_annule"
    unassigned = SimpleNamespace(
        id=1,
        company_id=1,
        driver_id=7,
        status=BookingStatus.IN_PROGRESS,
        price_amount=Decimal("45.00"),
    )
    with pytest.raises(PrepRefusal) as driver_error:
        refuse_booking(unassigned, driver_id=8, company_id=1)
    assert driver_error.value.code == "chauffeur_non_affecte"
    clinic = SimpleNamespace(
        id=1,
        company_id=1,
        driver_id=7,
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="clinic",
        price_amount=Decimal("45.00"),
        is_round_trip=False,
        parent_booking_id=None,
    )
    with pytest.raises(PrepRefusal) as payer_error:
        refuse_booking(clinic, driver_id=7, company_id=1)
    assert payer_error.value.code == "institution_facturee"
    foreign = SimpleNamespace(
        id=1,
        company_id=2,
        driver_id=7,
        status=BookingStatus.IN_PROGRESS,
        price_amount=Decimal("45.00"),
    )
    with pytest.raises(PrepRefusal) as company_error:
        refuse_booking(foreign, driver_id=7, company_id=1)
    assert company_error.value.code == "autre_entreprise"


def _race(app, booking_id, amounts, keys):
    barrier = threading.Barrier(2)
    found = []
    guard = threading.Lock()

    def _run(amount, key):
        with app.app_context():
            barrier.wait()
            try:
                view = _record(_booking(booking_id, "45.00"), key, amount)
                item = (True, view["recognized_net_chf"])
            except PrepRefusal as exc:
                item = (False, exc.code)
            finally:
                db.session.remove()
            with guard:
                found.append(item)

    threads = [
        threading.Thread(target=_run, args=(amount, key))
        for amount, key in zip(amounts, keys, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return found
