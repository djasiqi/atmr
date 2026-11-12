"""Libération des spécimens. Aucune facture n'est supprimée."""

import os
import threading
from decimal import Decimal

import pytest
from flask import Flask
from sqlalchemy import text

from application.settlement.patient_billing_prep import release_specimen_locks
from ext import db

pytestmark = pytest.mark.skipif(
    not os.getenv("P3L08_DATABASE_URL"),
    reason="P3L08_DATABASE_URL absente",
)


@pytest.fixture(scope="module")
def app():
    flask = Flask("p3l08")
    flask.config["SQLALCHEMY_DATABASE_URI"] = os.environ["P3L08_DATABASE_URL"]
    flask.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask)
    return flask


def _counts(connection):
    return {
        name: connection.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
        for name in (
            "invoices",
            "invoice_lines",
            "patient_billing_dossier",
            "invoice_payments",
            "invoice_credit_note",
        )
    }


def test_liberation_cible_et_second_passage(app):
    with app.app_context():
        before = _counts(db.session)
        prices = dict(
            db.session.execute(
                text("SELECT id, price_amount FROM booking ORDER BY id")
            ).all()
        )
        first = release_specimen_locks(1)
        released_ids = {item["booking_id"] for item in first["released"]}
        assert released_ids == {10}
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 10")
            ).scalar_one()
            is None
        )
        candidates = {
            row[0]
            for row in db.session.execute(
                text(
                    "SELECT id FROM booking "
                    "WHERE company_id = 1 AND invoice_line_id IS NULL"
                )
            ).all()
        }
        assert 10 in candidates
        assert 11 not in candidates
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 11")
            ).scalar_one()
            == 110
        )
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 12")
            ).scalar_one()
            == 120
        )
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 13")
            ).scalar_one()
            == 130
        )
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 14")
            ).scalar_one()
            == 140
        )
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 15")
            ).scalar_one()
            == 160
        )
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 18")
            ).scalar_one()
            == 150
        )
        reasons = {item["reason"] for item in first["ignored"]}
        assert "ambigu" in reasons
        assert "facture_emise" in reasons
        meta = db.session.execute(
            text("SELECT meta FROM invoices WHERE id = 1")
        ).scalar_one()
        assert meta["lock_released"] is True
        assert meta["blocking"] is False
        assert meta["specimen"] is True
        after = _counts(db.session)
        assert after == before
        assert (
            dict(
                db.session.execute(
                    text("SELECT id, price_amount FROM booking ORDER BY id")
                ).all()
            )
            == prices
        )
        second = release_specimen_locks(1)
        assert second["released"] == []
        assert second["wrote"] is False
        assert _counts(db.session) == before
        assert (
            db.session.execute(
                text("SELECT invoice_line_id FROM booking WHERE id = 10")
            ).scalar_one()
            is None
        )
        db.session.remove()


def test_deux_liberations_concurrentes(app):
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO invoices (id, company_id, invoice_number, status, meta, total_amount)
                VALUES (
                  17, 1, 'SPECIMEN-1-17', 'draft',
                  '{"specimen": true, "payable": false}'::jsonb, 45.00
                )
                """
            )
        )
        db.session.execute(
            text(
                "INSERT INTO invoice_lines (id, invoice_id, reservation_id) "
                "VALUES (170, 17, 17)"
            )
        )
        db.session.execute(
            text(
                "INSERT INTO booking (id, company_id, invoice_line_id, price_amount) "
                "VALUES (17, 1, 170, 45.00)"
            )
        )
        db.session.commit()
        invoices_before = db.session.execute(
            text("SELECT COUNT(*) FROM invoices")
        ).scalar_one()
        db.session.remove()
    barrier = threading.Barrier(2)
    found = []
    guard = threading.Lock()

    def _run():
        with app.app_context():
            barrier.wait()
            result = release_specimen_locks(1)
            with guard:
                found.append(result)
            db.session.remove()

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with app.app_context():
        link = db.session.execute(
            text("SELECT invoice_line_id, price_amount FROM booking WHERE id = 17")
        ).one()
        real = db.session.execute(
            text("SELECT invoice_line_id FROM booking WHERE id = 11")
        ).scalar_one()
        invoices_after = db.session.execute(
            text("SELECT COUNT(*) FROM invoices")
        ).scalar_one()
        line = db.session.execute(
            text("SELECT COUNT(*) FROM invoice_lines WHERE id = 170")
        ).scalar_one()
        db.session.remove()
    assert link.invoice_line_id is None
    assert link.price_amount == Decimal("45.00")
    assert real == 110
    assert invoices_after == invoices_before
    assert line == 1
    released = [item["booking_id"] for result in found for item in result["released"]]
    assert released.count(17) == 1
