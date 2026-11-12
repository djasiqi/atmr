"""Imputation atomique des acomptes à l'émission. Base jetable."""

import os
import threading
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask
from sqlalchemy import Integer, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import IntegrityError

from application.invoices.cash_allocation import (
    AllocationRefusal,
    find_patient_cash_replay,
    impute_confirmed_cash,
    patient_cash_applies,
)
from application.settlement import cash_ledger
from application.settlement.cash_ledger import record_capped_collection
from ext import db
from models.enums import BookingStatus

pytestmark = pytest.mark.skipif(
    not os.getenv("P3L02_DATABASE_URL"),
    reason="P3L02_DATABASE_URL absente",
)


@pytest.fixture(scope="module")
def app():
    flask = Flask("p3l02")
    flask.config["SQLALCHEMY_DATABASE_URI"] = os.environ["P3L02_DATABASE_URL"]
    flask.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask)
    cash_ledger._receipt = lambda _row: None
    cash_ledger._view = lambda row: {"id": row.id, "state": row.state, "replay": False}
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
    disposition = "partial" if Decimal(amount) < Decimal(str(booking.price_amount)) else None
    return record_capped_collection(
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": key,
            "tendered_chf": amount,
            "change_chf": "0.00",
            "change_confirmed": False,
            "disposition": disposition,
        },
        driver_id=7,
        company_id=1,
        actor_id=7,
        operation_key=key,
    )


def _invoice(invoice_id, total, number):
    db.session.execute(
        text(
            """
            INSERT INTO invoices (
                id, company_id, invoice_number, total_amount,
                amount_paid, balance_due, status
            ) VALUES (
                :id, 1, :number, :total, 0, :total, 'draft'
            )
            """
        ),
        {"id": invoice_id, "number": number, "total": total},
    )


def _line(line_id, invoice_id, reservation_id, amount, linked_ids):
    db.session.execute(
        text(
            """
            INSERT INTO invoice_lines (
                id, invoice_id, reservation_id, unit_price, line_total
            ) VALUES (
                :id, :invoice_id, :reservation_id, :amount, :amount
            )
            """
        ),
        {
            "id": line_id,
            "invoice_id": invoice_id,
            "reservation_id": reservation_id,
            "amount": amount,
        },
    )
    db.session.execute(
        text(
            """
            UPDATE booking
            SET invoice_line_id = :line_id
            WHERE id = ANY(:booking_ids) AND invoice_line_id IS NULL
            """
        ).bindparams(bindparam("booking_ids", type_=ARRAY(Integer))),
        {"line_id": line_id, "booking_ids": linked_ids},
    )


def _issue(invoice_id, line_id, dues, *, merged=False):
    replay = find_patient_cash_replay(1, list(dues))
    if replay is not None:
        db.session.commit()
        return replay
    total = sum(dues.values(), Decimal("0.00"))
    _invoice(invoice_id, total, f"F-TEST-{invoice_id}")
    if merged:
        first = next(iter(dues))
        _line(line_id, invoice_id, first, total, list(dues))
    else:
        for offset, booking_id in enumerate(sorted(dues)):
            _line(
                line_id + offset,
                invoice_id,
                booking_id,
                dues[booking_id],
                [booking_id],
            )
    impute_confirmed_cash(1, invoice_id, total, dues)
    db.session.commit()
    return invoice_id


def _money(invoice_id):
    return db.session.execute(
        text(
            """
            SELECT total_amount, amount_paid, balance_due
            FROM invoices WHERE id = :invoice_id
            """
        ),
        {"invoice_id": invoice_id},
    ).one()


def _payments(invoice_id):
    return db.session.execute(
        text(
            """
            SELECT amount, cash_collection_id
            FROM invoice_payments
            WHERE invoice_id = :invoice_id
            ORDER BY cash_collection_id
            """
        ),
        {"invoice_id": invoice_id},
    ).all()


def test_frontiere_avant_le_commit_et_hors_clinique():
    source = Path(
        "application/invoices/generate_invoice.py"
    ).read_text(encoding="utf-8")
    impute_at = source.index("impute_confirmed_cash(")
    commit_at = source.index("db.session.commit()", impute_at)
    pdf_at = source.index("generate_invoice_pdf", commit_at)
    assert impute_at < commit_at < pdf_at
    clinic = Path(
        "application/invoices/generate_clinic_monthly_invoice.py"
    ).read_text(encoding="utf-8")
    partner = Path(
        "application/invoices/add_partner_invoice_payment.py"
    ).read_text(encoding="utf-8")
    pdf = Path(
        "application/invoices/force_regenerate_invoice_pdf.py"
    ).read_text(encoding="utf-8")
    assert "impute_confirmed_cash" not in clinic
    assert "impute_confirmed_cash" not in partner
    assert "impute_confirmed_cash" not in pdf
    assert patient_cash_applies(
        clinic_company_id=None, bill_to_client_id=None, institution_patient_id=None
    )
    assert not patient_cash_applies(
        clinic_company_id=4, bill_to_client_id=None, institution_patient_id=None
    )
    assert not patient_cash_applies(
        clinic_company_id=None, bill_to_client_id=8, institution_patient_id=None
    )


def test_acompte_vingt_solde_vingt_cinq(app):
    with app.app_context():
        _collect(_booking(71), "p3-71", "20.00")
        invoice_id = _issue(200, 300, {71: Decimal("45.00")})
        row = _money(invoice_id)
        payments = _payments(invoice_id)
        line = db.session.execute(
            text("SELECT line_total FROM invoice_lines WHERE id = 300")
        ).scalar_one()
        price = db.session.execute(
            text("SELECT price_amount, invoice_line_id FROM booking WHERE id = 71")
        ).one()
        db.session.remove()
    assert row.total_amount == Decimal("45.00")
    assert row.amount_paid == Decimal("20.00")
    assert row.balance_due == Decimal("25.00")
    assert len(payments) == 1
    assert payments[0].amount == Decimal("20.00")
    assert line == Decimal("45.00")
    assert price.price_amount == Decimal("45.00")
    assert price.invoice_line_id == 300


def test_sans_acompte_et_prepaye(app):
    with app.app_context():
        empty = _issue(201, 301, {70: Decimal("45.00")})
        _collect(_booking(72), "p3-72a", "20.00")
        _collect(_booking(72), "p3-72b", "25.00")
        full = _issue(202, 302, {72: Decimal("45.00")})
        empty_row = _money(empty)
        full_row = _money(full)
        full_payments = _payments(full)
        db.session.remove()
    assert empty_row.amount_paid == Decimal("0.00")
    assert empty_row.balance_due == Decimal("45.00")
    assert full_row.amount_paid == Decimal("45.00")
    assert full_row.balance_due == Decimal("0.00")
    assert len(full_payments) == 2
    assert sum(item.amount for item in full_payments) == Decimal("45.00")


def test_deux_encaissements_et_correction(app):
    with app.app_context():
        _collect(_booking(73), "p3-73a", "20.00")
        _collect(_booking(73), "p3-73b", "15.00")
        partial = _issue(203, 303, {73: Decimal("45.00")})
        origin = _collect(_booking(74), "p3-74", "20.00")
        from application.settlement.cash_ledger import record_compensation

        record_compensation(
            _booking(74),
            {
                "action": "compensate_cash",
                "idempotency_key": "p3-74c",
                "net_chf": "-5.00",
                "compensates_collection_id": origin["id"],
                "reason": "billet compte deux fois",
            },
            driver_id=7,
            company_id=1,
            actor_id=9,
            operation_key="p3-74c",
        )
        corrected = _issue(204, 304, {74: Decimal("45.00")})
        partial_row = _money(partial)
        corrected_row = _money(corrected)
        corrected_payments = _payments(corrected)
        partial_count = len(_payments(partial))
        db.session.remove()
    assert partial_row.amount_paid == Decimal("35.00")
    assert partial_row.balance_due == Decimal("10.00")
    assert partial_count == 2
    assert corrected_row.amount_paid == Decimal("15.00")
    assert corrected_row.balance_due == Decimal("30.00")
    assert len(corrected_payments) == 1
    assert corrected_payments[0].amount == Decimal("15.00")


def test_incertain_et_conteste_ignores(app):
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO cash_collection (
                    booking_id, company_id, driver_id, debtor_label, amount_due_chf,
                    net_chf, decision_version, kind, client_operation_key, state, disposition
                ) VALUES
                (75, 1, 7, 'Essai', 45.00, 20.00, 'v1', 'collection', 'p3-75',
                 'uncertain', 'partial'),
                (76, 1, 7, 'Essai', 45.00, 20.00, 'v1', 'collection', 'p3-76',
                 'administrative', 'contested')
                """
            )
        )
        db.session.commit()
        left = _issue(205, 305, {75: Decimal("45.00")})
        right = _issue(206, 306, {76: Decimal("45.00")})
        assert _money(left).balance_due == Decimal("45.00")
        assert _money(right).balance_due == Decimal("45.00")
        assert _payments(left) == []
        assert _payments(right) == []
        db.session.remove()


def test_aller_retour_et_deux_patients(app):
    with app.app_context():
        _collect(_booking(77), "p3-77", "20.00")
        invoice_id = _issue(
            207,
            307,
            {77: Decimal("45.00"), 78: Decimal("45.00")},
            merged=True,
        )
        row = _money(invoice_id)
        line = db.session.execute(
            text("SELECT line_total FROM invoice_lines WHERE id = 307")
        ).scalar_one()
        links = db.session.execute(
            text(
                """
                SELECT id, invoice_line_id FROM booking
                WHERE id IN (77, 78) ORDER BY id
                """
            )
        ).all()
        _collect(_booking(79), "p3-79", "20.00")
        _collect(_booking(80), "p3-80", "10.00")
        first = _issue(208, 308, {79: Decimal("45.00")})
        second = _issue(209, 309, {80: Decimal("45.00")})
        first_pay = _payments(first)
        second_pay = _payments(second)
        db.session.remove()
    assert row.total_amount == Decimal("90.00")
    assert row.amount_paid == Decimal("20.00")
    assert row.balance_due == Decimal("70.00")
    assert line == Decimal("90.00")
    assert [item.invoice_line_id for item in links] == [307, 307]
    assert first_pay[0].amount == Decimal("20.00")
    assert second_pay[0].amount == Decimal("10.00")
    assert first_pay[0].cash_collection_id != second_pay[0].cash_collection_id


def test_rejeu_deja_impute_et_autre_entreprise(app):
    with app.app_context():
        replay = _issue(200, 300, {71: Decimal("45.00")})
        before = len(_payments(replay))
        again = _issue(299, 399, {71: Decimal("45.00")})
        after = len(_payments(replay))
        db.session.execute(
            text(
                """
                INSERT INTO cash_collection (
                    booking_id, company_id, driver_id, debtor_label, amount_due_chf,
                    net_chf, decision_version, kind, client_operation_key, state, disposition
                ) VALUES (
                    81, 2, 7, 'Essai', 45.00, 20.00, 'v1', 'collection', 'p3-81',
                    'administrative', 'partial'
                )
                """
            )
        )
        db.session.commit()
        _invoice(210, Decimal("45.00"), "F-TEST-210")
        with pytest.raises(AllocationRefusal) as foreign:
            impute_confirmed_cash(1, 210, Decimal("45.00"), {81: Decimal("45.00")})
        db.session.rollback()
        kept = db.session.execute(
            text("SELECT COUNT(*) FROM invoices WHERE id = 210")
        ).scalar_one()
        sequence = db.session.execute(
            text("SELECT sequence FROM invoice_sequences WHERE id = 1")
        ).scalar_one()
        historical = db.session.execute(
            text(
                """
                SELECT cash_collection_id FROM invoice_payments
                WHERE reference = 'historique'
                """
            )
        ).scalar_one()
        db.session.remove()
    assert again == replay
    assert after == before == 1
    assert foreign.value.code == "autre_entreprise"
    assert kept == 0
    assert sequence == 4
    assert historical is None


def test_echec_avant_commit_et_paiement(app, monkeypatch):
    with app.app_context():
        before_invoices = db.session.execute(
            text("SELECT COUNT(*) FROM invoices")
        ).scalar_one()
        before_payments = db.session.execute(
            text("SELECT COUNT(*) FROM invoice_payments")
        ).scalar_one()
        _invoice(211, Decimal("45.00"), "F-TEST-211")
        db.session.rollback()
        assert (
            db.session.execute(text("SELECT COUNT(*) FROM invoices")).scalar_one()
            == before_invoices
        )

        def _broken(*_args, **_kwargs):
            raise IntegrityError("insert", {}, Exception("cash_collection"))

        monkeypatch.setattr(
            "application.invoices.cash_allocation._shares",
            lambda *_args, **_kwargs: [(1, Decimal("20.00"))],
        )
        _collect(_booking(84), "p3-84", "20.00")
        _invoice(212, Decimal("45.00"), "F-TEST-212")
        _line(312, 212, 84, Decimal("45.00"), [84])
        monkeypatch.setattr(
            "application.invoices.cash_allocation.db.session.execute",
            _broken,
        )
        with pytest.raises(IntegrityError):
            impute_confirmed_cash(1, 212, Decimal("45.00"), {84: Decimal("45.00")})
        db.session.rollback()
        monkeypatch.undo()
        assert (
            db.session.execute(text("SELECT COUNT(*) FROM invoices WHERE id = 212")).scalar_one()
            == 0
        )
        assert (
            db.session.execute(text("SELECT COUNT(*) FROM invoice_payments")).scalar_one()
            == before_payments
        )
        db.session.remove()


def test_echec_pdf_ne_reimpute_pas(app):
    with app.app_context():
        invoice_id = _issue(200, 300, {71: Decimal("45.00")})
        paid_before = _money(invoice_id).amount_paid
        count_before = len(_payments(invoice_id))
        db.session.execute(
            text(
                """
                UPDATE invoices SET status = 'draft'
                WHERE id = :invoice_id
                """
            ),
            {"invoice_id": invoice_id},
        )
        db.session.commit()
        replay = find_patient_cash_replay(1, [71])
        db.session.commit()
        db.session.remove()
    assert replay == invoice_id
    assert paid_before == Decimal("20.00")
    assert count_before == 1


def test_emissions_simultanees(app):
    _collect_ready = threading.Event()
    with app.app_context():
        _collect(_booking(86), "p3-86", "20.00")
        db.session.remove()
    found = []

    def _run():
        with app.app_context():
            _collect_ready.wait(5)
            try:
                found.append(_issue(220, 320, {86: Decimal("45.00")}))
            except Exception as exc:  # pragma: no cover - diagnostic
                found.append(exc)
            finally:
                db.session.remove()

    first = threading.Thread(target=_run)
    second = threading.Thread(target=_run)
    first.start()
    second.start()
    _collect_ready.set()
    first.join(10)
    second.join(10)
    with app.app_context():
        invoices = db.session.execute(
            text(
                """
                SELECT COUNT(*) FROM booking b
                JOIN invoice_lines line ON line.id = b.invoice_line_id
                WHERE b.id = 86
                """
            )
        ).scalar_one()
        payments = db.session.execute(
            text(
                """
                SELECT COUNT(*) FROM invoice_payments payment
                JOIN cash_collection collection
                  ON collection.id = payment.cash_collection_id
                WHERE collection.booking_id = 86
                """
            )
        ).scalar_one()
        db.session.remove()
    assert invoices == 1
    assert payments == 1
    assert len(found) == 2
    assert found[0] == found[1]


def test_encaissement_pendant_emission(app):
    locked = threading.Event()
    started = threading.Event()
    outcome = {}

    def _emit():
        with app.app_context():
            db.session.execute(
                text("SELECT id FROM booking WHERE id = 90 FOR UPDATE")
            )
            locked.set()
            assert started.wait(5)
            time.sleep(0.3)
            _invoice(230, Decimal("45.00"), "F-TEST-230")
            _line(330, 230, 90, Decimal("45.00"), [90])
            impute_confirmed_cash(1, 230, Decimal("45.00"), {90: Decimal("45.00")})
            db.session.commit()
            outcome["balance"] = _money(230).balance_due
            db.session.remove()

    def _cash():
        assert locked.wait(5)
        started.set()
        with app.app_context():
            _collect(_booking(90), "p3-90", "20.00")
            db.session.remove()

    invoice_thread = threading.Thread(target=_emit)
    cash_thread = threading.Thread(target=_cash)
    invoice_thread.start()
    cash_thread.start()
    invoice_thread.join(10)
    cash_thread.join(10)
    with app.app_context():
        linked = db.session.execute(
            text(
                """
                SELECT COUNT(*) FROM invoice_payments payment
                JOIN cash_collection collection
                  ON collection.id = payment.cash_collection_id
                WHERE collection.booking_id = 90
                """
            )
        ).scalar_one()
        kept = db.session.execute(
            text("SELECT COUNT(*) FROM cash_collection WHERE booking_id = 90")
        ).scalar_one()
        price = db.session.execute(
            text("SELECT price_amount FROM booking WHERE id = 90")
        ).scalar_one()
        db.session.remove()
    assert outcome["balance"] == Decimal("45.00")
    assert linked == 0
    assert kept == 1
    assert price == Decimal("45.00")
