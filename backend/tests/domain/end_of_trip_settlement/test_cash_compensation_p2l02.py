"""Corrections d'encaissement. L'origine reste en place. Base jetable."""

import os
import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest
from flask import Flask
from sqlalchemy import text

from application.settlement import cash_ledger
from application.settlement.cash_ledger import (
    record_capped_collection,
    record_compensation,
)
from application.settlement.production_gates import refuse_booking
from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db
from models.enums import BookingStatus

pytestmark = pytest.mark.skipif(
    not os.getenv("P2L02_DATABASE_URL"),
    reason="P2L02_DATABASE_URL absente",
)


@pytest.fixture(scope="module")
def app():
    flask = Flask("p2l02")
    flask.config["SQLALCHEMY_DATABASE_URI"] = os.environ["P2L02_DATABASE_URL"]
    flask.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask)

    def _plain(row):
        return {"id": row.id, "state": row.state, "imputable": False, "replay": False}

    original_view = cash_ledger._view
    original_receipt = cash_ledger._receipt
    cash_ledger._view = _plain
    cash_ledger._receipt = lambda _row: None
    from application.settlement import patient_billing_prep

    original_followup = patient_billing_prep.reconcile_followup_balance
    patient_billing_prep.reconcile_followup_balance = lambda _booking: False
    yield flask
    cash_ledger._view = original_view
    cash_ledger._receipt = original_receipt
    patient_billing_prep.reconcile_followup_balance = original_followup


@pytest.fixture(scope="module", autouse=True)
def _reservations_isolees(app):
    """Réservations propres au fichier. La base jetable part vide.

    Les identifiants 40 à 48 et la facture 50 sont ceux déjà cités par les
    assertions. Ils sont créés ici pour ne plus dépendre d'un jeu préexistant.
    """
    with app.app_context():
        db.session.execute(
            text(
                """
                INSERT INTO "user" (id, public_id, password, force_password_change)
                VALUES
                    (1, 'p2l02-owner', 'x', false),
                    (2, 'p2l02-client', 'x', false),
                    (3, 'p2l02-driver', 'x', false)
                """
            )
        )
        db.session.execute(
            text(
                """
                INSERT INTO company (id, name, user_id)
                VALUES (1, 'P2L02 Isolation', 1)
                """
            )
        )
        db.session.execute(
            text(
                """
                INSERT INTO client (id, user_id, company_id)
                VALUES (1, 2, 1)
                """
            )
        )
        db.session.execute(
            text(
                """
                INSERT INTO driver (id, user_id, company_id)
                VALUES (7, 3, 1)
                """
            )
        )
        db.session.execute(
            text(
                """
                INSERT INTO booking (
                    id, company_id, client_id, user_id, driver_id,
                    customer_name, pickup_location, dropoff_location, amount, status
                )
                SELECT n, 1, 1, 2, 7, 'Essai', 'Gare', 'Hopital', 45, 'IN_PROGRESS'
                FROM generate_series(40, 48) AS n
                """
            )
        )
        db.session.execute(
            text(
                """
                INSERT INTO invoices (
                    id, company_id, client_id, period_month, period_year,
                    invoice_number, currency, subtotal_amount, late_fee_amount,
                    reminder_fee_amount, total_amount, amount_paid, balance_due,
                    issued_at, due_date, status, reminder_level, vat_total_amount
                ) VALUES (
                    50, 1, 1, 11, 2026, 'P2L02-50', 'CHF',
                    45.00, 0, 0, 45.00, 0, 45.00,
                    TIMESTAMPTZ '2026-11-01 00:00:00+00',
                    TIMESTAMPTZ '2026-12-31 00:00:00+00',
                    'SENT', 0, 0
                )
                """
            )
        )
        for sequence, table in (
            ("user_id_seq", '"user"'),
            ("company_id_seq", "company"),
            ("client_id_seq", "client"),
            ("driver_id_seq", "driver"),
            ("booking_id_seq", "booking"),
            ("invoices_id_seq", "invoices"),
        ):
            db.session.execute(
                text(
                    f"SELECT setval('{sequence}', GREATEST((SELECT MAX(id) FROM {table}), 1))"
                )
            )
        db.session.commit()
        db.session.remove()


def _booking(booking_id, due="45.00", *, company_id=1):
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


def _collect(booking, key, amount):
    return record_capped_collection(
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": key,
            "tendered_chf": amount,
            "change_chf": "0.00",
            "change_confirmed": False,
            "disposition": "partial",
        },
        driver_id=7,
        company_id=int(booking.company_id),
        actor_id=7,
        operation_key=key,
    )


def _correct(booking, key, origin_id, net, reason="erreur de saisie"):
    return record_compensation(
        booking,
        {
            "action": "compensate_cash",
            "idempotency_key": key,
            "net_chf": net,
            "compensates_collection_id": origin_id,
            "reason": reason,
        },
        driver_id=7,
        company_id=int(booking.company_id),
        actor_id=9,
        operation_key=key,
    )


def test_correction_partielle_puis_encaissement(app):
    booking = _booking(40)
    with app.app_context():
        origin = _collect(booking, "enc-40", "20.00")
        corrected = _correct(booking, "comp-5", origin["id"], "-5.00")
        assert corrected["recognized_net_chf"] == "15.00"
        assert corrected["remaining_due_chf"] == "30.00"
        kept = db.session.execute(
            text("SELECT net_chf, kind FROM cash_collection WHERE id = :id"),
            {"id": origin["id"]},
        ).one()
        assert kept.net_chf == Decimal("20.00")
        assert kept.kind == "collection"
        follow = _collect(booking, "enc-40-suite", "30.00")
        assert follow["recognized_net_chf"] == "45.00"
        with pytest.raises(PrepRefusal) as overrun:
            _collect(booking, "enc-40-trop", "1.00")
        assert overrun.value.code == "depassement_autorise"
        db.session.remove()


def test_annulation_complete_et_corrections_successives(app):
    booking = _booking(41)
    limited = _booking(42)
    with app.app_context():
        origin = _collect(booking, "enc-41", "20.00")
        cleared = _correct(booking, "comp-20", origin["id"], "-20.00")
        assert cleared["recognized_net_chf"] == "0.00"
        assert cleared["remaining_due_chf"] == "45.00"
        first = _collect(limited, "enc-42", "20.00")
        _correct(limited, "comp-42-a", first["id"], "-5.00")
        second = _correct(limited, "comp-42-b", first["id"], "-10.00")
        assert second["recognized_net_chf"] == "5.00"
        with pytest.raises(PrepRefusal) as too_much:
            _correct(limited, "comp-42-c", first["id"], "-6.00")
        assert too_much.value.code == "depassement_autorise"
        still = db.session.execute(
            text("SELECT net_chf FROM cash_collection WHERE id = :id"),
            {"id": first["id"]},
        ).scalar_one()
        assert still == Decimal("20.00")
        db.session.remove()


def test_correction_excessive_apres_cinq(app):
    booking = _booking(43)
    with app.app_context():
        origin = _collect(booking, "enc-43", "20.00")
        _correct(booking, "comp-43-a", origin["id"], "-5.00")
        with pytest.raises(PrepRefusal) as refused:
            _correct(booking, "comp-43-b", origin["id"], "-20.00")
        assert refused.value.code == "depassement_autorise"
        count = db.session.execute(
            text(
                "SELECT COUNT(*) FROM cash_collection "
                "WHERE compensates_collection_id = :id"
            ),
            {"id": origin["id"]},
        ).scalar_one()
        assert count == 1
        db.session.remove()


def test_origines_et_signes_refuses(app):
    booking = _booking(44)
    other = _booking(45)
    with app.app_context():
        origin = _collect(booking, "enc-44", "20.00")
        with pytest.raises(PrepRefusal) as missing:
            record_compensation(
                booking,
                {
                    "action": "compensate_cash",
                    "idempotency_key": "sans-origine",
                    "net_chf": "-1.00",
                    "reason": "sans origine",
                },
                driver_id=7,
                company_id=1,
                actor_id=9,
                operation_key="sans-origine",
            )
        assert missing.value.code == "origine_absente"
        with pytest.raises(PrepRefusal) as unknown:
            _correct(booking, "origine-inconnue", 99999, "-1.00")
        assert unknown.value.code == "origine_absente"
        with pytest.raises(PrepRefusal) as company:
            record_compensation(
                booking,
                {
                    "action": "compensate_cash",
                    "idempotency_key": "autre-ent",
                    "net_chf": "-1.00",
                    "compensates_collection_id": origin["id"],
                    "reason": "autre entreprise",
                },
                driver_id=7,
                company_id=2,
                actor_id=9,
                operation_key="autre-ent",
            )
        assert company.value.code == "autre_entreprise"
        with pytest.raises(PrepRefusal) as booking_error:
            _correct(other, "autre-res", origin["id"], "-1.00")
        assert booking_error.value.code == "autre_reservation"
        with pytest.raises(PrepRefusal) as zero:
            _correct(booking, "nul", origin["id"], "0.00")
        assert zero.value.code == "montant_nul"
        with pytest.raises(PrepRefusal) as positive:
            _correct(booking, "positif", origin["id"], "5.00")
        assert positive.value.code == "signe_invalide"
        db.session.remove()


def test_historique_idempotence_et_imputation(app):
    booking = _booking(46)
    with app.app_context():
        origin = _collect(booking, "enc-46", "20.00")
        _correct(
            booking, "comp-46", origin["id"], "-5.00", reason="billet compte deux fois"
        )
        replay = _correct(
            booking, "comp-46", origin["id"], "-5.00", reason="billet compte deux fois"
        )
        assert replay["replay"] is True
        db.session.remove()
        with app.app_context():
            again = _correct(
                booking,
                "comp-46",
                origin["id"],
                "-5.00",
                reason="billet compte deux fois",
            )
            assert again["replay"] is True
            with pytest.raises(PrepRefusal) as conflict:
                _correct(booking, "comp-46", origin["id"], "-4.00")
            assert conflict.value.code == "conflit_idempotence"
            actions = db.session.execute(
                text(
                    "SELECT action, actor_id FROM cash_collection_event "
                    "WHERE collection_id IN ("
                    "SELECT id FROM cash_collection WHERE booking_id = 46"
                    ") ORDER BY id"
                )
            ).all()
            db.session.execute(
                text(
                    "INSERT INTO invoice_payments "
                    "(invoice_id, amount, method, paid_at, cash_collection_id) "
                    "VALUES (50, 15.00, 'cash', TIMESTAMPTZ '2026-11-02 09:00:00+00', :origin)"
                ),
                {"origin": origin["id"]},
            )
            db.session.commit()
            with pytest.raises(PrepRefusal) as imputed:
                _correct(booking, "comp-46-impute", origin["id"], "-1.00")
            assert imputed.value.code == "correction_apres_imputation"
            db.session.execute(
                text("DELETE FROM invoice_payments WHERE cash_collection_id = :origin"),
                {"origin": origin["id"]},
            )
            db.session.commit()
            notes = db.session.execute(
                text("SELECT COUNT(*) FROM invoice_credit_note")
            ).scalar_one()
            handovers = db.session.execute(
                text("SELECT COUNT(*) FROM cash_handover")
            ).scalar_one()
            refunds = db.session.execute(
                text("SELECT COUNT(*) FROM invoice_refund")
            ).scalar_one()
            linked = db.session.execute(
                text(
                    "SELECT COUNT(*) FROM invoice_payments "
                    "WHERE cash_collection_id IS NOT NULL"
                )
            ).scalar_one()
            flags = db.session.execute(
                text(
                    "SELECT bool_or(production_activation) "
                    "OR bool_or(legally_approved) FROM cash_collection"
                )
            ).scalar_one()
            db.session.remove()
    assert [row.action for row in actions] == ["declare_cash", "compensate_cash"]
    assert actions[1].actor_id == 9
    assert notes == 0
    assert handovers == 0
    assert refunds == 0
    assert linked == 0
    assert flags is False


def test_corrections_concurrentes(app):
    booking = _booking(47)
    admissible = _booking(48)
    with app.app_context():
        origin = _collect(booking, "enc-47", "20.00")
        both = _collect(admissible, "enc-48", "20.00")
        origin_id = origin["id"]
        both_id = both["id"]
        db.session.remove()
    over = _race(app, booking, origin_id, (("-15.00", "race-a"), ("-15.00", "race-b")))
    assert sum(ok for ok, _code in over) == 1
    assert any(code == "depassement_autorise" for ok, code in over if not ok)
    ok = _race(app, admissible, both_id, (("-10.00", "race-c"), ("-5.00", "race-d")))
    assert all(flag for flag, _code in ok)
    with app.app_context():
        sums = db.session.execute(
            text(
                "SELECT compensates_collection_id, SUM(net_chf) "
                "FROM cash_collection WHERE compensates_collection_id IN (:a, :b) "
                "GROUP BY compensates_collection_id"
            ),
            {"a": origin_id, "b": both_id},
        ).all()
        db.session.remove()
    found = {row[0]: row[1] for row in sums}
    assert found[origin_id] == Decimal("-15.00")
    assert found[both_id] == Decimal("-15.00")


def test_chauffeur_non_affecte():
    booking = SimpleNamespace(
        id=1,
        company_id=1,
        driver_id=7,
        status=BookingStatus.IN_PROGRESS,
        price_amount=Decimal("45.00"),
    )
    with pytest.raises(PrepRefusal) as refused:
        refuse_booking(booking, driver_id=8, company_id=1)
    assert refused.value.code == "chauffeur_non_affecte"


def _race(app, booking, origin_id, pairs):
    barrier = threading.Barrier(2)
    found = []
    guard = threading.Lock()

    def _run(net, key):
        with app.app_context():
            barrier.wait()
            try:
                view = _correct(booking, key, origin_id, net)
                item = (True, view["recognized_net_chf"])
            except PrepRefusal as exc:
                item = (False, exc.code)
            finally:
                db.session.remove()
            with guard:
                found.append(item)

    threads = [threading.Thread(target=_run, args=(net, key)) for net, key in pairs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return found
