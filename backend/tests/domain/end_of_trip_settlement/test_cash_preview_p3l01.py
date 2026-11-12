"""Prévisualisation des acomptes. Lecture seule, base jetable."""

import os
from decimal import Decimal

import pytest
from flask import Flask
from sqlalchemy import text

from application.invoices.cash_preview import (
    load_patient_cash_preview,
    merge_leg_positions,
)
from application.invoices.period_invoice_preview import (
    PeriodPreviewLine,
    _attach_patient_cash,
)
from ext import db

pytestmark = pytest.mark.skipif(
    not os.getenv("P3L01_DATABASE_URL"),
    reason="P3L01_DATABASE_URL absente",
)


@pytest.fixture(scope="module")
def app():
    flask = Flask("p3l01")
    flask.config["SQLALCHEMY_DATABASE_URI"] = os.environ["P3L01_DATABASE_URL"]
    flask.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(flask)
    return flask


def _counts(connection):
    names = (
        "invoices",
        "invoice_lines",
        "invoice_payments",
        "invoice_credit_note",
        "cash_collection",
        "cash_handover",
    )
    return {
        name: connection.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
        for name in names
    }


def _insert(connection, **values):
    columns = ", ".join(values)
    names = ", ".join(f":{key}" for key in values)
    connection.execute(
        text(f"INSERT INTO cash_collection ({columns}) VALUES ({names})"),
        values,
    )


def _line(**overrides):
    values = {
        "booking_id": 70,
        "company_id": 1,
        "driver_id": 7,
        "debtor_label": "Essai",
        "amount_due_chf": "45.00",
        "decision_version": "v1",
        "kind": "collection",
        "client_operation_key": "preview-70",
        "state": "administrative",
        "disposition": "partial",
        "net_chf": "20.00",
    }
    values.update(overrides)
    return values


def test_previsualisation_des_acomptes(app):
    dues = {
        70: Decimal("45.00"),
        71: Decimal("45.00"),
        72: Decimal("45.00"),
        73: Decimal("45.00"),
        74: Decimal("45.00"),
        75: Decimal("45.00"),
        76: Decimal("45.00"),
        77: Decimal("45.00"),
        78: Decimal("45.00"),
        79: Decimal("45.00"),
        80: Decimal("45.00"),
        81: Decimal("45.00"),
    }
    with app.app_context():
        before = _counts(db.session)
        price_before = db.session.execute(
            text("SELECT price_amount FROM booking WHERE id = 70")
        ).scalar_one()
        _insert(db.session, **_line(booking_id=71, client_operation_key="p71", net_chf="20.00"))
        _insert(
            db.session,
            **_line(booking_id=72, client_operation_key="p72a", net_chf="20.00"),
        )
        _insert(
            db.session,
            **_line(booking_id=72, client_operation_key="p72b", net_chf="15.00"),
        )
        origin = db.session.execute(
            text(
                "INSERT INTO cash_collection ("
                "booking_id, company_id, driver_id, debtor_label, amount_due_chf, "
                "net_chf, decision_version, kind, client_operation_key, state, disposition"
                ") VALUES ("
                "73, 1, 7, 'Essai', 45.00, 20.00, 'v1', 'collection', 'p73', "
                "'administrative', 'partial') RETURNING id"
            )
        ).scalar_one()
        _insert(
            db.session,
            **_line(
                booking_id=73,
                client_operation_key="p73c",
                kind="compensation",
                state="administrative",
                disposition="correction",
                net_chf="-5.00",
                compensates_collection_id=origin,
            ),
        )
        _insert(
            db.session,
            **_line(
                booking_id=74,
                client_operation_key="p74",
                state="uncertain",
                disposition="partial",
            ),
        )
        _insert(
            db.session,
            **_line(
                booking_id=75,
                client_operation_key="p75",
                state="administrative",
                disposition="contested",
            ),
        )
        _insert(
            db.session,
            **_line(
                booking_id=76,
                company_id=2,
                client_operation_key="p76",
                net_chf="20.00",
            ),
        )
        allocated = db.session.execute(
            text(
                "INSERT INTO cash_collection ("
                "booking_id, company_id, driver_id, debtor_label, amount_due_chf, "
                "net_chf, decision_version, kind, client_operation_key, state, disposition"
                ") VALUES ("
                "77, 1, 7, 'Essai', 45.00, 20.00, 'v1', 'collection', 'p77', "
                "'administrative', 'partial') RETURNING id"
            )
        ).scalar_one()
        db.session.execute(
            text(
                "INSERT INTO invoice_payments "
                "(invoice_id, amount, method, cash_collection_id) "
                "VALUES (50, 20.00, 'cash', :collection_id)"
            ),
            {"collection_id": allocated},
        )
        _insert(
            db.session,
            **_line(booking_id=78, client_operation_key="p78", net_chf="20.00"),
        )
        _insert(
            db.session,
            **_line(
                booking_id=79,
                client_operation_key="p79",
                state="received",
                disposition=None,
                net_chf="45.00",
            ),
        )
        _insert(
            db.session,
            **_line(booking_id=80, client_operation_key="p80", net_chf="20.00"),
        )
        db.session.flush()
        prepared = _counts(db.session)
        positions = load_patient_cash_preview(1, dues)
        after = _counts(db.session)
        price_after = db.session.execute(
            text("SELECT price_amount, invoice_line_id FROM booking WHERE id = 70")
        ).one()
        db.session.rollback()

    assert after == prepared
    assert prepared["cash_collection"] > before["cash_collection"]
    assert price_before == Decimal("45.00")
    assert price_after.price_amount == Decimal("45.00")
    assert price_after.invoice_line_id is None
    assert positions[70]["recognized_net_chf"] == "0.00"
    assert positions[70]["remaining_due_chf"] == "45.00"
    assert positions[70]["contractual_chf"] == "45.00"
    assert positions[71]["recognized_net_chf"] == "20.00"
    assert positions[71]["remaining_due_chf"] == "25.00"
    assert positions[72]["remaining_due_chf"] == "10.00"
    assert positions[73]["recognized_net_chf"] == "15.00"
    assert positions[73]["remaining_due_chf"] == "30.00"
    assert positions[74]["remaining_due_chf"] == "45.00"
    assert positions[74]["imputable_collection_ids"] == []
    assert positions[75]["remaining_due_chf"] == "45.00"
    assert positions[76]["remaining_due_chf"] == "45.00"
    assert positions[77]["remaining_due_chf"] == "45.00"
    assert positions[77]["imputable_collection_ids"] == []
    assert positions[79]["remaining_due_chf"] == "0.00"
    assert positions[79]["contractual_chf"] == "45.00"
    outbound = positions[78]
    inbound = positions[81]
    merged = merge_leg_positions([outbound, inbound])
    assert merged["contractual_chf"] == "90.00"
    assert merged["recognized_net_chf"] == "20.00"
    assert merged["remaining_due_chf"] == "70.00"
    assert merged["cash_legs"][0]["contractual_chf"] == "45.00"
    assert merged["cash_legs"][1]["remaining_due_chf"] == "45.00"
    assert positions[80]["remaining_due_chf"] == "25.00"
    assert positions[81]["remaining_due_chf"] == "45.00"
    line = PeriodPreviewLine(
        booking_id=71,
        scheduled_at=None,
        amount_ht=45.0,
        origin_amount_ht=45.0,
        description="Transport",
        source_type="ride",
        is_locked=False,
        already_invoiced=False,
    )
    annotated = _attach_patient_cash([line], positions, merge_leg_positions)
    assert annotated[0].amount_ht == 45.0
    assert annotated[0].recognized_net_chf == "20.00"
    assert annotated[0].remaining_due_chf == "25.00"


def test_schema_absent_n_invente_pas_de_zero(app, monkeypatch):
    monkeypatch.setattr(
        "application.invoices.cash_preview._ready", lambda *_args: False
    )
    with app.app_context():
        assert load_patient_cash_preview(1, {70: Decimal("45.00")}) is None
        db.session.rollback()
