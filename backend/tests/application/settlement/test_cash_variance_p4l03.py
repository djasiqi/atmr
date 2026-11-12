"""P4-L03 — écarts de caisse, sans paiement patient."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from threading import Thread

import pytest
from sqlalchemy.exc import IntegrityError

from application.settlement.cash_ledger import reconcile, render_control_csv
from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db as app_db
from models import (
    CashCollectionEvent,
    CashHandover,
    CashReconciliation,
    Invoice,
    InvoiceCreditNote,
    InvoicePayment,
)
from models.enums import InvoiceStatus
from services.recipe_environment import database_name
from tests.application.settlement.test_cash_handover_p4l02 import (
    _confirm,
    _declare,
    _headers,
)
from tests.application.settlement.test_cash_position_p4l01 import (
    _collection,
    _held,
    _world,
)

pytestmark = pytest.mark.integration

REGISTER = "/api/v1/companies/me/cash-register"


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.fail(f"Base refusée pour cette recette : {current or 'absente'}")


def _count(world, origin, key, counted, *, handover_id, reason="inventaire"):
    return reconcile(
        origin.id,
        {
            "counted_chf": counted,
            "idempotency_key": key,
            "handover_id": handover_id,
            "reason": reason,
        },
        company_id=world["company"].id,
        actor_id=world["owner"].id,
    )


def _handover_id(view) -> int:
    return int(view["handover_id"])


def test_une_remise_et_deux_remises(db, prep):
    single = _world(db, due="90.00")
    origin = _collection(single, single["booking"], single["driver"], net="90.00", due="90.00")
    declared = _declare(single, "90.00", "totale")
    _confirm(single, "totale", collection_id=origin.id)
    before = _held(single["company"].id)["held_chf"]
    counted = _count(single, origin, "pile", "90.00", handover_id=_handover_id(declared), reason="pile")
    assert counted["reconciliation_variance_chf"] == "0.00"
    assert counted["replay"] is False
    assert _held(single["company"].id)["held_chf"] == before == "0.00"
    replay = _count(single, origin, "pile", "90.00", handover_id=_handover_id(declared), reason="pile")
    assert replay["replay"] is True
    assert CashReconciliation.query.filter_by(collection_id=origin.id).count() == 1

    several = _world(db, due="90.00")
    cash = _collection(several, several["booking"], several["driver"], net="90.00", due="90.00")
    first = _declare(several, "40.00", "part-40")
    second = _declare(several, "30.00", "part-30")
    _confirm(several, "part-40", collection_id=cash.id)
    _confirm(several, "part-30", collection_id=cash.id)
    assert _held(several["company"].id)["held_chf"] == "20.00"
    with pytest.raises(PrepRefusal) as ambiguous:
        reconcile(
            cash.id,
            {"counted_chf": "40.00", "idempotency_key": "sans-cible", "reason": "ambigu"},
            company_id=several["company"].id,
            actor_id=several["owner"].id,
        )
    assert ambiguous.value.code == "remise_ambigue"
    _count(several, cash, "rap-40", "40.00", handover_id=_handover_id(first), reason="première remise")
    _count(several, cash, "rap-30", "30.00", handover_id=_handover_id(second), reason="deuxième remise")
    assert _held(several["company"].id)["held_chf"] == "20.00"
    assert CashHandover.query.filter_by(collection_id=cash.id, state="confirmed").count() == 2
    assert CashReconciliation.query.filter_by(collection_id=cash.id, status="constate").count() == 2


def test_ecarts_regularisation_et_declaree(db, prep):
    short = _world(db, due="90.00")
    origin = _collection(short, short["booking"], short["driver"], net="90.00", due="90.00")
    declared = _declare(short, "90.00", "manque")
    with pytest.raises(PrepRefusal) as pending:
        _count(short, origin, "trop-tot", "85.00", handover_id=_handover_id(declared))
    assert pending.value.code == "remise_non_confirmee"
    _confirm(short, "manque", collection_id=origin.id)
    held = _held(short["company"].id)["held_chf"]
    gap = _count(short, origin, "moins-cinq", "85.00", handover_id=_handover_id(declared), reason="manque constaté")
    assert gap["reconciliation_variance_chf"] == "-5.00"
    assert _held(short["company"].id)["held_chf"] == held
    line_id = gap["reconciliation_id"]
    partial = reconcile(
        origin.id,
        {
            "regularizes_id": line_id,
            "amount_chf": "2.00",
            "idempotency_key": "reg-2",
            "reason": "retrouvé en partie",
        },
        company_id=short["company"].id,
        actor_id=short["owner"].id,
    )
    opened = next(item for item in partial["reconciliations"] if item["id"] == line_id)
    assert opened["variance_chf"] == "-5.00"
    assert opened["status"] == "constate"
    assert opened["open_variance_chf"] == "-3.00"
    child = next(item for item in partial["reconciliations"] if item["status"] == "regularisation")
    assert child["variance_chf"] == "2.00"
    assert child["regularizes_id"] == line_id
    with pytest.raises(PrepRefusal) as over:
        reconcile(
            origin.id,
            {
                "regularizes_id": line_id,
                "amount_chf": "4.00",
                "idempotency_key": "reg-trop",
                "reason": "trop",
            },
            company_id=short["company"].id,
            actor_id=short["owner"].id,
        )
    assert over.value.code == "depassement_autorise"
    closed = reconcile(
        origin.id,
        {
            "regularizes_id": line_id,
            "amount_chf": "3.00",
            "idempotency_key": "reg-3",
            "reason": "solde du manque",
        },
        company_id=short["company"].id,
        actor_id=short["owner"].id,
    )
    still = next(item for item in closed["reconciliations"] if item["id"] == line_id)
    assert still["variance_chf"] == "-5.00"
    assert still["open_variance_chf"] == "0.00"
    assert _held(short["company"].id)["held_chf"] == held

    extra = _world(db, due="90.00")
    other = _collection(extra, extra["booking"], extra["driver"], net="90.00", due="90.00")
    plus = _declare(extra, "90.00", "excedent")
    _confirm(extra, "excedent", collection_id=other.id)
    positive = _count(extra, other, "plus-cinq", "95.00", handover_id=_handover_id(plus), reason="excédent")
    assert positive["reconciliation_variance_chf"] == "5.00"
    with pytest.raises(PrepRefusal) as conflict:
        _count(extra, other, "plus-cinq", "94.00", handover_id=_handover_id(plus), reason="excédent")
    assert conflict.value.code == "conflit_idempotence"
    stored = app_db.session.get(CashReconciliation, positive["reconciliation_id"])
    assert f"{Decimal(stored.counted_chf):.2f}" == "95.00"


def test_facture_droits_export_et_historique(db, prep, client):
    world = _world(db, due="90.00")
    origin = _collection(world, world["booking"], world["driver"], net="90.00", due="90.00")
    declared = _declare(world, "90.00", "dossier")
    _confirm(world, "dossier", collection_id=origin.id)
    invoice = Invoice(
        company_id=world["company"].id,
        client_id=world["patient"].id,
        period_month=11,
        period_year=2026,
        invoice_number=f"P4E-{uuid.uuid4().hex[:8]}",
        currency="CHF",
        subtotal_amount=Decimal("90.00"),
        late_fee_amount=Decimal("0.00"),
        reminder_fee_amount=Decimal("0.00"),
        total_amount=Decimal("90.00"),
        amount_paid=Decimal("90.00"),
        balance_due=Decimal("0.00"),
        vat_total_amount=Decimal("0.00"),
        issued_at=datetime(2026, 11, 1, tzinfo=UTC),
        due_date=datetime(2026, 12, 31, tzinfo=UTC),
        status=InvoiceStatus.PAID,
        reminder_level=0,
    )
    app_db.session.add(invoice)
    app_db.session.commit()
    price = world["booking"].amount
    payments = InvoicePayment.query.count()
    notes = InvoiceCreditNote.query.count()
    _count(
        world,
        origin,
        "ecart-facture",
        "85.00",
        handover_id=_handover_id(declared),
        reason="écart sans patient",
    )
    app_db.session.expire_all()
    stored_invoice = app_db.session.get(Invoice, invoice.id)
    assert stored_invoice.amount_paid == Decimal("90.00")
    assert stored_invoice.balance_due == Decimal("0.00")
    assert stored_invoice.total_amount == Decimal("90.00")
    assert world["booking"].amount == price
    assert InvoicePayment.query.count() == payments
    assert InvoiceCreditNote.query.count() == notes
    actions = [
        row.action
        for row in app_db.session.query(CashCollectionEvent)
        .filter(CashCollectionEvent.collection_id == origin.id)
        .order_by(CashCollectionEvent.id)
        .all()
    ]
    assert actions == ["declare_handover", "confirm_handover", "reconcile"]
    fresh = reconcile(
        origin.id,
        {"counted_chf": "85.00", "idempotency_key": "ecart-facture", "handover_id": _handover_id(declared), "reason": "écart sans patient"},
        company_id=world["company"].id,
        actor_id=world["owner"].id,
    )
    csv = render_control_csv([fresh])
    assert "collection;remise_id;attendu;confirme;ecart;motif;statut" in csv.split("\n", 1)[0]
    assert "écart sans patient" in csv
    assert ";constate" in csv
    assert "-5.00" in csv
    assert str(_handover_id(declared)) in csv

    denied = client.post(
        f"{REGISTER}/{origin.id}/reconcile",
        json={"counted_chf": "85.00", "idempotency_key": "chauffeur", "reason": "non"},
        headers=_headers(client, world["driver_user"], world["company"].id, role="DRIVER"),
    )
    assert denied.status_code == 403
    other = _world(db)
    foreign = client.post(
        f"{REGISTER}/{origin.id}/reconcile",
        json={"counted_chf": "85.00", "idempotency_key": "etranger", "reason": "non"},
        headers=_headers(client, other["owner"], other["company"].id),
    )
    assert foreign.status_code == 404


def test_rapprochements_concurrents(db, prep, app):
    blocked = _world(db, due="20.00")
    origin = _collection(blocked, blocked["booking"], blocked["driver"], net="20.00", due="20.00")
    one = _declare(blocked, "20.00", "seul")
    _confirm(blocked, "seul", collection_id=origin.id)
    split = _world(db, due="20.00")
    both = _collection(split, split["booking"], split["driver"], net="20.00", due="20.00")
    left = _declare(split, "10.00", "gauche")
    right = _declare(split, "10.00", "droite")
    _confirm(split, "gauche", collection_id=both.id)
    _confirm(split, "droite", collection_id=both.id)

    def _race(world, collection_id, payloads):
        codes = []
        headers = _headers(app.test_client(), world["owner"], world["company"].id)

        def _run(payload):
            with app.app_context():
                try:
                    response = app.test_client().post(
                        f"{REGISTER}/{collection_id}/reconcile",
                        json=payload,
                        headers=headers,
                    )
                    codes.append(response.status_code)
                except Exception:
                    codes.append(0)
                finally:
                    app_db.session.remove()

        threads = [Thread(target=_run, args=(payload,)) for payload in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        return codes

    over = _race(
        blocked,
        origin.id,
        (
            {"counted_chf": "20.00", "idempotency_key": "course-a", "handover_id": _handover_id(one), "reason": "a"},
            {"counted_chf": "15.00", "idempotency_key": "course-b", "handover_id": _handover_id(one), "reason": "b"},
        ),
    )
    assert sorted(over) == [200, 409], over
    assert CashReconciliation.query.filter_by(handover_id=_handover_id(one)).count() == 1
    ok = _race(
        split,
        both.id,
        (
            {"counted_chf": "10.00", "idempotency_key": "lot-a", "handover_id": _handover_id(left), "reason": "a"},
            {"counted_chf": "10.00", "idempotency_key": "lot-b", "handover_id": _handover_id(right), "reason": "b"},
        ),
    )
    assert sorted(ok) == [200, 200], ok


def test_contrainte_un_constat_par_remise(db, prep):
    world = _world(db, due="20.00")
    origin = _collection(world, world["booking"], world["driver"], net="20.00", due="20.00")
    declared = _declare(world, "20.00", "sql")
    _confirm(world, "sql", collection_id=origin.id)
    handover_id = _handover_id(declared)
    common = {
        "handover_id": handover_id,
        "collection_id": origin.id,
        "company_id": world["company"].id,
        "driver_id": world["driver"].id,
        "expected_chf": Decimal("20.00"),
        "counted_chf": Decimal("20.00"),
        "variance_chf": Decimal("0.00"),
        "reason": "sql",
        "status": "constate",
        "author_user_id": world["owner"].id,
    }
    app_db.session.add(CashReconciliation(idempotency_key="sql-a", **common))
    app_db.session.commit()
    app_db.session.add(CashReconciliation(idempotency_key="sql-b", **common))
    with pytest.raises(IntegrityError):
        app_db.session.commit()
    app_db.session.rollback()
