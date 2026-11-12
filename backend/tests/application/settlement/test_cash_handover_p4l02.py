"""P4-L02 — plusieurs remises, sans second paiement patient."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from threading import Thread

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy.exc import IntegrityError

from application.invoices.issue_refund import RefundRefusal, record_refund
from application.settlement.cash_ledger import confirm_handover, declare_handover
from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db as app_db
from models import (
    CashCollection,
    CashCollectionEvent,
    CashHandover,
    Invoice,
    InvoiceCreditNote,
    InvoicePayment,
)
from models.enums import InvoiceStatus
from services.recipe_environment import database_name
from tests.application.settlement.test_cash_position_p4l01 import (
    _collection,
    _compensation,
    _held,
    _world,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")


def _declare(world, amount, key):
    return declare_handover(
        world["booking"],
        driver_id=world["driver"].id,
        company_id=world["company"].id,
        actor_id=world["driver_user"].id,
        idempotency_key=key,
        amount_chf=amount,
    )


def _confirm(world, key, *, collection_id):
    return confirm_handover(
        collection_id,
        company_id=world["company"].id,
        actor_id=world["owner"].id,
        idempotency_key=key,
    )


def _headers(
    client, user, company_id: int, *, role: str | None = None
) -> dict[str, str]:
    claims = {
        "role": role or user.role.value,
        "company_id": company_id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def test_remise_totale_et_deux_partielles(db, prep):
    total = _world(db, due="20.00")
    origin = _collection(
        total, total["booking"], total["driver"], net="20.00", due="20.00"
    )
    declared = _declare(total, None, "totale")
    assert declared["handover_state"] == "declared"
    assert declared["handover_amount_chf"] == "20.00"
    assert declared["handovers"][0]["amount_chf"] == "20.00"
    assert declared["handovers"][0]["state"] == "declared"
    assert declared["handovers"][0]["idempotency_key"] == "totale"
    assert _held(total["company"].id)["held_chf"] == "20.00"
    confirmed = _confirm(total, "totale", collection_id=origin.id)
    assert confirmed["replay"] is False
    assert confirmed["handover_state"] == "confirmed"
    assert _held(total["company"].id)["held_chf"] == "0.00"
    assert app_db.session.get(CashCollection, origin.id).net_chf == Decimal("20.00")

    partial = _world(db, due="20.00")
    second = _collection(
        partial, partial["booking"], partial["driver"], net="20.00", due="20.00"
    )
    _declare(partial, "10.00", "part-1")
    _declare(partial, "10.00", "part-2")
    _confirm(partial, "part-1", collection_id=second.id)
    both = _confirm(partial, "part-2", collection_id=second.id)
    assert sorted(item["amount_chf"] for item in both["handovers"]) == [
        "10.00",
        "10.00",
    ]
    assert {item["state"] for item in both["handovers"]} == {"confirmed"}
    position = _held(partial["company"].id)
    assert position["handed_over_chf"] == "20.00"
    assert position["held_chf"] == "0.00"
    assert (
        CashHandover.query.filter_by(collection_id=second.id, state="confirmed").count()
        == 2
    )


def test_depassement_rejeu_et_conflit(db, prep):
    world = _world(db, due="20.00")
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="20.00"
    )
    _declare(world, "15.00", "quinze")
    _declare(world, "10.00", "dix")
    assert _held(world["company"].id)["held_chf"] == "20.00"
    _confirm(world, "quinze", collection_id=origin.id)
    with pytest.raises(PrepRefusal) as refused:
        _confirm(world, "dix", collection_id=origin.id)
    assert refused.value.code == "depassement_autorise"
    assert _held(world["company"].id)["held_chf"] == "5.00"
    with pytest.raises(PrepRefusal) as late:
        _declare(world, "6.00", "trop-tard")
    assert late.value.code == "depassement_autorise"
    replay = _declare(world, "15.00", "quinze")
    assert replay["replay"] is True
    assert CashHandover.query.filter_by(collection_id=origin.id).count() == 2
    again = _confirm(world, "quinze", collection_id=origin.id)
    assert again["replay"] is True
    with pytest.raises(PrepRefusal) as conflict:
        _declare(world, "14.00", "quinze")
    assert conflict.value.code == "conflit_idempotence"
    with pytest.raises(PrepRefusal) as zero:
        _declare(world, "0.00", "nul")
    assert zero.value.code == "montant_nul"
    with pytest.raises(PrepRefusal) as negative:
        _declare(world, "-1.00", "negatif")
    assert negative.value.code == "montant_nul"


def test_correction_et_remboursement_avant_remise(db, prep):
    world = _world(db, due="20.00")
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="20.00"
    )
    _compensation(origin, net="-5.00", disposition="correction")
    with pytest.raises(PrepRefusal) as too_much:
        _declare(world, "16.00", "trop-correction")
    assert too_much.value.code == "depassement_autorise"
    _declare(world, "15.00", "apres-correction")
    _confirm(world, "apres-correction", collection_id=origin.id)
    assert _held(world["company"].id)["held_chf"] == "0.00"

    example = _world(db, due="90.00")
    shown = _collection(
        example, example["booking"], example["driver"], net="90.00", due="90.00"
    )
    _declare(example, "40.00", "exemple-40")
    _confirm(example, "exemple-40", collection_id=shown.id)
    _declare(example, "30.00", "exemple-30")
    _confirm(example, "exemple-30", collection_id=shown.id)
    shown_position = _held(example["company"].id)
    assert shown_position["handed_over_chf"] == "70.00"
    assert shown_position["held_chf"] == "20.00"

    refunded = _world(db, due="90.00")
    cash = _collection(
        refunded, refunded["booking"], refunded["driver"], net="90.00", due="90.00"
    )
    _declare(refunded, "40.00", "remise-40")
    _confirm(refunded, "remise-40", collection_id=cash.id)
    with pytest.raises(RefundRefusal) as blocked:
        record_refund(
            company_id=refunded["company"].id,
            created_by_user_id=refunded["owner"].id,
            amount_chf=Decimal("51.00"),
            reason="Espèces déjà remises",
            idempotency_key="rbt-trop-remise",
            confirm=True,
            cash_collection_id=cash.id,
        )
    assert blocked.value.code == "especes_deja_remises"
    record_refund(
        company_id=refunded["company"].id,
        created_by_user_id=refunded["owner"].id,
        amount_chf=Decimal("30.00"),
        reason="Reste chez le chauffeur",
        idempotency_key="rbt-reste",
        confirm=True,
        cash_collection_id=cash.id,
    )
    assert _held(refunded["company"].id)["held_chf"] == "20.00"
    with pytest.raises(PrepRefusal) as over_refund:
        _declare(refunded, "21.00", "apres-remboursement")
    assert over_refund.value.code == "depassement_autorise"
    _declare(refunded, "20.00", "reste-chauffeur")
    _confirm(refunded, "reste-chauffeur", collection_id=cash.id)
    position = _held(refunded["company"].id)
    assert position["held_chf"] == "0.00"
    assert Decimal(position["held_chf"]) >= 0


def test_lecture_facture_et_droits(db, prep, client):
    world = _world(db, due="20.00")
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="20.00"
    )
    invoice = Invoice(
        company_id=world["company"].id,
        client_id=world["patient"].id,
        period_month=11,
        period_year=2026,
        invoice_number=f"P4R-{uuid.uuid4().hex[:8]}",
        currency="CHF",
        subtotal_amount=Decimal("20.00"),
        late_fee_amount=Decimal("0.00"),
        reminder_fee_amount=Decimal("0.00"),
        total_amount=Decimal("20.00"),
        amount_paid=Decimal("20.00"),
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
    _declare(world, "10.00", "sans-paiement")
    _confirm(world, "sans-paiement", collection_id=origin.id)
    app_db.session.expire_all()
    stored = app_db.session.get(Invoice, invoice.id)
    assert stored.amount_paid == Decimal("20.00")
    assert stored.balance_due == Decimal("0.00")
    assert stored.total_amount == Decimal("20.00")
    assert world["booking"].amount == price
    assert InvoicePayment.query.count() == payments
    assert InvoiceCreditNote.query.count() == notes
    actions = [
        row.action
        for row in app_db.session.query(CashCollectionEvent)
        .join(CashCollection, CashCollectionEvent.collection_id == CashCollection.id)
        .filter(CashCollection.id == origin.id)
        .order_by(CashCollectionEvent.id)
        .all()
    ]
    assert actions == ["declare_handover", "confirm_handover"]

    driver_headers = _headers(
        client, world["driver_user"], world["company"].id, role="DRIVER"
    )
    denied = client.post(
        f"/api/v1/companies/me/cash-register/{origin.id}/confirm-handover",
        json={"idempotency_key": "sans-paiement"},
        headers=driver_headers,
    )
    assert denied.status_code == 403
    other = _world(db)
    foreign = client.post(
        f"/api/v1/companies/me/cash-register/{origin.id}/confirm-handover",
        json={"idempotency_key": "sans-paiement"},
        headers=_headers(client, other["owner"], other["company"].id),
    )
    assert foreign.status_code == 404


def test_confirmations_concurrentes(db, prep, app):
    blocked = _world(db, due="20.00")
    origin = _collection(
        blocked, blocked["booking"], blocked["driver"], net="20.00", due="20.00"
    )
    _declare(blocked, "15.00", "course-a")
    _declare(blocked, "15.00", "course-b")
    admitted = _world(db, due="20.00")
    both = _collection(
        admitted, admitted["booking"], admitted["driver"], net="20.00", due="20.00"
    )
    _declare(admitted, "10.00", "lot-a")
    _declare(admitted, "10.00", "lot-b")

    def _race(world, collection_id, keys):
        codes = []
        headers = _headers(app.test_client(), world["owner"], world["company"].id)

        def _run(key):
            with app.app_context():
                try:
                    response = app.test_client().post(
                        f"/api/v1/companies/me/cash-register/{collection_id}/confirm-handover",
                        json={"idempotency_key": key},
                        headers=headers,
                    )
                    codes.append(response.status_code)
                except Exception:
                    codes.append(0)
                finally:
                    app_db.session.remove()

        threads = [Thread(target=_run, args=(key,)) for key in keys]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        return codes

    over = _race(blocked, origin.id, ("course-a", "course-b"))
    assert sorted(over) == [200, 409], over
    assert _held(blocked["company"].id)["held_chf"] == "5.00"
    ok = _race(admitted, both.id, ("lot-a", "lot-b"))
    assert sorted(ok) == [200, 200], ok
    assert _held(admitted["company"].id)["held_chf"] == "0.00"


def test_deux_remises_autorisees_par_la_base(db, prep):
    world = _world(db, due="20.00")
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="20.00"
    )
    first = CashHandover(
        collection_id=origin.id,
        company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount_chf=Decimal("10.00"),
        idempotency_key="sql-a",
        state="declared",
        declared_by_user_id=world["owner"].id,
    )
    second = CashHandover(
        collection_id=origin.id,
        company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount_chf=Decimal("10.00"),
        idempotency_key="sql-b",
        state="declared",
        declared_by_user_id=world["owner"].id,
    )
    app_db.session.add(first)
    app_db.session.add(second)
    app_db.session.commit()
    assert CashHandover.query.filter_by(collection_id=origin.id).count() == 2
    app_db.session.add(
        CashHandover(
            collection_id=origin.id,
            company_id=world["company"].id,
            driver_id=world["driver"].id,
            amount_chf=Decimal("1.00"),
            idempotency_key="sql-a",
            state="declared",
            declared_by_user_id=world["owner"].id,
        )
    )
    with pytest.raises(IntegrityError):
        app_db.session.commit()
    app_db.session.rollback()
    app_db.session.add(
        CashHandover(
            collection_id=origin.id,
            company_id=world["company"].id,
            driver_id=world["driver"].id,
            amount_chf=Decimal("0.00"),
            idempotency_key="sql-zero",
            state="declared",
            declared_by_user_id=world["owner"].id,
        )
    )
    with pytest.raises(IntegrityError):
        app_db.session.commit()
    app_db.session.rollback()
