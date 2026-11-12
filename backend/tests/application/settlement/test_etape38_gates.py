"""Gardes espèces et dossier patient. Base jetable, rien n'est émis."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select

from ext import db
from models.client import Client
from models.enums import (
    BookingStatus,
    ClientType,
    InvoiceLineType,
    InvoiceStatus,
    PaymentStatus,
    UserRole,
)
from models.invoice import Invoice, InvoiceLine
from models.payment import Payment
from models.settlement_production import (
    CashCollection,
    CashReconciliation,
    PatientBillingDossier,
)
from models.user import User
from services.recipe_environment import database_name
from tests.application.settlement.test_production_prep_etape28 import (
    CASH,
    DOSSIERS,
    PATIENT,
    PATIENT_TX,
    RECIPE,
    REGISTER,
    _booking,
    _cash,
    _company_driver,
    _driver,
    _headers,
    _open_cash,
    _owner,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")


@pytest.fixture
def world(db, app, prep):
    current = database_name()
    if not (current or "").startswith(RECIPE):
        pytest.fail(f"Base refusée pour cette recette : {current or 'absente'}")
    suffix = uuid.uuid4().hex[:8]
    company, owner, driver, driver_user = _company_driver(f"Garde-{suffix}", suffix)
    other, other_owner, _driver, _user = _company_driver(
        f"SECRET-GARDE-{suffix}", f"b-{suffix}"
    )
    db.session.commit()
    return {
        "app": app,
        "suffix": suffix,
        "company": company,
        "owner": owner,
        "other_owner": other_owner,
        "other": other,
        "driver": driver,
        "driver_user": driver_user,
    }


def _pay(booking, amount: float) -> None:
    db.session.add(
        Payment(
            amount=amount,
            method="credit_card",
            status=PaymentStatus.COMPLETED,
            booking_id=booking.id,
        )
    )
    db.session.commit()


def _issued_invoice(world, booking) -> Invoice:
    user = User()
    user.username = f"cli-{uuid.uuid4().hex[:8]}"
    user.email = f"{user.username}@lirie-e2e.invalid"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.set_password("recette-isolee", force_change=False)
    db.session.add(user)
    db.session.flush()
    patient = Client(
        user_id=user.id,
        company_id=world["company"].id,
        client_type=ClientType.TRANSPORT,
        billing_address="Rue fictive 1, 1000 Recette",
    )
    db.session.add(patient)
    db.session.flush()
    invoice = Invoice(
        company_id=world["company"].id,
        client_id=patient.id,
        invoice_number=f"F-{uuid.uuid4().hex[:8]}",
        period_year=2026,
        period_month=10,
        status=InvoiceStatus.SENT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal("45.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("45.00"),
    )
    db.session.add(invoice)
    db.session.flush()
    db.session.add(
        InvoiceLine(
            invoice_id=invoice.id,
            reservation_id=booking.id,
            type=InvoiceLineType.RIDE,
            description="Transport fictif",
            qty=Decimal("1.00"),
            unit_price=Decimal("45.00"),
            line_total=Decimal("45.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("45.00"),
        )
    )
    db.session.commit()
    return invoice


def _collections(booking_id: int) -> int:
    return int(
        db.session.scalar(
            select(func.count())
            .select_from(CashCollection)
            .where(CashCollection.booking_id == booking_id)
        )
        or 0
    )


def _open(client, world, booking):
    return client.post(CASH.format(booking_id=booking.id), json={}, headers=_driver(world))


def test_course_deja_payee_facturee_et_institution(client, world):
    paid = _booking(world, customer_name="Déjà payé")
    _pay(paid, 45.0)
    blocked = _open(client, world, paid)
    assert blocked.status_code == 409
    assert blocked.get_json()["error_code"] == "deja_paye"
    assert _collections(paid.id) == 0

    surplus = _booking(world, customer_name="Excédent")
    _pay(surplus, 60.0)
    extra = _open(client, world, surplus)
    assert extra.status_code == 409
    assert extra.get_json()["error_code"] == "deja_paye"

    issued_booking = _booking(world, customer_name="Facturé")
    invoice = _issued_invoice(world, issued_booking)
    issued = _open(client, world, issued_booking)
    assert issued.status_code == 409
    assert issued.get_json()["error_code"] == "facture_emise"
    assert _collections(issued_booking.id) == 0
    assert invoice.status == InvoiceStatus.SENT

    clinic = _booking(world, billed_to_type="clinic", customer_name="Clinique")
    institution = _open(client, world, clinic)
    assert institution.status_code == 409
    assert institution.get_json()["error_code"] == "institution_facturee"

    cancelled = _booking(world, customer_name="Annulé")
    cancelled.status = BookingStatus.CANCELED
    db.session.commit()
    stopped = _open(client, world, cancelled)
    assert stopped.status_code == 409
    assert stopped.get_json()["error_code"] == "transport_annule"

    loose = _booking(world, customer_name="Sans chauffeur")
    loose.driver_id = None
    db.session.commit()
    unassigned = _open(client, world, loose)
    assert unassigned.status_code == 403
    assert unassigned.get_json()["error_code"] == "chauffeur_non_affecte"


def test_aller_retour_prepaye_et_couverture_ambigue(client, world):
    parent = _booking(world, customer_name="Aller")
    child = _booking(
        world,
        customer_name="Retour",
        parent_booking_id=parent.id,
        is_round_trip=True,
    )
    _pay(parent, 90.0)
    parent_cash = _open(client, world, parent)
    assert parent_cash.status_code == 409
    assert parent_cash.get_json()["error_code"] == "deja_paye"
    child_cash = _open(client, world, child)
    assert child_cash.status_code == 409
    assert child_cash.get_json()["error_code"] == "aller_retour_bloque"
    assert _collections(parent.id) == 0
    assert _collections(child.id) == 0

    unclear_parent = _booking(world, customer_name="Aller ambigu")
    unclear_child = _booking(
        world,
        customer_name="Retour ambigu",
        parent_booking_id=unclear_parent.id,
        is_round_trip=True,
    )
    _pay(unclear_parent, 80.0)
    unclear = _open(client, world, unclear_parent)
    assert unclear.status_code == 409
    assert unclear.get_json()["error_code"] == "couverture_ambigue"
    assert _collections(unclear_parent.id) == 0
    assert _collections(unclear_child.id) == 0


def test_montant_perime_dossier_confirme_et_mission_terminee(client, world):
    booking = _booking(world, customer_name="Montant")
    opened = _open_cash(client, world, booking)
    assert opened.status_code == 200
    stored_due = opened.get_json()["amount_due_chf"]
    booking.price_amount = Decimal("50.00")
    booking.amount = 50.0
    db.session.commit()
    stale = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "perime",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    assert stale.status_code == 409
    assert stale.get_json()["error_code"] == "decision_obsolete"
    row = db.session.scalar(
        select(CashCollection).where(CashCollection.booking_id == booking.id)
    )
    assert row.state == "administrative"
    assert f"{Decimal(row.amount_due_chf):.2f}" == stored_due
    assert row.tendered_chf is None

    fresh = _booking(world, customer_name="Confirmé")
    _open_cash(client, world, fresh)
    first = _cash(
        client,
        world,
        fresh,
        {
            "action": "declare_cash",
            "idempotency_key": "une",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    assert first.status_code == 200
    second = _cash(
        client,
        world,
        fresh,
        {
            "action": "declare_cash",
            "idempotency_key": "deux",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    assert second.status_code == 409
    assert second.get_json()["error_code"] == "depassement_autorise"
    assert _collections(fresh.id) == 1

    done = _booking(world, customer_name="Terminée payée")
    done.status = BookingStatus.COMPLETED
    db.session.commit()
    _pay(done, 45.0)
    finished = _open(client, world, done)
    assert finished.status_code == 409
    assert finished.get_json()["error_code"] == "deja_paye"
    assert _collections(done.id) == 0


def test_rapprochement_definitif_non_ecrase(client, world):
    exact = _booking(world, customer_name="Pile")
    _open_cash(client, world, exact)
    _cash(
        client,
        world,
        exact,
        {
            "action": "declare_cash",
            "idempotency_key": "recu",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    handed = _cash(
        client,
        world,
        exact,
        {"action": "declare_handover", "idempotency_key": "remise"},
    )
    collection_id = handed.get_json()["id"]
    client.post(f"{REGISTER}/{collection_id}/confirm-handover", headers=_owner(world))
    normal = client.post(
        f"{REGISTER}/{collection_id}/reconcile",
        json={"counted_chf": "45.00", "idempotency_key": "normal"},
        headers=_owner(world),
    )
    assert normal.status_code == 200
    assert normal.get_json()["reconciliation_variance_chf"] == "0.00"
    rewrite = client.post(
        f"{REGISTER}/{collection_id}/reconcile",
        json={"counted_chf": "40.00", "idempotency_key": "correction"},
        headers=_owner(world),
    )
    assert rewrite.status_code == 409
    assert rewrite.get_json()["error_code"] == "rapprochement_deja_enregistre"
    stored = db.session.scalar(
        select(CashReconciliation).where(CashReconciliation.company_id == world["company"].id)
    )
    assert f"{Decimal(stored.counted_chf):.2f}" == "45.00"
    assert f"{Decimal(stored.variance_chf):.2f}" == "0.00"
    assert f"{Decimal(stored.expected_chf):.2f}" == "45.00"

    short = _booking(world, customer_name="Écart")
    _open_cash(client, world, short)
    _cash(
        client,
        world,
        short,
        {
            "action": "declare_cash",
            "idempotency_key": "recu-ecart",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    handed_short = _cash(
        client,
        world,
        short,
        {"action": "declare_handover", "idempotency_key": "remise-ecart"},
    )
    short_id = handed_short.get_json()["id"]
    client.post(f"{REGISTER}/{short_id}/confirm-handover", headers=_owner(world))
    gap = client.post(
        f"{REGISTER}/{short_id}/reconcile",
        json={"counted_chf": "40.00", "idempotency_key": "ecart"},
        headers=_owner(world),
    )
    assert gap.status_code == 200
    assert gap.get_json()["reconciliation_variance_chf"] == "-5.00"
    assert gap.get_json()["amount_due_chf"] == "45.00"
    assert gap.get_json()["net_chf"] == "45.00"
    again = client.post(
        f"{REGISTER}/{short_id}/reconcile",
        json={"counted_chf": "45.00", "idempotency_key": "autre"},
        headers=_owner(world),
    )
    assert again.status_code == 409
    rows = db.session.scalars(
        select(CashReconciliation).where(
            CashReconciliation.company_id == world["company"].id
        )
    ).all()
    assert len(rows) == 2
    kept = [row for row in rows if f"{Decimal(row.variance_chf):.2f}" == "-5.00"]
    assert len(kept) == 1
    assert f"{Decimal(kept[0].counted_chf):.2f}" == "40.00"


def test_conservation_sans_quittance(client, world):
    booking = _booking(world, customer_name="Conservation")
    opened = _open_cash(client, world, booking)
    collection_id = opened.get_json()["id"]
    held = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "garde",
            "tendered_chf": "30.00",
            "change_chf": "0.00",
            "change_confirmed": False,
            "disposition": "held",
        },
    )
    assert held.status_code == 200
    assert held.get_json()["state"] == "administrative"
    assert held.get_json()["amount_due_chf"] == "45.00"
    assert held.get_json()["net_chf"] == "0.00"
    receipt = client.get(f"{REGISTER}/{collection_id}/receipt", headers=_owner(world))
    assert receipt.status_code == 404
    imputed = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "impute",
            "tendered_chf": "30.00",
            "change_chf": "0.00",
            "change_confirmed": False,
            "disposition": "imputed",
        },
    )
    assert imputed.status_code == 409
    row = db.session.get(CashCollection, collection_id)
    assert row.state == "administrative"
    assert f"{Decimal(row.amount_due_chf):.2f}" == "45.00"


def test_ouverture_concurrente(app, world):
    booking = _booking(world)
    booking_id = int(booking.id)
    driver_id = int(world["driver"].id)
    company_id = int(world["company"].id)
    actor_id = int(world["driver_user"].id)
    db.session.remove()
    from application.settlement.cash_ledger import open_collection
    from models.booking import Booking

    barrier = threading.Barrier(2)
    opened: list[int] = []
    errors: list[str] = []

    def attempt() -> None:
        with app.app_context():
            db.session.remove()
            try:
                current = db.session.get(Booking, booking_id)
                barrier.wait(timeout=5)
                view = open_collection(
                    current,
                    driver_id=driver_id,
                    company_id=company_id,
                    actor_id=actor_id,
                )
                opened.append(int(view["id"]))
            except Exception as exc:  # noqa: BLE001
                errors.append(type(exc).__name__)
            finally:
                db.session.remove()

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
    assert errors == []
    assert len(set(opened)) == 1
    assert _collections(booking_id) == 1


def test_dossier_patient_representant_et_brouillon(client, world):
    booking = _booking(world, customer_name="Patient signature")
    opened = client.post(
        PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world)
    )
    assert opened.status_code == 200
    dossier_id = opened.get_json()["id"]
    prior = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "prior_information",
            "idempotency_key": "info",
            "signatory_role": "patient",
            "presented_text": "brouillon non approuve",
        },
        headers=_driver(world),
    )
    assert prior.status_code == 200
    assert prior.get_json()["contractual_basis"] == "prior_presented"
    assert prior.get_json()["legally_binding"] is False
    refused = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "refusal", "idempotency_key": "refus", "signatory_role": "patient"},
        headers=_driver(world),
    )
    assert refused.status_code == 200
    assert refused.get_json()["contractual_basis"] == "refused"
    representative = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "representative",
            "idempotency_key": "rep",
            "signatory_role": "representative",
            "representative_name": "Proche fictif",
            "representative_kind": "proche",
        },
        headers=_driver(world),
    )
    assert representative.status_code == 200
    assert representative.get_json()["representative_powers_verified"] is False
    assert representative.get_json()["representative_review"] == "pending_review"
    driver_review = client.post(
        f"{DOSSIERS}/{dossier_id}/review",
        json={"result": "documented_scope", "idempotency_key": "chauffeur"},
        headers=_driver(world),
    )
    assert driver_review.status_code == 403
    approved = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "approve_representative", "idempotency_key": "pouvoir"},
        headers=_driver(world),
    )
    assert approved.status_code == 403
    review = client.post(
        f"{DOSSIERS}/{dossier_id}/review",
        json={
            "result": "documented_scope",
            "idempotency_key": "revue",
            "document_reference": "piece-fictive",
            "scope_note": "périmètre annoncé, non interprété",
            "relation": "proche",
            "powers_verified": False,
        },
        headers=_owner(world),
    )
    assert review.status_code == 200, review.get_data(as_text=True)
    body = review.get_json()
    assert body["representative_review"] == "documented_scope"
    assert body["representative_powers_verified"] is False
    assert body["legally_binding"] is False
    assert body["real_signature_collected"] is False
    forged = client.post(
        f"{DOSSIERS}/{dossier_id}/review",
        json={
            "result": "documented_scope",
            "idempotency_key": "forge",
            "creates_acceptance": True,
            "contractual_basis": "acceptance_unapproved",
        },
        headers=_owner(world),
    )
    assert forged.status_code == 403
    assert forged.get_json()["error_code"] == "acceptation_non_fabriquee"
    foreign = client.post(
        f"{DOSSIERS}/{dossier_id}/review",
        json={"result": "refused", "idempotency_key": "etranger"},
        headers=_headers(world["app"], world["other_owner"]),
    )
    assert foreign.status_code == 404

    booking.price_amount = Decimal("50.00")
    db.session.commit()
    changed = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "acceptance", "idempotency_key": "apres", "signatory_role": "patient"},
        headers=_driver(world),
    )
    assert changed.status_code == 409
    assert changed.get_json()["error_code"] == "decision_obsolete"
    stored = db.session.get(PatientBillingDossier, dossier_id)
    assert f"{Decimal(stored.amount_chf):.2f}" == "45.00"
    assert stored.contractual_basis == "representative"

    draft_booking = _booking(world, customer_name="Brouillon")
    client.post(PATIENT.format(booking_id=draft_booking.id), json={}, headers=_driver(world))
    fixture = client.post(
        PATIENT_TX.format(booking_id=draft_booking.id),
        json={
            "action": "declare_recipe_fixture",
            "idempotency_key": "fond",
            "recipe_fixture": "fictif-identifie",
        },
        headers=_driver(world),
    )
    assert fixture.status_code == 200
    draft = client.post(
        PATIENT_TX.format(booking_id=draft_booking.id),
        json={"action": "prepare_draft", "idempotency_key": "brouillon"},
        headers=_driver(world),
    )
    assert draft.status_code == 200, draft.get_data(as_text=True)
    assert draft.get_json()["draft_status"] == "draft"
    assert draft.get_json()["invoice_issued"] is False
    assert draft.get_json()["qr_specimen"]["payable"] is False
    emit = client.post(
        PATIENT_TX.format(booking_id=draft_booking.id),
        json={"action": "issue_invoice", "idempotency_key": "emettre"},
        headers=_driver(world),
    )
    assert emit.status_code == 409
    assert emit.get_json()["error_code"] == "emission_desactivee"
    image = client.post(
        PATIENT_TX.format(booking_id=draft_booking.id),
        json={
            "action": "acceptance",
            "idempotency_key": "image",
            "signature_image": "data:image/png;base64,AAAA",
        },
        headers=_driver(world),
    )
    assert image.status_code == 400
    listing = client.get(DOSSIERS, headers=_owner(world))
    assert listing.get_json()["real_signature_collection"] is False
    assert all(
        row["booking_id"] != booking.id
        for row in client.get(
            DOSSIERS, headers=_headers(world["app"], world["other_owner"])
        ).get_json()["cases"]
    )
