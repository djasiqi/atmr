"""Certification locale du brouillon chauffeur et de son isolation.

Base jetable uniquement. Aucune émission, aucun paiement.
"""

from __future__ import annotations

import base64
import threading
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import func, select

from application.invoices.active_invoice_claim import booking_has_active_invoice_claim
from application.invoices.issued_invoice_guard import invoice_is_issued
from application.settlement.patient_billing_prep import specimen_pdf
from domain.end_of_trip_settlement.production_prep_barrier import prep_open
from ext import db
from models.enums import BookingStatus, InvoiceStatus, UserRole
from models.invoice import Invoice, InvoiceLine
from models.user import User
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration

RECIPE = "lirie_e2e_"
PATIENT = "/api/v1/driver/me/bookings/{booking_id}/patient-billing-dossier"
PATIENT_TX = PATIENT + "/transitions"
DOSSIERS = "/api/v1/companies/me/patient-billing-dossiers"
INVOICES = "/api/v1/invoices/companies/{company_id}/invoices"


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")


@pytest.fixture
def world(db, app, prep):
    current = database_name()
    if not (current or "").startswith(RECIPE):
        pytest.fail(f"Base refusée pour cette recette : {current or 'absente'}")
    suffix = uuid.uuid4().hex[:8]
    company, owner, driver, driver_user = _company_driver(f"a-{suffix}")
    other, other_owner, _driver, _user = _company_driver(f"b-{suffix}")
    db.session.commit()
    return {
        "app": app,
        "company": company,
        "owner": owner,
        "other": other,
        "other_owner": other_owner,
        "driver": driver,
        "driver_user": driver_user,
    }


def _user(role: UserRole, label: str) -> User:
    row = User()
    row.username = label
    row.email = f"{label}@lirie-e2e.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("recette-isolee", force_change=False)
    db.session.add(row)
    db.session.flush()
    return row


def _company_driver(label: str):
    from models import Company, Driver

    owner = _user(UserRole.company, f"co-{label}")
    company = Company(
        name=f"Certif-{label}",
        user_id=owner.id,
        platform_billing_access_state="active",
    )
    db.session.add(company)
    db.session.flush()
    driver_user = _user(UserRole.driver, f"ch-{label}")
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    db.session.add(driver)
    db.session.flush()
    return company, owner, driver, driver_user


def _booking(world, **overrides):
    from models.booking import Booking

    row = Booking(
        company_id=world["company"].id,
        executing_company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount=45.0,
        price_amount=Decimal("45.00"),
        status=overrides.pop("status", BookingStatus.IN_PROGRESS),
        scheduled_time=datetime(2026, 10, 27, 9, 0, tzinfo=UTC),
        pickup_location="Départ fictif",
        dropoff_location="Arrivée fictive",
        customer_name="Patient fictif",
        billed_to_type=overrides.pop("billed_to_type", "patient"),
        is_round_trip=False,
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    db.session.add(row)
    db.session.commit()
    return row


def _headers(app, user: User) -> dict[str, str]:
    claims = {
        "role": user.role.value,
        "company_id": None,
        "aud": "atmr-api",
        "token_version": int(getattr(user, "token_version", 0) or 0),
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def _driver(world):
    return _headers(world["app"], world["driver_user"])


def _owner(world):
    return _headers(world["app"], world["owner"])


def _ready(client, world, booking, key: str = "fixture-1"):
    opened = client.post(
        PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world)
    )
    assert opened.status_code == 200
    basis = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "declare_recipe_fixture",
            "idempotency_key": key,
            "recipe_fixture": "fictif-identifie",
        },
        headers=_driver(world),
    )
    assert basis.status_code == 200
    assert basis.get_json()["legally_binding"] is False


def _lines(booking_id: int) -> list[InvoiceLine]:
    return list(
        db.session.scalars(
            select(InvoiceLine).where(InvoiceLine.reservation_id == booking_id)
        ).all()
    )


def test_marqueur_inutilisable_hors_recette(client, world, monkeypatch):
    booking = _booking(world)
    client.post(PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world))
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "0")
    refused = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "declare_recipe_fixture",
            "idempotency_key": "hors-recette",
            "recipe_fixture": "fictif-identifie",
        },
        headers=_driver(world),
    )
    assert refused.status_code == 404
    assert _lines(booking.id) == []
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setenv("FLASK_CONFIG", "production")
    assert prep_open() is False


def test_payeur_ou_annulation_bloquent_le_brouillon(client, world):
    changed = _booking(world)
    _ready(client, world, changed, key="fixture-change")
    changed.price_amount = Decimal("90.00")
    db.session.commit()
    blocked = client.post(
        PATIENT_TX.format(booking_id=changed.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-change"},
        headers=_driver(world),
    )
    assert blocked.status_code == 409
    assert blocked.get_json()["error_code"] == "decision_obsolete"
    assert _lines(changed.id) == []

    cancelled = _booking(world, status=BookingStatus.CANCELED)
    opened = client.post(
        PATIENT.format(booking_id=cancelled.id), json={}, headers=_driver(world)
    )
    assert opened.status_code == 409
    assert opened.get_json()["error_code"] == "transport_annule"


def test_portail_retrouve_le_brouillon_et_empeche_le_doublon(client, world):
    booking = _booking(world)
    _ready(client, world, booking)
    created = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-portail"},
        headers=_driver(world),
    )
    assert created.status_code == 200
    body = created.get_json()
    invoice = db.session.get(Invoice, body["draft_invoice_id"])
    assert invoice is not None
    assert invoice.status == InvoiceStatus.DRAFT
    assert invoice_is_issued(invoice) is False
    assert invoice.billing_party_id is not None
    assert invoice.company_id == world["company"].id
    assert invoice.vat_total_amount == Decimal("0.00")
    assert invoice.vat_breakdown["note"] == "tva_non_configuree"
    assert invoice.total_amount == Decimal("45.00")
    lines = _lines(booking.id)
    assert len(lines) == 1
    assert lines[0].qty == Decimal("1.00")
    db.session.refresh(booking)
    assert booking.invoice_line_id == lines[0].id
    assert booking_has_active_invoice_claim(int(booking.id)) is True

    listing = client.get(DOSSIERS, headers=_owner(world))
    assert listing.status_code == 200
    found = [
        item
        for item in listing.get_json()["cases"]
        if item["booking_id"] == booking.id
    ]
    assert found[0]["draft_invoice_id"] == body["draft_invoice_id"]
    assert found[0]["draft_status"] == "draft"

    invoices = client.get(
        INVOICES.format(company_id=world["company"].id) + "?status=draft",
        headers=_owner(world),
    )
    assert invoices.status_code == 200
    payload = invoices.get_json()
    rows = payload.get("data") or payload.get("items") or payload.get("invoices") or []
    if isinstance(payload.get("data"), dict):
        rows = payload["data"].get("items") or payload["data"].get("results") or rows
    identifiers = {row.get("id") for row in rows if isinstance(row, dict)}
    assert body["draft_invoice_id"] in identifiers

    foreign = client.get(
        INVOICES.format(company_id=world["company"].id),
        headers=_headers(world["app"], world["other_owner"]),
    )
    assert foreign.status_code in {403, 404}
    hidden = client.get(DOSSIERS, headers=_headers(world["app"], world["other_owner"]))
    assert hidden.status_code == 200
    assert all(item["booking_id"] != booking.id for item in hidden.get_json()["cases"])

    generate = client.post(
        INVOICES.format(company_id=world["company"].id) + "/generate",
        json={
            "client_id": invoice.client_id,
            "period_year": 2026,
            "period_month": 10,
            "reservation_ids": [booking.id],
        },
        headers=_owner(world),
    )
    assert generate.status_code != 500
    assert len(_lines(booking.id)) == 1
    pdf = base64.b64decode(specimen_pdf(body))
    assert b"NE PAS PAYER" in pdf
    assert body["qr_specimen"]["payable"] is False
    assert body["qr_specimen"]["spc_payload"] is None


def test_meme_cle_contenu_different_est_un_conflit(client, world):
    booking = _booking(world)
    _ready(client, world, booking, key="fixture-conflit")
    first = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-conflit"},
        headers=_driver(world),
    )
    assert first.status_code == 200
    conflict = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "prepare_draft",
            "idempotency_key": "draft-conflit",
            "signatory_role": "patient",
        },
        headers=_driver(world),
    )
    assert conflict.status_code == 409
    assert conflict.get_json()["error_code"] == "conflit_idempotence"
    assert len(_lines(booking.id)) == 1


def test_deux_preparations_simultanees_une_seule_facture(app, world):
    booking = _booking(world)
    client = app.test_client()
    _ready(client, world, booking, key="fixture-race")
    booking_id = int(booking.id)
    headers = _driver(world)
    barrier = threading.Barrier(2)
    results: list[tuple[int, dict]] = []
    lock = threading.Lock()

    def attempt(key: str) -> None:
        local = app.test_client()
        try:
            barrier.wait(timeout=8)
            response = local.post(
                PATIENT_TX.format(booking_id=booking_id),
                json={"action": "prepare_draft", "idempotency_key": key},
                headers=headers,
            )
            payload = response.get_json(silent=True) or {}
            with lock:
                results.append((response.status_code, payload))
        except Exception as exc:  # noqa: BLE001
            with lock:
                results.append((0, {"error": f"{type(exc).__name__}:{exc}"}))

    threads = [
        threading.Thread(target=attempt, args=("course-a",)),
        threading.Thread(target=attempt, args=("course-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert len(results) == 2
    assert all(code not in {500, 0} for code, _payload in results)
    db.session.remove()
    lines = _lines(booking_id)
    assert len(lines) == 1
    invoices = db.session.scalar(
        select(func.count())
        .select_from(Invoice)
        .where(Invoice.id == lines[0].invoice_id)
    )
    assert int(invoices) == 1
