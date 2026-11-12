"""Brouillon patient dans le moteur existant. Base jetable, émission fermée."""

from __future__ import annotations

import base64
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import func, select

from application.invoices.issued_invoice_guard import invoice_is_issued
from application.settlement.patient_billing_prep import specimen_pdf
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


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")


@pytest.fixture
def world(db, app, prep):
    current = database_name()
    if not (current or "").startswith(RECIPE):
        pytest.fail(f"Base refusée pour cette recette : {current or 'absente'}")
    suffix = uuid.uuid4().hex[:8]
    company, owner, driver, driver_user = _company_driver(suffix)
    db.session.commit()
    return {
        "app": app,
        "company": company,
        "owner": owner,
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
        name=f"Facture-{label}",
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


def _booking(world):
    from models.booking import Booking

    row = Booking(
        company_id=world["company"].id,
        executing_company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount=45.0,
        price_amount=Decimal("45.00"),
        status=BookingStatus.IN_PROGRESS,
        scheduled_time=datetime(2026, 10, 27, 9, 0, tzinfo=UTC),
        pickup_location="Départ fictif",
        dropoff_location="Arrivée fictive",
        customer_name="Patient fictif",
        billed_to_type="patient",
        is_round_trip=False,
    )
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


def test_brouillon_unique_non_emetteur_et_qr_non_payable(client, world):
    booking = _booking(world)
    opened = client.post(PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world))
    assert opened.status_code == 200
    missing = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-1"},
        headers=_driver(world),
    )
    assert missing.status_code == 409
    assert missing.get_json()["error_code"] == "fondement_insuffisant"
    accepted = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "acceptance", "idempotency_key": "accept-1", "signatory_role": "patient"},
        headers=_driver(world),
    )
    assert accepted.status_code == 200
    still = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-2"},
        headers=_driver(world),
    )
    assert still.status_code == 409
    basis = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "declare_recipe_fixture",
            "idempotency_key": "fixture-1",
            "recipe_fixture": "fictif-identifie",
        },
        headers=_driver(world),
    )
    assert basis.status_code == 200
    assert basis.get_json()["legally_binding"] is False
    first = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-3"},
        headers=_driver(world),
    )
    assert first.status_code == 200
    body = first.get_json()
    assert body["draft_status"] == "draft"
    assert body["invoice_issued"] is False
    assert body["qr_bill"] is False
    assert body["qr_specimen"]["payable"] is False
    assert body["qr_specimen"]["reference_type"] == "NON"
    assert body["qr_specimen"]["spc_payload"] is None
    assert "NE PAS PAYER" in body["qr_specimen"]["banner"]
    second = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-4"},
        headers=_driver(world),
    )
    assert second.status_code == 200
    assert second.get_json()["draft_invoice_id"] == body["draft_invoice_id"]
    replay = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "prepare_draft", "idempotency_key": "draft-3"},
        headers=_driver(world),
    )
    assert replay.status_code == 200
    assert replay.get_json()["replay"] is True
    issued = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "issue_invoice", "idempotency_key": "emit-1"},
        headers=_driver(world),
    )
    assert issued.status_code == 409
    invoice = db.session.get(Invoice, body["draft_invoice_id"])
    assert invoice.status == InvoiceStatus.DRAFT
    assert invoice_is_issued(invoice) is False
    assert invoice.billing_party_id is not None
    lines = db.session.scalars(
        select(InvoiceLine).where(InvoiceLine.reservation_id == booking.id)
    ).all()
    assert len(lines) == 1
    assert lines[0].total_with_vat == Decimal("45.00")
    assert int(db.session.scalar(select(func.count()).select_from(Invoice))) >= 1
    listing = client.get(DOSSIERS, headers=_owner(world))
    assert listing.status_code == 200
    found = [item for item in listing.get_json()["cases"] if item["booking_id"] == booking.id]
    assert found[0]["draft_invoice_id"] == body["draft_invoice_id"]


def test_representant_reste_en_revue_administrative(client, world):
    booking = _booking(world)
    client.post(PATIENT.format(booking_id=booking.id), json={}, headers=_driver(world))
    review = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "representative",
            "idempotency_key": "rep-1",
            "representative_name": "Représentant fictif",
            "representative_kind": "curateur",
            "representative_proof_expected": "justificatif",
            "signatory_role": "representative",
        },
        headers=_driver(world),
    )
    assert review.status_code == 200
    body = review.get_json()
    assert body["representative_powers_verified"] is False
    assert body["representative_decision"] == "pending_review"
    refused = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "approve_representative", "idempotency_key": "rep-2"},
        headers=_driver(world),
    )
    assert refused.status_code == 403
    pdf = base64.b64decode(specimen_pdf(body))
    assert b"NE PAS PAYER" in pdf
