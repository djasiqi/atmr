"""P5-L04 — la fiche patient lit les versions et le PDF archivé.

Aucune facture. Une consultation ne régénère pas le document.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token

from application.settlement.mandate_registry import sign_demonstration
from ext import db as app_db
from models import Booking, Client, Company, Driver, Invoice, InvoicePayment, User
from models.convention_demo import PatientCompanyConventionDemo
from models.enums import BookingStatus, PricingModelType, UserRole
from models.invoice import InvoiceCreditNote, InvoiceRefund
from models.service_area_pricing import PricingProfile, PricingProfileVersion
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration

TRACE = [{"x": 8.0, "y": 8.0}, {"x": 30.0, "y": 18.0}]


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_LOCAL_RECIPE", "1")
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")


@pytest.fixture
def uploads(app, tmp_path):
    previous = app.config.get("UPLOAD_FOLDER")
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    yield tmp_path
    if previous is None:
        app.config.pop("UPLOAD_FOLDER", None)
    else:
        app.config["UPLOAD_FOLDER"] = previous


def _user(role: UserRole) -> User:
    token = uuid.uuid4().hex[:8]
    row = User()
    row.username = f"p54_{role.value.lower()}_{token}"
    row.email = f"p54_{token}@example.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("password123", force_change=False)
    app_db.session.add(row)
    app_db.session.flush()
    return row


def _world():
    owner = _user(UserRole.company)
    company = Company(name=f"P54 {uuid.uuid4().hex[:6]}", user_id=owner.id)
    app_db.session.add(company)
    app_db.session.flush()
    driver_user = _user(UserRole.driver)
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    app_db.session.add(driver)
    app_db.session.flush()
    patient_user = _user(UserRole.client)
    patient = Client(user_id=patient_user.id, company_id=company.id)
    app_db.session.add(patient)
    app_db.session.commit()
    return {"company": company, "patient": patient, "driver": driver, "owner": owner}


def _booking(world) -> Booking:
    profile = PricingProfile(
        company_id=world["company"].id,
        name="Grille 1",
        model_type=PricingModelType.FLAT,
        currency="CHF",
    )
    app_db.session.add(profile)
    app_db.session.flush()
    version = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=1,
        rules_json={"amount": "45.00"},
    )
    app_db.session.add(version)
    app_db.session.flush()
    patient_user_id = app_db.session.get(Client, world["patient"].id).user_id
    row = Booking(
        company_id=world["company"].id,
        client_id=world["patient"].id,
        user_id=patient_user_id,
        driver_id=world["driver"].id,
        customer_name="Patient fictif",
        pickup_location="Gare",
        dropoff_location="Hopital",
        scheduled_time=datetime(2026, 11, 15, 10, 0, tzinfo=UTC),
        amount=45.0,
        price_amount=Decimal("45.00"),
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="patient",
        pricing_profile_version_id=version.id,
        assistance_detail="transport",
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _mandate(world) -> PatientCompanyConventionDemo:
    row = PatientCompanyConventionDemo(
        client_id=world["patient"].id,
        company_id=world["company"].id,
        debtor_label="Patient fictif",
        tariff_version="1",
        service_scope="transport",
        duration_label="Spécimen sans durée juridique.",
        status="demo_draft",
        document_version="specimen-1",
        version_number=1,
        active=True,
        idempotency_key=f"p54-{uuid.uuid4().hex[:12]}",
        legally_effective=False,
        specimen_trace={
            "mandate": {
                "covered_tariff_versions": ["1"],
                "covered_services": ["transport"],
                "history": [{"action": "recorded", "detail": "fixture"}],
            }
        },
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _headers(client, world) -> dict[str, str]:
    claims = {
        "role": world["owner"].role.value,
        "company_id": world["company"].id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(world["owner"].public_id),
            additional_claims=claims,
        )
    return {"Authorization": f"Bearer {token}"}


def _counts() -> dict[str, int]:
    return {
        "invoices": int(Invoice.query.count() or 0),
        "payments": int(InvoicePayment.query.count() or 0),
        "credits": int(InvoiceCreditNote.query.count() or 0),
        "refunds": int(InvoiceRefund.query.count() or 0),
    }


def test_fiche_lit_historique_et_pdf_sans_facture(
    client, db, prep, uploads, monkeypatch
):
    del db
    world = _world()
    booking = _booking(world)
    mandate = _mandate(world)
    before = _counts()
    payer = booking.billed_to_type
    sign_demonstration(
        booking,
        {
            "trace": TRACE,
            "signer": "representative",
            "representative_name": "Ada Martin",
            "representative_quality": "proche",
            "representative_relation": "fille",
            "acknowledged": True,
            "tariff_version_presented": "1",
            "version_number": 1,
            "idempotency_key": "signature-fiche-01",
        },
    )
    headers = _headers(client, world)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("le PDF archivé ne doit pas être régénéré")

    monkeypatch.setattr("routes.convention_demo.specimen_pdf", _forbidden)
    monkeypatch.setattr("application.settlement.convention_demo._pdf", _forbidden)
    stored = client.get(
        f"/api/v1/companies/me/clients/{world['patient'].id}/convention-demo/specimen.pdf"
        f"?mandate={mandate.id}",
        headers=headers,
    )
    assert stored.status_code == 200, stored.get_data(as_text=True)
    assert stored.data.startswith(b"%PDF")
    listed = client.get(
        f"/api/v1/companies/me/clients/{world['patient'].id}/mandates",
        headers=headers,
    )
    assert listed.status_code == 200
    body = listed.get_json()
    assert body["billing_authorized"] is False
    current = body["mandates"][0]
    assert current["version_number"] == 1
    assert current["active"] is True
    assert current["covered_services"] == ["transport"]
    assert current["covered_tariff_versions"] == ["1"]
    assert current["demonstration_signature"]["signer"] == "representative"
    assert current["demonstration_signature"]["documentary_status"] == "archived"
    assert current["admin_review_required"] is True
    assert current["legally_effective"] is False
    opened = client.post(
        f"/api/v1/companies/me/clients/{world['patient'].id}/mandates",
        headers=headers,
        json={
            "idempotency_key": "fiche-version-02",
            "debtor_label": "Patient fictif",
            "tariff_version": "2",
            "service_scope": "brancard",
            "duration_label": "Spécimen sans durée juridique.",
            "document_version": "specimen-1",
        },
    )
    assert opened.status_code == 200, opened.get_data(as_text=True)
    assert opened.get_json()["status"] == "demo_draft"
    assert opened.get_json()["demonstration_signature"] is None
    assert opened.get_json()["legally_effective"] is False
    again = client.get(
        f"/api/v1/companies/me/clients/{world['patient'].id}/mandates",
        headers=headers,
    )
    versions = {row["version_number"]: row for row in again.get_json()["mandates"]}
    assert versions[1]["active"] is False
    assert versions[1]["status"] == "demo_specimen_signed"
    assert versions[1]["demonstration_signature"]["archive_reference"]
    assert versions[2]["active"] is True
    assert versions[2]["status"] == "demo_draft"
    assert versions[2]["demonstration_signature"] is None
    assert versions[2]["tariff_version"] == "2"
    assert versions[2]["service_scope"] == "brancard"
    kept = client.get(
        f"/api/v1/companies/me/clients/{world['patient'].id}/convention-demo/specimen.pdf"
        f"?mandate={mandate.id}",
        headers=headers,
    )
    assert kept.status_code == 200
    assert kept.data == stored.data
    relative = versions[1]["demonstration_signature"]["archive_reference"]
    (uploads / relative).unlink()
    gone = client.get(
        f"/api/v1/companies/me/clients/{world['patient'].id}/convention-demo/specimen.pdf"
        f"?mandate={mandate.id}",
        headers=headers,
    )
    assert gone.status_code == 409
    assert gone.get_json()["error_code"] == "archive_absente"
    app_db.session.expire_all()
    assert (
        app_db.session.get(PatientCompanyConventionDemo, mandate.id).status
        == "demo_specimen_signed"
    )
    missing = client.get(
        f"/api/v1/companies/me/clients/{world['patient'].id}/convention-demo/specimen.pdf"
        "?mandate=999999",
        headers=headers,
    )
    assert missing.status_code == 409
    assert missing.get_json()["error_code"] == "specimen_absent"
    other = _world()
    foreign = client.get(
        f"/api/v1/companies/me/clients/{world['patient'].id}/convention-demo/specimen.pdf"
        f"?mandate={mandate.id}",
        headers=_headers(client, other),
    )
    assert foreign.status_code == 409
    assert foreign.get_json()["error_code"] == "patient_autre_entreprise"
    leaked = client.get(
        f"/api/v1/companies/me/clients/{other['patient'].id}/convention-demo/specimen.pdf"
        f"?mandate={mandate.id}",
        headers=_headers(client, other),
    )
    assert leaked.status_code == 409
    assert leaked.get_json()["error_code"] in {
        "specimen_absent",
        "patient_autre_entreprise",
    }
    app_db.session.expire_all()
    assert app_db.session.get(Booking, booking.id).billed_to_type == payer
    assert _counts() == before
