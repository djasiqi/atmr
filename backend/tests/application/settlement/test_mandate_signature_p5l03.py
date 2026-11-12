"""P5-L03 — signature tactile d'une version et archive PDF locale.

La signature ne rend pas le mandat opposable et n'émet aucune facture.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pypdf import PdfReader

from application.settlement.convention_demo import (
    ConventionRefusal,
    archived_pdf_for,
    tactile_reference,
)
from application.settlement.mandate_registry import (
    open_version,
    read_applicability,
    sign_demonstration,
)
from ext import db as app_db
from models import (
    Booking,
    Client,
    Company,
    Driver,
    Invoice,
    InvoicePayment,
    User,
)
from models.convention_demo import PatientCompanyConventionDemo
from models.enums import BookingStatus, PricingModelType, UserRole
from models.invoice import InvoiceCreditNote, InvoiceRefund
from models.service_area_pricing import PricingProfile, PricingProfileVersion
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration

TRACE = [{"x": 10.0, "y": 12.0}, {"x": 48.0, "y": 30.0}, {"x": 80.0, "y": 16.0}]


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
    row.username = f"p53_{role.value.lower()}_{token}"
    row.email = f"p53_{token}@example.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("password123", force_change=False)
    app_db.session.add(row)
    app_db.session.flush()
    return row


def _world():
    owner = _user(UserRole.company)
    company = Company(name=f"P53 {uuid.uuid4().hex[:6]}", user_id=owner.id)
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
    return {"company": company, "patient": patient, "driver": driver}


def _tariff(world, number: int) -> PricingProfileVersion:
    profile = PricingProfile(
        company_id=world["company"].id,
        name=f"Grille {number}",
        model_type=PricingModelType.FLAT,
        currency="CHF",
    )
    app_db.session.add(profile)
    app_db.session.flush()
    version = PricingProfileVersion(
        pricing_profile_id=profile.id,
        version=number,
        rules_json={"amount": "45.00"},
    )
    app_db.session.add(version)
    app_db.session.commit()
    return version


def _booking(world, version: PricingProfileVersion) -> Booking:
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
        idempotency_key=f"p53-{uuid.uuid4().hex[:12]}",
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


def _body(
    version: int, *, signer: str = "patient", key: str = "signature-patient-01"
) -> dict:
    body = {
        "trace": TRACE,
        "signer": signer,
        "acknowledged": True,
        "tariff_version_presented": "1",
        "version_number": version,
        "idempotency_key": key,
    }
    if signer == "representative":
        body["representative_name"] = "Ada Martin"
        body["representative_quality"] = "proche"
        body["representative_relation"] = "fille"
    return body


def _counts() -> dict[str, int]:
    return {
        "invoices": int(Invoice.query.count() or 0),
        "payments": int(InvoicePayment.query.count() or 0),
        "credits": int(InvoiceCreditNote.query.count() or 0),
        "refunds": int(InvoiceRefund.query.count() or 0),
    }


def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _signature(row: PatientCompanyConventionDemo) -> dict:
    trace = row.specimen_trace if isinstance(row.specimen_trace, dict) else {}
    mandate = trace.get("mandate") if isinstance(trace.get("mandate"), dict) else {}
    found = mandate.get("signature")
    return found if isinstance(found, dict) else {}


def test_patient_archive_rejeu_et_version_suivante(db, prep, uploads):
    del db
    world = _world()
    booking = _booking(world, _tariff(world, 1))
    mandate = _mandate(world)
    before = _counts()
    price = booking.price_amount
    signed = sign_demonstration(booking, _body(1))
    reference = signed["demonstration_signature"]["archive_reference"]
    path = uploads / reference
    assert signed["replay"] is False
    assert signed["status"] == "demo_specimen_signed"
    assert signed["version_number"] == 1
    assert signed["legally_effective"] is False
    assert signed["billing_authorized"] is False
    assert signed["invoice_issued"] is False
    assert signed["demonstration_signature"]["signer"] == "patient"
    assert signed["demonstration_signature"]["documentary_status"] == "archived"
    assert signed["demonstration_signature"]["tactile_reference"] == tactile_reference(
        TRACE
    )
    assert path.is_file()
    text = _pdf_text(path)
    assert "SANS VALEUR CONTRACTUELLE" in text
    assert "non juridiquement opposable" in text
    assert "Version du mandat : 1" in text
    assert "transport" in text
    assert tactile_reference(TRACE) in text
    assert archived_pdf_for(world["company"].id, mandate.id) == path.read_bytes()
    first_hash = path.read_bytes()
    replay = sign_demonstration(booking, _body(1, key="signature-patient-02"))
    assert replay["replay"] is True
    assert path.read_bytes() == first_hash
    assert list(uploads.rglob("*.pdf")) == [path]
    path.unlink()
    restored = sign_demonstration(booking, _body(1, key="signature-patient-03"))
    assert restored["replay"] is True
    assert restored["demonstration_signature"]["archive_reference"] == reference
    assert path.is_file()
    assert tactile_reference(TRACE) in _pdf_text(path)
    kept = path.read_bytes()
    opened = open_version(
        world["company"].id,
        world["patient"].id,
        {
            "idempotency_key": "version-suivante-02",
            "debtor_label": "Patient fictif",
            "tariff_version": "2",
            "service_scope": "brancard",
            "duration_label": "Spécimen sans durée juridique.",
            "document_version": "specimen-1",
        },
    )
    assert opened["version_number"] == 2
    assert opened["status"] == "demo_draft"
    assert opened["demonstration_signature"] is None
    assert path.read_bytes() == kept
    app_db.session.expire_all()
    previous = app_db.session.get(PatientCompanyConventionDemo, mandate.id)
    assert previous.status == "demo_specimen_signed"
    assert previous.active is False
    assert previous.legally_effective is False
    assert _signature(previous)["version_number"] == 1
    coverage = read_applicability(world["company"].id, booking.id)
    assert coverage["code"] == "avenant_requis"
    assert coverage["version_number"] == 2
    assert coverage["billing_authorized"] is False
    assert coverage["legally_effective"] is False
    with pytest.raises(ConventionRefusal) as refused:
        sign_demonstration(booking, _body(1, key="signature-ancienne-04"))
    assert refused.value.code == "version_inactive"
    current = (
        app_db.session.query(PatientCompanyConventionDemo)
        .filter_by(
            company_id=world["company"].id,
            client_id=world["patient"].id,
            version_number=2,
        )
        .one()
    )
    assert current.status == "demo_draft"
    assert _signature(current) == {}
    assert booking.price_amount == price
    assert _counts() == before


def test_representant_et_isolation(db, prep, uploads):
    del db
    world = _world()
    booking = _booking(world, _tariff(world, 1))
    mandate = _mandate(world)
    signed = sign_demonstration(
        booking, _body(1, signer="representative", key="signature-representant")
    )
    signature = signed["demonstration_signature"]
    assert signature["signer"] == "representative"
    assert signature["representative_name"] == "Ada Martin"
    assert signature["representative_quality"] == "proche"
    assert signature["representation_review_required"] is True
    assert signed["admin_review_required"] is True
    assert signed["legally_effective"] is False
    assert signed["billing_authorized"] is False
    text = _pdf_text(uploads / signature["archive_reference"])
    assert "Ada Martin" in text or "proche" in text
    assert "Vérification administrative : à effectuer." in text
    assert "non juridiquement opposable" in text
    other = _world()
    foreign = _booking(other, _tariff(other, 1))
    with pytest.raises(ConventionRefusal) as denied:
        sign_demonstration(foreign, _body(1, key="signature-intrus"))
    assert denied.value.code == "specimen_sans_brouillon"
    with pytest.raises(ConventionRefusal) as hidden:
        archived_pdf_for(other["company"].id, mandate.id)
    assert hidden.value.code == "specimen_absent"
    app_db.session.expire_all()
    stored = app_db.session.get(PatientCompanyConventionDemo, mandate.id)
    assert stored.status == "demo_specimen_signed"
    assert stored.company_id == world["company"].id


def test_echec_documentaire_sans_archive_fausse(db, prep, uploads, monkeypatch):
    del db
    world = _world()
    booking = _booking(world, _tariff(world, 1))
    mandate = _mandate(world)
    with pytest.raises(ConventionRefusal) as changed:
        sign_demonstration(booking, _body(9, key="signature-version-inconnue"))
    assert changed.value.code == "version_changee"
    app_db.session.expire_all()
    assert (
        app_db.session.get(PatientCompanyConventionDemo, mandate.id).status
        == "demo_draft"
    )

    def _broken(*_args, **_kwargs):
        raise OSError("disque indisponible")

    monkeypatch.setattr(
        "application.settlement.mandate_registry.store_mandate_pdf",
        _broken,
    )
    with pytest.raises(ConventionRefusal) as failed:
        sign_demonstration(booking, _body(1, key="signature-echec-disque"))
    assert failed.value.code == "archive_indisponible"
    app_db.session.expire_all()
    stored = app_db.session.get(PatientCompanyConventionDemo, mandate.id)
    assert stored.status == "demo_draft"
    assert stored.legally_effective is False
    assert _signature(stored) == {}
    assert list(uploads.rglob("*.pdf")) == []
