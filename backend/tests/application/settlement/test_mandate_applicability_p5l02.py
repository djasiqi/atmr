"""P5-L02 — la version active couvre-t-elle le transport ?

Lecture seule. Aucune facture. ``billing_authorized`` et ``legally_effective``
restent faux.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from application.settlement.convention_demo import ConventionRefusal
from application.settlement.mandate_registry import read_applicability
from ext import db as app_db
from models import (
    Booking,
    CashCollection,
    Client,
    Company,
    Driver,
    Invoice,
    InvoicePayment,
    User,
)
from models.convention_demo import PatientCompanyConventionDemo
from models.enums import BookingStatus, PricingModelType, UserRole
from models.service_area_pricing import PricingProfile, PricingProfileVersion
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_LOCAL_RECIPE", "1")
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.fail(f"Base refusée pour cette recette : {current or 'absente'}")


def _user(role: UserRole) -> User:
    token = uuid.uuid4().hex[:8]
    row = User()
    row.username = f"p52_{role.value.lower()}_{token}"
    row.email = f"p52_{token}@example.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("password123", force_change=False)
    app_db.session.add(row)
    app_db.session.flush()
    return row


def _world(db):
    del db
    owner = _user(UserRole.company)
    company = Company(name=f"P52 {uuid.uuid4().hex[:6]}", user_id=owner.id)
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


def _booking(world, version: PricingProfileVersion, *, service: str | None, amount: str = "45.00"):
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
        amount=float(amount),
        price_amount=Decimal(amount),
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="patient",
        pricing_profile_version_id=version.id,
        assistance_detail=service,
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _mandate(world, *, tariffs: list[str], services: list[str], version_number: int = 1, active: bool = True, review: bool = False):
    trace = {
        "mandate": {
            "covered_tariff_versions": tariffs,
            "covered_services": services,
            "history": [{"action": "recorded", "detail": "fixture"}],
        }
    }
    if review:
        trace["mandate"]["signature"] = {
            "signer": "representative",
            "admin_review_required": True,
            "legally_effective": False,
        }
    row = PatientCompanyConventionDemo(
        client_id=world["patient"].id,
        company_id=world["company"].id,
        debtor_label="Patient fictif",
        representative_name="Représentant fictif" if review else None,
        tariff_version=tariffs[0],
        service_scope="transport",
        duration_label="Spécimen sans durée juridique.",
        status="demo_draft",
        document_version="specimen-1",
        version_number=version_number,
        active=active,
        idempotency_key=f"p52-{uuid.uuid4().hex[:12]}",
        legally_effective=False,
        specimen_trace=trace,
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _counts() -> dict[str, int]:
    return {
        "invoices": int(Invoice.query.count() or 0),
        "payments": int(InvoicePayment.query.count() or 0),
        "cash": int(CashCollection.query.count() or 0),
    }


def test_deux_transports_meme_version_et_tarif_hors_perimetre(db, prep):
    world = _world(db)
    planned = _tariff(world, 1)
    other = _tariff(world, 2)
    mandate = _mandate(world, tariffs=["1"], services=["transport"])
    first = _booking(world, planned, service="transport", amount="45.00")
    second = _booking(world, planned, service="transport", amount="45.00")
    foreign_tariff = _booking(world, other, service="transport", amount="80.00")
    before = _counts()
    stamp = (
        mandate.tariff_version,
        mandate.specimen_trace,
        mandate.active,
        first.price_amount,
        first.billed_to_type,
        first.invoice_line_id,
    )
    covered = read_applicability(world["company"].id, first.id)
    again = read_applicability(world["company"].id, first.id)
    reused = read_applicability(world["company"].id, second.id)
    gap = read_applicability(world["company"].id, foreign_tariff.id)
    assert covered["code"] == "applicable"
    assert covered["mandate_id"] == mandate.id
    assert covered["version_number"] == 1
    assert covered["active"] is True
    assert covered["amount"] == "45.00"
    assert covered["billing_authorized"] is False
    assert covered["legally_effective"] is False
    assert again == covered
    assert reused["mandate_id"] == mandate.id
    assert reused["version_number"] == 1
    assert gap["code"] == "avenant_requis"
    assert gap["avenant_motif"] == "tarif"
    assert gap["mandate_id"] == mandate.id
    assert gap["billing_authorized"] is False
    app_db.session.expire_all()
    stored = app_db.session.get(PatientCompanyConventionDemo, mandate.id)
    booking = app_db.session.get(Booking, first.id)
    assert (
        stored.tariff_version,
        stored.specimen_trace,
        stored.active,
        booking.price_amount,
        booking.billed_to_type,
        booking.invoice_line_id,
    ) == stamp
    assert _counts() == before


def test_prestation_representant_et_absences(db, prep):
    world = _world(db)
    planned = _tariff(world, 1)
    mandate = _mandate(world, tariffs=["1"], services=["transport"], review=True)
    uncovered = _booking(world, planned, service="brancard")
    body = read_applicability(world["company"].id, uncovered.id)
    assert body["code"] == "avenant_requis"
    assert body["avenant_motif"] == "prestation"
    assert body["mandate_id"] == mandate.id
    plain = _booking(world, planned, service="transport")
    reviewed = read_applicability(world["company"].id, plain.id)
    assert reviewed["code"] == "applicable"
    assert reviewed["admin_review_required"] is True
    assert reviewed["billing_authorized"] is False
    assert reviewed["legally_effective"] is False

    other_user = _user(UserRole.client)
    other = Client(user_id=other_user.id, company_id=world["company"].id)
    app_db.session.add(other)
    app_db.session.commit()
    outsider = Booking(
        company_id=world["company"].id,
        client_id=other.id,
        user_id=other_user.id,
        driver_id=world["driver"].id,
        customer_name="Autre patient",
        pickup_location="Gare",
        dropoff_location="Hopital",
        scheduled_time=datetime(2026, 11, 16, 10, 0, tzinfo=UTC),
        amount=45.0,
        price_amount=Decimal("45.00"),
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="patient",
        pricing_profile_version_id=planned.id,
    )
    app_db.session.add(outsider)
    app_db.session.commit()
    rejected = read_applicability(world["company"].id, outsider.id)
    assert rejected["code"] == "autre_patient"
    assert rejected["mandate_id"] is None

    empty = _world(db)
    lonely_tariff = _tariff(empty, 1)
    lonely = _booking(empty, lonely_tariff, service="transport")
    missing = read_applicability(empty["company"].id, lonely.id)
    assert missing["code"] == "mandat_absent"
    assert missing["mandate_id"] is None
    assert missing["billing_authorized"] is False

    with pytest.raises(ConventionRefusal) as foreign:
        read_applicability(empty["company"].id, first_booking_of(world))
    assert foreign.value.code == "reservation_absente"


def first_booking_of(world) -> int:
    found = (
        Booking.query.filter_by(company_id=world["company"].id)
        .order_by(Booking.id)
        .first()
    )
    assert found is not None
    return int(found.id)


def test_version_inactive_ignoree(db, prep):
    world = _world(db)
    old = _tariff(world, 1)
    current = _tariff(world, 2)
    previous = _mandate(world, tariffs=["1"], services=["transport"], version_number=1, active=False)
    active = _mandate(world, tariffs=["2"], services=["transport"], version_number=2, active=True)
    stale = _booking(world, old, service="transport")
    fresh = _booking(world, current, service="transport")
    refused = read_applicability(world["company"].id, stale.id)
    used = read_applicability(world["company"].id, fresh.id)
    assert refused["code"] == "avenant_requis"
    assert refused["mandate_id"] == active.id
    assert refused["version_number"] == 2
    assert used["code"] == "applicable"
    assert used["mandate_id"] == active.id
    assert used["version_number"] == 2
    assert previous.active is False
    app_db.session.expire_all()
    assert app_db.session.get(PatientCompanyConventionDemo, previous.id).active is False
    assert app_db.session.get(PatientCompanyConventionDemo, previous.id).tariff_version == "1"
