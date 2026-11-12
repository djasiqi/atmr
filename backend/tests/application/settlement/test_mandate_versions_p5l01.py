"""P5-L01 — versions d'un mandat de démonstration.

Une seule version active. Les précédentes restent. Aucune facture.
``legally_effective`` reste faux. Le client 39846 n'est pas recopié ici.
"""

from __future__ import annotations

import uuid
from threading import Thread

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from application.settlement.convention_demo import ConventionRefusal
from application.settlement.mandate_registry import (
    list_for_client,
    open_version,
    refuse_historical_change,
)
from ext import db as app_db
from models import CashCollection, Client, Company, Invoice, InvoicePayment, User
from models.convention_demo import PatientCompanyConventionDemo
from models.enums import UserRole
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_LOCAL_RECIPE", "1")
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")


def _user(role: UserRole) -> User:
    token = uuid.uuid4().hex[:8]
    row = User()
    row.username = f"p5_{role.value.lower()}_{token}"
    row.email = f"p5_{token}@example.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("password123", force_change=False)
    app_db.session.add(row)
    app_db.session.flush()
    return row


def _world(db):
    del db
    owner = _user(UserRole.company)
    company = Company(name=f"P5 {uuid.uuid4().hex[:6]}", user_id=owner.id)
    app_db.session.add(company)
    app_db.session.flush()
    patient_user = _user(UserRole.client)
    patient = Client(user_id=patient_user.id, company_id=company.id)
    app_db.session.add(patient)
    app_db.session.commit()
    assert int(patient.id) != 39846
    return {"company": company, "patient": patient}


def _historical(
    world, *, tariff: str = "demo-2026.1", status: str = "demo_draft"
) -> PatientCompanyConventionDemo:
    row = PatientCompanyConventionDemo(
        client_id=world["patient"].id,
        company_id=world["company"].id,
        debtor_label="Patient fictif",
        tariff_version=tariff,
        service_scope="Transports simulés de cette entreprise uniquement.",
        duration_label="Spécimen sans durée juridique.",
        status=status,
        document_version="specimen-1",
        version_number=1,
        active=True,
        idempotency_key=f"legacy-{uuid.uuid4().hex[:12]}",
        legally_effective=False,
        specimen_trace={
            "mandate": {"history": [{"action": "recorded", "detail": "copie"}]}
        },
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _payload(key: str, tariff: str) -> dict[str, str]:
    return {
        "idempotency_key": key,
        "debtor_label": "Patient fictif",
        "tariff_version": tariff,
        "service_scope": f"Périmètre {tariff}",
        "duration_label": "Durée de démonstration",
        "document_version": "specimen-2",
    }


def _counts() -> dict[str, int]:
    return {
        "invoices": int(Invoice.query.count() or 0),
        "payments": int(InvoicePayment.query.count() or 0),
        "cash": int(CashCollection.query.count() or 0),
    }


def test_version_active_et_historique_conserve(db, prep):
    world = _world(db)
    origin = _historical(world)
    kept = (
        int(origin.id),
        origin.tariff_version,
        origin.status,
        origin.document_version,
        origin.specimen_trace,
        origin.legally_effective,
    )
    before = _counts()
    opened = open_version(
        world["company"].id,
        world["patient"].id,
        _payload("version-deux", "demo-2026.2"),
    )
    assert opened["replay"] is False
    assert opened["version_number"] == 2
    assert opened["active"] is True
    assert opened["legally_effective"] is False
    assert opened["tariff_version"] == "demo-2026.2"
    app_db.session.expire_all()
    stored = app_db.session.get(PatientCompanyConventionDemo, origin.id)
    assert stored.active is False
    assert (
        int(stored.id),
        stored.tariff_version,
        stored.status,
        stored.document_version,
        stored.specimen_trace,
        stored.legally_effective,
    ) == kept
    active = PatientCompanyConventionDemo.query.filter_by(
        client_id=world["patient"].id, active=True
    ).all()
    assert len(active) == 1
    assert active[0].tariff_version == "demo-2026.2"
    previous = PatientCompanyConventionDemo.query.filter_by(
        client_id=world["patient"].id, version_number=1
    ).one()
    assert previous.tariff_version == "demo-2026.1"
    assert _counts() == before
    listed = list_for_client(world["company"].id, world["patient"].id)
    assert len(listed["mandates"]) == 2
    assert listed["billing_authorized"] is False


def test_rejeu_conflit_et_version_figee(db, prep):
    world = _world(db)
    signed = _historical(world, status="demo_specimen_signed")
    payload = _payload("version-deux", "demo-2026.2")
    first = open_version(world["company"].id, world["patient"].id, payload)
    again = open_version(world["company"].id, world["patient"].id, payload)
    assert again["replay"] is True
    assert again["id"] == first["id"]
    assert (
        PatientCompanyConventionDemo.query.filter_by(
            client_id=world["patient"].id
        ).count()
        == 2
    )
    changed = dict(payload)
    changed["tariff_version"] = "demo-2026.9"
    with pytest.raises(ConventionRefusal) as conflict:
        open_version(world["company"].id, world["patient"].id, changed)
    assert conflict.value.code == "conflit_idempotence"
    assert (
        PatientCompanyConventionDemo.query.filter_by(
            client_id=world["patient"].id
        ).count()
        == 2
    )
    with pytest.raises(ConventionRefusal) as frozen:
        refuse_historical_change(
            world["company"].id,
            signed.id,
            {"tariff_version": "demo-autre"},
        )
    assert frozen.value.code == "version_figee"
    app_db.session.expire_all()
    assert (
        app_db.session.get(PatientCompanyConventionDemo, signed.id).tariff_version
        == "demo-2026.1"
    )
    assert (
        app_db.session.get(PatientCompanyConventionDemo, signed.id).status
        == "demo_specimen_signed"
    )


def test_isolation_et_couple_distinct(db, prep):
    first = _world(db)
    second = _world(db)
    _historical(first, tariff="tarif-a")
    _historical(second, tariff="tarif-b")
    open_version(
        first["company"].id, first["patient"].id, _payload("suite-alpha", "tarif-a2")
    )
    with pytest.raises(ConventionRefusal) as foreign:
        open_version(
            second["company"].id, first["patient"].id, _payload("intrus-b", "tarif-x")
        )
    assert foreign.value.code == "patient_autre_entreprise"
    assert (
        PatientCompanyConventionDemo.query.filter_by(
            client_id=second["patient"].id, active=True
        )
        .one()
        .tariff_version
        == "tarif-b"
    )
    assert (
        PatientCompanyConventionDemo.query.filter_by(
            client_id=first["patient"].id, active=True
        )
        .one()
        .tariff_version
        == "tarif-a2"
    )


def test_double_activation_et_contrainte_juridique(db, prep):
    world = _world(db)
    origin = _historical(world)
    opened = open_version(
        world["company"].id,
        world["patient"].id,
        _payload("version-deux", "demo-2026.2"),
    )
    extra = PatientCompanyConventionDemo(
        client_id=world["patient"].id,
        company_id=world["company"].id,
        debtor_label="Patient fictif",
        tariff_version="demo-2026.3",
        service_scope="Périmètre interdit",
        duration_label="Durée de démonstration",
        status="demo_draft",
        document_version="specimen-3",
        version_number=3,
        active=True,
        idempotency_key="version-trois-sql",
        legally_effective=False,
    )
    app_db.session.add(extra)
    with pytest.raises(IntegrityError):
        app_db.session.commit()
    app_db.session.rollback()
    definition = app_db.session.execute(
        text(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conname = 'ck_convention_demo_not_legal'"
        )
    ).scalar()
    assert definition is not None
    assert "false" in str(definition).lower()
    index = app_db.session.execute(
        text(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'uq_convention_demo_active'"
        )
    ).scalar()
    assert index is not None
    assert "UNIQUE" in str(index).upper()
    assert "active" in str(index).lower()
    stored = app_db.session.get(PatientCompanyConventionDemo, origin.id)
    stored.legally_effective = True
    with pytest.raises(IntegrityError):
        app_db.session.commit()
    app_db.session.rollback()
    assert (
        PatientCompanyConventionDemo.query.filter_by(
            client_id=world["patient"].id, active=True
        )
        .one()
        .id
        == opened["id"]
    )
    assert (
        app_db.session.get(PatientCompanyConventionDemo, origin.id).legally_effective
        is False
    )


def test_activations_concurrentes(db, prep, app):
    world = _world(db)
    _historical(world)
    company_id = int(world["company"].id)
    patient_id = int(world["patient"].id)
    codes: list[str] = []

    def _run(key: str, tariff: str) -> None:
        with app.app_context():
            try:
                opened = open_version(company_id, patient_id, _payload(key, tariff))
                codes.append("replay" if opened["replay"] else "ok")
            except ConventionRefusal as exc:
                codes.append(exc.code)
            except Exception as exc:
                codes.append(f"{type(exc).__name__}:{exc}")
            finally:
                app_db.session.remove()

    threads = [
        Thread(target=_run, args=("course-a-aaaa", "demo-a")),
        Thread(target=_run, args=("course-b-bbbb", "demo-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert sorted(codes) == ["ok", "ok"], codes
    rows = PatientCompanyConventionDemo.query.filter_by(
        client_id=world["patient"].id
    ).all()
    assert len(rows) == 3
    assert len([row for row in rows if row.active]) == 1
    assert {row.legally_effective for row in rows} == {False}
