"""Tests du seed documentation Institution (Lot 2)."""

from __future__ import annotations

import pytest

from models import (
    Institution,
    InstitutionNotification,
    InstitutionPatient,
    TransportRequest,
    User,
)
from models.enums import CarrierSource, InstitutionRole, MissionType, RequestStatus
from services.docs.institution_docs_seed import (
    DOCS_EMAIL_DOMAIN,
    DOCS_INSTITUTION_EMAIL,
    DOCS_INSTITUTION_NAME,
    DOCS_USER_EMAIL,
    assert_institution_docs_seed_environment,
    reset_and_seed_institution_docs,
)
from tests.helpers.institution_auth import institution_bearer_headers


@pytest.fixture
def docs_password(monkeypatch):
    monkeypatch.setenv("INSTITUTION_DOCS_PASSWORD", "test-docs-password-for-pytest")
    return "test-docs-password-for-pytest"


def _tenant_counts():
    institution = Institution.query.filter_by(contact_email=DOCS_INSTITUTION_EMAIL).one()
    return {
        "institutions": Institution.query.filter_by(
            contact_email=DOCS_INSTITUTION_EMAIL
        ).count(),
        "users": User.query.filter_by(email=DOCS_USER_EMAIL).count(),
        "patients": InstitutionPatient.query.filter_by(
            institution_id=institution.id
        ).count(),
        "requests": TransportRequest.query.filter_by(
            institution_id=institution.id
        ).count(),
        "notifications": InstitutionNotification.query.filter_by(
            institution_id=institution.id
        ).count(),
    }


def test_seed_refuse_environnement_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ALLOW_INSTITUTION_DOCS_SEED", raising=False)
    monkeypatch.delenv("ALLOW_NON_DEMO_SEED", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:x@db/atmr_prod")
    with pytest.raises(RuntimeError, match="production/staging"):
        assert_institution_docs_seed_environment()


def test_seed_refuse_base_staging_sans_override(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("ALLOW_INSTITUTION_DOCS_SEED", raising=False)
    monkeypatch.delenv("ALLOW_NON_DEMO_SEED", raising=False)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:x@db/lirie_staging")
    monkeypatch.delenv("SQLALCHEMY_DATABASE_URI", raising=False)
    with pytest.raises(RuntimeError, match="non autorisée"):
        assert_institution_docs_seed_environment()


def test_seed_refuse_sans_mot_de_passe(monkeypatch, db):
    monkeypatch.delenv("INSTITUTION_DOCS_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="INSTITUTION_DOCS_PASSWORD"):
        reset_and_seed_institution_docs(commit=False)


def test_seed_cree_tenant_docs_deterministe(docs_password, db, client):
    summary = reset_and_seed_institution_docs(commit=False)

    institutions = Institution.query.filter_by(contact_email=DOCS_INSTITUTION_EMAIL).all()
    assert len(institutions) == 1
    institution = institutions[0]
    assert institution.name == DOCS_INSTITUTION_NAME
    assert institution.institution_type == "clinic"

    user = User.query.filter_by(email=DOCS_USER_EMAIL).one()
    assert user.institution_id == institution.id
    assert user.institution_role == InstitutionRole.ADMIN.value
    assert user.first_name == "Test"
    assert user.last_name == "Documentation"

    patients = InstitutionPatient.query.filter_by(institution_id=institution.id).all()
    assert len(patients) == 3
    identities = {(p.first_name, p.last_name) for p in patients}
    assert identities == {("TEST", "Test"), ("Alice", "Exemple"), ("Marc", "Démonstration")}

    refs = {
        req.external_reference: req
        for req in TransportRequest.query.filter_by(institution_id=institution.id).all()
    }
    assert set(refs) == {
        "DOCS-REQ-001",
        "DOCS-REQ-002",
        "DOCS-REQ-003",
        "DOCS-REQ-004",
    }

    req_001 = refs["DOCS-REQ-001"]
    assert req_001.status == RequestStatus.SENT.value
    assert req_001.mission_type == MissionType.PATIENT_TRANSPORT.value
    assert req_001.carrier_source == CarrierSource.LIRIE.value

    req_002 = refs["DOCS-REQ-002"]
    assert req_002.mission_type == MissionType.MATERIAL_DELIVERY.value
    assert req_002.delivery_description == "Livraison de documents"
    assert req_002.status == RequestStatus.SENT.value

    req_003 = refs["DOCS-REQ-003"]
    assert req_003.status == RequestStatus.EXTERNAL_ASSIGNED.value
    assert req_003.carrier_source == CarrierSource.EXTERNAL.value
    assert req_003.external_carrier_name == "Taxi Démo Genève"
    assert req_003.external_carrier_email == f"externe@{DOCS_EMAIL_DOMAIN}"

    req_004 = refs["DOCS-REQ-004"]
    assert req_004.status == RequestStatus.ACCEPTED.value

    notifications = InstitutionNotification.query.filter_by(
        institution_id=institution.id
    ).all()
    assert len(notifications) == 3
    assert {n.event_type for n in notifications} == {
        "request_sent",
        "request_converted",
        "booking_message",
    }

    emails = [user.email, institution.contact_email, req_003.external_carrier_email]
    assert all(str(email).endswith(f"@{DOCS_EMAIL_DOMAIN}") for email in emails)

    headers = institution_bearer_headers(
        db,
        user,
        institution,
        institution_role=InstitutionRole.ADMIN.value,
    )
    pdf = client.get(
        f"/api/v1/institutions/exports/requests/{req_003.id}/pdf?variant=operational",
        headers=headers,
    )
    assert pdf.status_code == 200
    assert pdf.data[:4] == b"%PDF"
    assert summary["patients"] == 3


def test_seed_idempotent_reset_sans_doublon(docs_password, db):
    first = reset_and_seed_institution_docs(commit=False)
    counts_1 = _tenant_counts()
    second = reset_and_seed_institution_docs(commit=False)
    counts_2 = _tenant_counts()

    assert counts_1 == counts_2
    assert counts_2 == {
        "institutions": 1,
        "users": 1,
        "patients": 3,
        "requests": 4,
        "notifications": 3,
    }
    assert first["institution_public_id"] == second["institution_public_id"]
