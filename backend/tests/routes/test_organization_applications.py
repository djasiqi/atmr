"""Préinscription entreprise et institution : aucun droit n'est ouvert."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from models import Company, Institution, PortalGrant, User, UserRole
from models.enums import OrganizationApplicationStatus
from models.organization_application import (
    OrganizationApplication,
    OrganizationApplicationEvent,
    OrganizationApplicationToken,
)
from services.organization_applications import SLA_NOTICE
from tests.security.test_portal_grant_phase1 import _company, _headers, _user

pytestmark = pytest.mark.integration


@pytest.fixture
def org_tables(db):
    OrganizationApplication.__table__.create(bind=db.engine, checkfirst=True)
    OrganizationApplicationEvent.__table__.create(bind=db.engine, checkfirst=True)
    OrganizationApplicationToken.__table__.create(bind=db.engine, checkfirst=True)
    PortalGrant.__table__.create(bind=db.engine, checkfirst=True)
    return db


def _email() -> str:
    return f"phase2a-{uuid.uuid4().hex[:10]}@example.test"


def _company_payload(email: str, **extra) -> dict:
    payload = {
        "kind": "company",
        "legal_name": "Transports du Lac",
        "ide_number": "CHE-123.456.789",
        "street": "Rue du Port 4",
        "postal_code": "1003",
        "city": "Lausanne",
        "applicant_first_name": "Nora",
        "applicant_last_name": "Meier",
        "applicant_function": "Directrice",
        "applicant_email": email,
        "applicant_phone": "+41790001122",
        "privacy_consent": True,
    }
    payload.update(extra)
    return payload


def _institution_payload(email: str, **extra) -> dict:
    payload = {
        "kind": "institution",
        "establishment_name": "Clinique des Alpes",
        "establishment_category": "clinique",
        "street": "Chemin des Soins 2",
        "postal_code": "1950",
        "city": "Sion",
        "applicant_first_name": "Luc",
        "applicant_last_name": "Favre",
        "applicant_function": "Responsable admissions",
        "applicant_email": email,
        "applicant_phone": "0271234567",
        "privacy_consent": True,
    }
    payload.update(extra)
    return payload


def _post(client, payload):
    return client.post("/api/v1/organization-applications", json=payload)


def _token(app) -> str:
    links = app.config.get("_ORG_APP_LINKS") or []
    assert links, "aucun lien de confirmation"
    return links[-1]["token"]


def _verify(client, token: str):
    return client.post(
        "/api/v1/organization-applications/verify",
        json={"token": token},
    )


def test_entreprise_puis_confirmation(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    email = _email()
    companies_before = org_tables.session.query(Company).count()
    grants_before = org_tables.session.query(PortalGrant).count()
    users_before = org_tables.session.query(User).count()
    created = _post(client, _company_payload(email))
    assert created.status_code == 202
    assert "lien" in created.get_json()["message"]
    assert org_tables.session.query(Company).count() == companies_before
    assert org_tables.session.query(PortalGrant).count() == grants_before
    assert org_tables.session.query(User).count() == users_before
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    assert row.status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value
    assert row.user_id is None
    confirmed = _verify(client, _token(app))
    body = confirmed.get_json()
    assert confirmed.status_code == 200
    assert body["status"] == "received"
    assert body["sla_notice"] == SLA_NOTICE
    assert "due_at" not in body
    assert body["email_verified"] is True
    mine = client.get(
        "/api/v1/organization-applications/mine",
        headers={"Authorization": f"Bearer {body['tracking_token']}"},
    )
    assert mine.status_code == 200
    assert mine.get_json()["organization_name"] == "Transports du Lac"
    reservations = client.get(
        "/api/v1/companies/me/reservations",
        headers={"Authorization": f"Bearer {body['tracking_token']}"},
    )
    assert reservations.status_code != 200
    org_tables.session.refresh(row)
    assert row.status == "received"
    assert org_tables.session.query(PortalGrant).count() == grants_before


def test_institution_separee_de_l_entreprise(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    email = _email()
    institutions_before = org_tables.session.query(Institution).count()
    created = _post(client, _institution_payload(email))
    assert created.status_code == 202
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    assert row.kind == "institution"
    assert row.legal_name is None
    assert row.establishment_category == "clinique"
    assert org_tables.session.query(Institution).count() == institutions_before
    confirmed = _verify(client, _token(app))
    assert confirmed.status_code == 200
    assert confirmed.get_json()["kind"] == "institution"
    institution_me = client.get(
        "/api/v1/institutions/me",
        headers={"Authorization": f"Bearer {confirmed.get_json()['tracking_token']}"},
    )
    assert institution_me.status_code != 200


def test_lien_expire_et_renvoi(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    email = _email()
    assert _post(client, _company_payload(email)).status_code == 202
    first = _token(app)
    token_row = (
        OrganizationApplicationToken.query.filter_by(purpose="email_verify")
        .order_by(OrganizationApplicationToken.id.desc())
        .first()
    )
    assert token_row is not None
    token_row.expires_at = datetime.now(UTC) - timedelta(minutes=5)
    org_tables.session.flush()
    expired = _verify(client, first)
    assert expired.status_code == 400
    assert expired.get_json()["error"] == "lien_invalide"
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    assert row.status == "email_unverified"
    resent = _post(client, _company_payload(email))
    assert resent.status_code == 202
    assert OrganizationApplication.query.filter_by(applicant_email=email).count() == 1
    assert _verify(client, first).status_code == 400
    second = _verify(client, _token(app))
    assert second.status_code == 200
    assert second.get_json()["status"] == "received"


def test_doublon_apres_reception_ne_cree_pas_un_second_dossier(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    email = _email()
    assert _post(client, _company_payload(email)).status_code == 202
    assert _verify(client, _token(app)).status_code == 200
    again = _post(client, _company_payload(email, legal_name="Autre nom"))
    assert again.status_code == 202
    assert OrganizationApplication.query.filter_by(applicant_email=email).count() == 1
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    assert row.legal_name == "Transports du Lac"
    assert row.status == "received"


def test_organisation_connue_sans_rattachement(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    owner = _user(org_tables, UserRole.COMPANY)
    company = _company(org_tables, owner, approved=True)
    company.name = f"Transports {uuid.uuid4().hex[:8]}"
    company.uid_ide = "CHE-999.888.777"
    org_tables.session.flush()
    companies_before = org_tables.session.query(Company).count()
    email = _email()
    assert (
        _post(
            client,
            _company_payload(
                email,
                legal_name=company.name,
                ide_number="CHE-999.888.777",
            ),
        ).status_code
        == 202
    )
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    assert row.matched_company_id == company.id
    assert org_tables.session.query(Company).count() == companies_before
    confirmed = _verify(client, _token(app))
    assert confirmed.get_json()["existing_organization_hint"] is True
    org_tables.session.refresh(owner)
    assert owner.role == UserRole.COMPANY


def test_compte_client_existant_ne_change_pas_de_role(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    client_user = _user(org_tables, UserRole.CLIENT)
    grants_before = org_tables.session.query(PortalGrant).count()
    created = _post(client, _company_payload(client_user.email))
    assert created.status_code == 202
    assert "déjà" not in created.get_json()["message"].lower()
    assert "deja" not in created.get_json()["message"].lower()
    row = OrganizationApplication.query.filter_by(
        applicant_email=client_user.email.lower()
    ).one()
    assert row.user_id == client_user.id
    org_tables.session.refresh(client_user)
    assert client_user.role == UserRole.CLIENT
    assert _verify(client, _token(app)).status_code == 200
    assert org_tables.session.query(PortalGrant).count() == grants_before


def test_suivi_isole_entre_deux_demandes(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    first_email = _email()
    second_email = _email()
    assert (
        _post(
            client, _company_payload(first_email, legal_name="Alpha Transport")
        ).status_code
        == 202
    )
    first_link = _token(app)
    assert (
        _post(
            client,
            _institution_payload(second_email, establishment_name="Beta Soins"),
        ).status_code
        == 202
    )
    second_link = _token(app)
    first = _verify(client, first_link).get_json()
    second = _verify(client, second_link).get_json()
    mine = client.get(
        "/api/v1/organization-applications/mine",
        headers={"Authorization": f"Bearer {first['tracking_token']}"},
    )
    assert mine.status_code == 200
    assert mine.get_json()["organization_name"] == "Alpha Transport"
    assert mine.get_json()["organization_name"] != second["organization_name"]
    stranger = client.get(
        "/api/v1/organization-applications/mine",
        headers={"Authorization": "Bearer pas-un-jeton"},
    )
    assert stranger.status_code == 401


def test_payload_professionnel_et_contournement_refuses(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    email = _email()
    refused = _post(
        client,
        _company_payload(email, role="COMPANY", portal_grant={"status": "active"}),
    )
    assert refused.status_code == 400
    assert refused.get_json()["error"] == "droit_refuse"
    assert OrganizationApplication.query.filter_by(applicant_email=email).count() == 0
    missing = _post(client, {"kind": "company", "applicant_email": email})
    assert missing.status_code == 400
    direct = client.post(
        "/api/v1/organization-applications/portal-grant",
        json={"status": "active", "portal": "company"},
    )
    assert direct.status_code in {404, 405}
    approve = client.post(
        "/api/v1/organization-applications/mine",
        json={"status": "approved"},
    )
    assert approve.status_code == 405
    honeypot_email = _email()
    honeypot = _post(
        client, _company_payload(honeypot_email, website="https://spam.example")
    )
    assert honeypot.status_code == 202
    assert (
        OrganizationApplication.query.filter_by(applicant_email=honeypot_email).count()
        == 0
    )


def test_conflit_unique_simultane_renvoie_le_meme_accuse(app, org_tables):
    from unittest.mock import patch

    from services.organization_applications import submit_application

    def raise_unique(*_args, **_kwargs):
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    with patch.object(org_tables.session, "flush", side_effect=raise_unique):
        result = submit_application(_company_payload(_email()))
    assert "lien de confirmation" in result["message"]


def test_echec_courriel_ne_propage_pas_et_ne_journalise_pas_le_secret(app, caplog):
    import logging
    from unittest.mock import patch

    from services.organization_applications import deliver_verification_link

    secret = f"secret-{uuid.uuid4().hex}"
    with app.app_context():
        app.config["TESTING"] = False
        try:
            with (
                caplog.at_level(logging.WARNING),
                patch(
                    "services.organization_applications.mail.send",
                    side_effect=AssertionError("expéditeur absent"),
                ),
            ):
                deliver_verification_link("phase2a-courriel@example.test", secret)
        finally:
            app.config["TESTING"] = True
    assert secret not in caplog.text
    assert "impossible" in caplog.text


def test_session_client_valide_n_ouvre_pas_les_portails(client, app, org_tables):
    user = _user(org_tables, UserRole.CLIENT)
    headers = _headers(app, user)
    reservations = client.get("/api/v1/companies/me/reservations", headers=headers)
    invoices = client.get("/api/v1/invoices/companies/1/invoices", headers=headers)
    institution = client.get("/api/v1/institutions/me", headers=headers)
    assert reservations.status_code in {401, 403}
    assert invoices.status_code in {401, 403}
    assert institution.status_code in {401, 403}
    assert 422 not in {
        reservations.status_code,
        invoices.status_code,
        institution.status_code,
    }


def test_indice_unique_empeche_deux_dossiers_ouverts(org_tables):
    email = _email()
    first = OrganizationApplication(
        public_id=str(uuid.uuid4()),
        kind="company",
        status="email_unverified",
        applicant_first_name="A",
        applicant_last_name="B",
        applicant_function="Direction",
        applicant_email=email,
        applicant_phone="+41790000000",
        legal_name="Un",
        street="Rue 1",
        postal_code="1000",
        city="Lausanne",
        country="CH",
    )
    second = OrganizationApplication(
        public_id=str(uuid.uuid4()),
        kind="company",
        status="received",
        applicant_first_name="A",
        applicant_last_name="B",
        applicant_function="Direction",
        applicant_email=email,
        applicant_phone="+41790000000",
        legal_name="Deux",
        street="Rue 1",
        postal_code="1000",
        city="Lausanne",
        country="CH",
    )
    nested = org_tables.session.begin_nested()
    org_tables.session.add(first)
    org_tables.session.flush()
    org_tables.session.add(second)
    with pytest.raises(IntegrityError):
        org_tables.session.flush()
    nested.rollback()
