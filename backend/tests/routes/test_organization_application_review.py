"""Instruction des préinscriptions : aucun portail n'est ouvert."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Query

from models import Company, Institution, PortalGrant, User, UserRole
from models.organization_application import (
    OrganizationApplication,
    OrganizationApplicationEvent,
    OrganizationApplicationToken,
)
from tests.helpers.institution_auth import institution_bearer_headers
from tests.routes.test_organization_applications import (
    _company_payload,
    _email,
    _institution_payload,
    _post,
    _token,
    _verify,
)
from tests.security.test_portal_grant_phase1 import _headers, _user

pytestmark = pytest.mark.integration


@pytest.fixture
def org_tables(db):
    OrganizationApplication.__table__.create(bind=db.engine, checkfirst=True)
    OrganizationApplicationEvent.__table__.create(bind=db.engine, checkfirst=True)
    OrganizationApplicationToken.__table__.create(bind=db.engine, checkfirst=True)
    PortalGrant.__table__.create(bind=db.engine, checkfirst=True)
    return db


def _received(client, app, payload):
    app.config["_ORG_APP_LINKS"] = []
    app.config["_ORG_APP_NOTICES"] = []
    assert _post(client, payload).status_code == 202
    confirmed = _verify(client, _token(app))
    assert confirmed.status_code == 200
    return confirmed.get_json()["tracking_token"]


def _admin(app, db):
    user = _user(db, UserRole.ADMIN)
    return user, _headers(app, user)


def _decide(client, headers, public_id, payload):
    return client.post(
        f"/api/v1/admin/organization-applications/{public_id}/decisions",
        json=payload,
        headers=headers,
    )


def test_liste_distingue_le_dossier_incomplet(client, app, org_tables):
    app.config["_ORG_APP_LINKS"] = []
    unverified = _email()
    company_email = _email()
    institution_email = _email()
    assert _post(client, _company_payload(unverified, legal_name="Incomplet SA")).status_code == 202
    _received(client, app, _company_payload(company_email, legal_name="Transports du Lac"))
    _received(
        client,
        app,
        _institution_payload(institution_email, establishment_name="Clinique des Alpes"),
    )
    admin, headers = _admin(app, org_tables)
    listed = client.get("/api/v1/admin/organization-applications", headers=headers)
    assert listed.status_code == 200
    items = listed.get_json()["items"]
    by_email = {item["applicant_email"]: item for item in items}
    assert by_email[unverified]["incomplete"] is True
    assert by_email[unverified]["queue_label"] == "Incomplet"
    assert by_email[unverified]["submitted_at"] is None
    assert by_email[company_email]["incomplete"] is False
    assert by_email[company_email]["queue_label"] == "À traiter"
    assert by_email[institution_email]["kind"] == "institution"
    filtered = client.get(
        "/api/v1/admin/organization-applications?kind=institution&q=Clinique",
        headers=headers,
    )
    names = [item["organization_name"] for item in filtered.get_json()["items"]]
    assert names == ["Clinique des Alpes"]
    assert admin.role == UserRole.ADMIN


def test_parcours_examen_complement_reprise_et_refus(client, app, org_tables):
    companies_before = org_tables.session.query(Company).count()
    institutions_before = org_tables.session.query(Institution).count()
    grants_before = org_tables.session.query(PortalGrant).count()
    email = _email()
    tracking = _received(client, app, _company_payload(email))
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    _, headers = _admin(app, org_tables)
    locks = []
    original = Query.with_for_update

    def spy(self, *args, **kwargs):
        locks.append(True)
        return original(self, *args, **kwargs)

    with patch.object(Query, "with_for_update", spy):
        started = _decide(client, headers, row.public_id, {"action": "start_review"})
    assert started.status_code == 200
    assert started.get_json()["status"] == "in_review"
    assert started.get_json()["notification"]["state"] == "queued"
    assert started.get_json()["notification"]["delivered"] is False
    assert locks
    again = _decide(client, headers, row.public_id, {"action": "start_review"})
    assert again.status_code == 200
    assert again.get_json()["notification"] is None
    review_events = [
        event.event_type
        for event in OrganizationApplicationEvent.query.filter_by(application_id=row.id)
        if event.event_type == "review_started"
    ]
    assert review_events == ["review_started"]
    missing = _decide(
        client,
        headers,
        row.public_id,
        {"action": "request_info", "note": "court"},
    )
    assert missing.status_code == 400
    requested = _decide(
        client,
        headers,
        row.public_id,
        {"action": "request_info", "note": "Joignez le registre du commerce."},
    )
    assert requested.status_code == 200
    assert requested.get_json()["status"] == "needs_info"
    assert requested.get_json()["sla"]["due_at"] is None
    assert requested.get_json()["sla"]["calendar"] == "pending"
    assert requested.get_json()["sla"]["suspensions"][0]["until"] is None
    too_soon = _decide(client, headers, row.public_id, {"action": "resume_review"})
    assert too_soon.status_code == 400
    received = _decide(
        client,
        headers,
        row.public_id,
        {"action": "record_info", "note": "Registre reçu par courriel."},
    )
    assert received.status_code == 200
    assert received.get_json()["status"] == "needs_info"
    resumed = _decide(client, headers, row.public_id, {"action": "resume_review"})
    assert resumed.status_code == 200
    assert resumed.get_json()["status"] == "in_review"
    refused = _decide(
        client,
        headers,
        row.public_id,
        {"action": "refuse", "note": "Le représentant n'a pas pu être joint."},
    )
    assert refused.status_code == 200
    assert refused.get_json()["status"] == "refused"
    repeat = _decide(
        client,
        headers,
        row.public_id,
        {"action": "refuse", "note": "Le représentant n'a pas pu être joint."},
    )
    assert repeat.status_code == 200
    assert repeat.get_json()["notification"] is None
    notices = [item["template"] for item in app.config["_ORG_APP_NOTICES"]]
    assert notices == ["review_started", "info_requested", "info_received", "review_resumed", "refused"]
    mine = client.get(
        "/api/v1/organization-applications/mine",
        headers={"Authorization": f"Bearer {tracking}"},
    )
    public_types = [event["type"] for event in mine.get_json()["events"]]
    assert "notification_queued" not in public_types
    assert "refused" in public_types
    assert org_tables.session.query(Company).count() == companies_before
    assert org_tables.session.query(Institution).count() == institutions_before
    assert org_tables.session.query(PortalGrant).count() == grants_before


def test_observation_interne_et_approbation_bloquee(client, app, org_tables):
    email = _email()
    tracking = _received(
        client,
        app,
        _institution_payload(email, establishment_name="EMS de la Côte"),
    )
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    admin, headers = _admin(app, org_tables)
    noted = _decide(
        client,
        headers,
        row.public_id,
        {"action": "observe", "note": "Pièce d'identité à revoir plus tard."},
    )
    assert noted.status_code == 200
    identity = _decide(
        client,
        headers,
        row.public_id,
        {
            "action": "check_identity",
            "confirmed": False,
            "note": "Le nom ne correspond pas au registre.",
        },
    )
    assert identity.status_code == 200
    assert identity.get_json()["checks"]["identity"]["confirmed"] is False
    mine = client.get(
        "/api/v1/organization-applications/mine",
        headers={"Authorization": f"Bearer {tracking}"},
    )
    public_notes = " ".join(event.get("note") or "" for event in mine.get_json()["events"])
    assert "Pièce d'identité" not in public_notes
    blocked = client.post(
        f"/api/v1/admin/organization-applications/{row.public_id}/approve",
        json={},
        headers=headers,
    )
    assert blocked.status_code == 403
    assert blocked.get_json()["code"] == "activation_bloquee"
    org_tables.session.refresh(row)
    assert row.status == "received"
    assert admin.id != row.user_id


def test_auto_decision_et_comptes_refuses(client, app, org_tables):
    email = _email()
    _received(client, app, _company_payload(email))
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    admin, headers = _admin(app, org_tables)
    admin.email = email
    org_tables.session.flush()
    denied = _decide(client, headers, row.public_id, {"action": "start_review"})
    assert denied.status_code == 403
    assert denied.get_json()["code"] == "auto_decision_refusee"
    client_user = _user(org_tables, UserRole.CLIENT)
    company_user = _user(org_tables, UserRole.COMPANY)
    for user in (client_user, company_user):
        response = client.get(
            "/api/v1/admin/organization-applications",
            headers=_headers(app, user),
        )
        assert response.status_code == 403
    institution_user = _user(org_tables, UserRole.INSTITUTION)
    institution = Institution(
        name="EMS de contrôle",
        public_id=str(uuid.uuid4()),
    )
    org_tables.session.add(institution)
    org_tables.session.flush()
    institution_response = client.get(
        "/api/v1/admin/organization-applications",
        headers=institution_bearer_headers(org_tables, institution_user, institution),
    )
    assert institution_response.status_code == 403
    other = client.get(
        "/api/v1/admin/organization-applications/dossier-inexistant",
        headers=headers,
    )
    assert other.status_code == 404


def test_notification_remise_sans_pretendre_la_livraison(client, app, org_tables):
    email = _email()
    _received(client, app, _company_payload(email))
    row = OrganizationApplication.query.filter_by(applicant_email=email).one()
    from services.organization_application_review import _hand_to_mailer

    with patch("services.organization_application_review.mail.send") as send:
        app.config["MAIL_SUPPRESS_SEND"] = False
        result = _hand_to_mailer(row, "review_started", "Examen en cours.")
    send.assert_called_once()
    assert result["delivered"] is False
    assert result["state"] == "handed_to_mailer"
    message = send.call_args.args[0]
    assert "token=" not in message.body
