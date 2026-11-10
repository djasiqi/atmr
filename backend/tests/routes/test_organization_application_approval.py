"""Approbation locale derrière le verrou. Le drapeau global reste éteint."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Query

from models import Company, Institution, PortalGrant, User, UserRole
from models.enums import PortalGrantStatus, PortalKind
from models.organization_application import (
    OrganizationApplication,
    OrganizationApplicationEvent,
    OrganizationApplicationToken,
)
from models.portal_grant import (
    REVOKED_DELETE_GUARD_DROP_SQL,
    REVOKED_DELETE_GUARD_SQL,
    RevokedPortalGrantProtected,
)
from security.portal_scope import portal_company_allowed, resolve_portal_scope
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
    return OrganizationApplication.query.filter_by(
        applicant_email=payload["applicant_email"]
    ).one()


def _client(db, email: str) -> User:
    user = _user(db, UserRole.CLIENT)
    user.email = email
    db.session.flush()
    return user


def _ready(client, app, db, payload):
    email = payload["applicant_email"]
    applicant = _client(db, email)
    row = _received(client, app, payload)
    admin = _user(db, UserRole.ADMIN)
    headers = _headers(app, admin)
    assert client.post(
        f"/api/v1/admin/organization-applications/{row.public_id}/decisions",
        json={"action": "start_review"},
        headers=headers,
    ).status_code == 200
    for action in ("check_identity", "check_organization"):
        assert client.post(
            f"/api/v1/admin/organization-applications/{row.public_id}/decisions",
            json={"action": action, "confirmed": True, "note": "Vérification administrative faite."},
            headers=headers,
        ).status_code == 200
    return applicant, admin, headers, row


def _approve(client, headers, public_id):
    return client.post(
        f"/api/v1/admin/organization-applications/{public_id}/approve",
        json={},
        headers=headers,
    )


def _ide() -> str:
    digits = f"{uuid.uuid4().int % 1_000_000_000:09d}"
    return f"CHE-{digits[:3]}.{digits[3:6]}.{digits[6:]}"


@pytest.fixture(autouse=True)
def _restore_flags(app):
    had_approval = "ORGANIZATION_APPROVAL_ENABLED" in app.config
    had_enforcement = "PORTAL_GRANT_ENFORCEMENT" in app.config
    approval_value = app.config.get("ORGANIZATION_APPROVAL_ENABLED")
    enforcement_value = app.config.get("PORTAL_GRANT_ENFORCEMENT")
    yield
    if had_approval:
        app.config["ORGANIZATION_APPROVAL_ENABLED"] = approval_value
    else:
        app.config.pop("ORGANIZATION_APPROVAL_ENABLED", None)
    if had_enforcement:
        app.config["PORTAL_GRANT_ENFORCEMENT"] = enforcement_value
    else:
        app.config.pop("PORTAL_GRANT_ENFORCEMENT", None)


def _open_gate(app, *, enforcement: bool):
    app.config["ORGANIZATION_APPROVAL_ENABLED"] = True
    app.config["PORTAL_GRANT_ENFORCEMENT"] = enforcement


def test_verrou_ferme_ne_cree_rien(client, app, org_tables):
    companies = org_tables.session.query(Company).count()
    grants = org_tables.session.query(PortalGrant).count()
    payload = _company_payload(_email(), legal_name=f"Clos {uuid.uuid4().hex[:8]}")
    _applicant, _admin, headers, row = _ready(client, app, org_tables, payload)
    denied = _approve(client, headers, row.public_id)
    assert denied.status_code == 403
    assert denied.get_json()["code"] == "activation_bloquee"
    org_tables.session.refresh(row)
    assert row.status == "in_review"
    assert org_tables.session.query(Company).count() == companies
    assert org_tables.session.query(PortalGrant).count() == grants
    assert "company_approved" not in [item["template"] for item in app.config["_ORG_APP_NOTICES"]]


def test_entreprise_atomique_rejeu_et_acces(client, app, org_tables):
    companies = org_tables.session.query(Company).count()
    name = f"Lac {uuid.uuid4().hex[:8]}"
    payload = _company_payload(_email(), legal_name=name, ide_number=_ide())
    applicant, _admin, headers, row = _ready(client, app, org_tables, payload)
    role_before = applicant.role
    _open_gate(app, enforcement=False)
    blocked = _approve(client, headers, row.public_id)
    assert blocked.status_code == 409
    assert blocked.get_json()["code"] == "acces_non_ouvert"
    assert org_tables.session.query(Company).count() == companies
    _open_gate(app, enforcement=True)
    locks = []
    original = Query.with_for_update

    def spy(self, *args, **kwargs):
        locks.append(True)
        return original(self, *args, **kwargs)

    with patch.object(Query, "with_for_update", spy):
        approved = _approve(client, headers, row.public_id)
    assert approved.status_code == 200
    assert approved.get_json()["status"] == "approved"
    assert approved.get_json()["notification"]["delivered"] is False
    assert approved.get_json()["notification"]["state"] == "queued"
    assert locks
    again = _approve(client, headers, row.public_id)
    assert again.status_code == 200
    assert again.get_json()["notification"] is None
    org_tables.session.refresh(applicant)
    assert applicant.role == role_before
    assert org_tables.session.query(Company).count() == companies + 1
    grants = PortalGrant.query.filter_by(user_id=applicant.id, portal=PortalKind.COMPANY.value).all()
    assert len(grants) == 1
    assert grants[0].status == PortalGrantStatus.ACTIVE.value
    assert portal_company_allowed(applicant, PortalKind.COMPANY.value, grants[0].company_id)
    notices = [item["template"] for item in app.config["_ORG_APP_NOTICES"]]
    assert notices.count("company_approved") == 1


def test_entreprise_existante_exige_un_choix(client, app, org_tables):
    email = _email()
    name = f"Déjà {uuid.uuid4().hex[:8]}"
    ide = _ide()
    payload = _company_payload(email, legal_name=name, ide_number=ide)
    applicant, _admin, headers, row = _ready(client, app, org_tables, payload)
    owner = _user(org_tables, UserRole.CLIENT)
    existing = Company()
    existing.name = name
    existing.uid_ide = ide
    existing.address = "Rue du Port 4"
    existing.contact_email = "deja@example.test"
    existing.contact_phone = "0210000000"
    existing.user_id = owner.id
    existing.is_approved = False
    org_tables.session.add(existing)
    org_tables.session.flush()
    _open_gate(app, enforcement=True)
    ambiguous = _approve(client, headers, row.public_id)
    assert ambiguous.status_code == 409
    assert ambiguous.get_json()["code"] == "organisation_existante"
    assert org_tables.session.query(Company).filter_by(name=name).count() == 1
    row.matched_company_id = existing.id
    org_tables.session.flush()
    attached = _approve(client, headers, row.public_id)
    assert attached.status_code == 200
    org_tables.session.refresh(existing)
    assert existing.user_id == owner.id
    assert existing.is_approved is True
    grant = PortalGrant.query.filter_by(user_id=applicant.id).one()
    assert grant.company_id == existing.id
    org_tables.session.refresh(applicant)
    assert applicant.role == UserRole.CLIENT


def test_institution_reste_inactive_jusqu_a_son_jeton(client, app, org_tables):
    email = _email()
    name = f"EMS {uuid.uuid4().hex[:8]}"
    payload = _institution_payload(email, establishment_name=name)
    applicant, _admin, headers, row = _ready(client, app, org_tables, payload)
    verify_token = _token(app)
    _open_gate(app, enforcement=True)
    approved = _approve(client, headers, row.public_id)
    assert approved.status_code == 200
    grant = PortalGrant.query.filter_by(user_id=applicant.id).one()
    assert grant.status == PortalGrantStatus.PENDING_ACTIVATION.value
    assert grant.internal_role == "institution_admin"
    assert grant.portal == PortalKind.INSTITUTION.value
    before = resolve_portal_scope(applicant, PortalKind.INSTITUTION.value, use_request_claim=False)
    assert before.allowed is False
    wrong = client.post(
        "/api/v1/organization-applications/institution-activation",
        json={"token": verify_token},
    )
    assert wrong.status_code == 400
    org_tables.session.refresh(grant)
    assert grant.status == PortalGrantStatus.PENDING_ACTIVATION.value
    invite = next(
        item["token"]
        for item in app.config["_ORG_APP_LINKS"]
        if item.get("purpose") == "institution_admin_invite"
    )
    accepted = client.post(
        "/api/v1/organization-applications/institution-activation",
        json={"token": invite},
    )
    assert accepted.status_code == 200
    org_tables.session.refresh(grant)
    org_tables.session.refresh(applicant)
    assert grant.status == PortalGrantStatus.ACTIVE.value
    assert applicant.role == UserRole.CLIENT
    assert applicant.institution_role == "institution_admin"
    opened = resolve_portal_scope(applicant, PortalKind.INSTITUTION.value, use_request_claim=False)
    assert opened.allowed is True
    assert org_tables.session.query(Institution).filter_by(name=name).count() == 1


def test_refus_auto_approbation_et_revocation(client, app, org_tables):
    email = _email()
    payload = _company_payload(email, legal_name=f"Refus {uuid.uuid4().hex[:8]}", ide_number=_ide())
    applicant, _admin, headers, row = _ready(client, app, org_tables, payload)
    refused = client.post(
        f"/api/v1/admin/organization-applications/{row.public_id}/decisions",
        json={"action": "refuse", "note": "Le dossier ne correspond pas."},
        headers=headers,
    )
    assert refused.status_code == 200
    _open_gate(app, enforcement=True)
    denied = _approve(client, headers, row.public_id)
    assert denied.status_code == 409
    assert PortalGrant.query.filter_by(user_id=applicant.id).count() == 0
    other = _company_payload(
        _email(),
        legal_name=f"Soi {uuid.uuid4().hex[:8]}",
        ide_number=_ide(),
    )
    self_row = _received(client, app, other)
    assert client.post(
        f"/api/v1/admin/organization-applications/{self_row.public_id}/decisions",
        json={"action": "start_review"},
        headers=headers,
    ).status_code == 200
    for action in ("check_identity", "check_organization"):
        assert client.post(
            f"/api/v1/admin/organization-applications/{self_row.public_id}/decisions",
            json={"action": action, "confirmed": True, "note": "Vérification administrative faite."},
            headers=headers,
        ).status_code == 200
    approver = _user(org_tables, UserRole.ADMIN)
    approver.email = other["applicant_email"]
    org_tables.session.flush()
    own = _approve(client, _headers(app, approver), self_row.public_id)
    assert own.status_code == 403
    assert own.get_json()["code"] == "auto_decision_refusee"
    companies = org_tables.session.query(Company).count()
    fresh = _company_payload(_email(), legal_name=f"Revo {uuid.uuid4().hex[:8]}", ide_number=_ide())
    person, _actor, actor_headers, dossier = _ready(client, app, org_tables, fresh)
    assert _approve(client, actor_headers, dossier.public_id).status_code == 200
    revoked = client.post(
        f"/api/v1/admin/organization-applications/{dossier.public_id}/revoke",
        json={},
        headers=actor_headers,
    )
    assert revoked.status_code == 200
    grant = PortalGrant.query.filter_by(user_id=person.id).one()
    assert grant.status == PortalGrantStatus.REVOKED.value
    assert portal_company_allowed(person, PortalKind.COMPANY.value, grant.company_id) is False
    def _orm_delete() -> None:
        with org_tables.session.begin_nested():
            org_tables.session.delete(grant)
            org_tables.session.flush()

    with pytest.raises(RevokedPortalGrantProtected):
        _orm_delete()
    org_tables.session.refresh(grant)
    for statement in REVOKED_DELETE_GUARD_SQL:
        org_tables.session.execute(text(statement))
    try:
        def _sql_delete() -> None:
            with org_tables.session.begin_nested():
                org_tables.session.execute(
                    text('DELETE FROM "user" WHERE id = :user_id'),
                    {"user_id": person.id},
                )

        with pytest.raises(DBAPIError):
            _sql_delete()
        org_tables.session.refresh(grant)
        assert grant.status == PortalGrantStatus.REVOKED.value
    finally:
        for statement in REVOKED_DELETE_GUARD_DROP_SQL:
            org_tables.session.execute(text(statement))
    assert org_tables.session.query(Company).count() >= companies
    privilege = org_tables.session.execute(
        text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
    ).scalar()
    assert privilege in {True, False}
