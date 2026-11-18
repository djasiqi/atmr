"""Invitations chauffeur. Le verrou d'activation reste fermé hors test."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Query

from models import Driver, PortalGrant, User, UserRole
from models.driver_invitation import DriverInvitation
from models.enums import DriverType, PortalGrantStatus, PortalKind
from security.portal_scope import portal_company_allowed
from services.driver_invitations import _hand_off
from tests.security.test_portal_grant_phase1 import _company, _headers, _user

pytestmark = pytest.mark.integration

PASSWORD = "Chauffeur-2e-Test!"


@pytest.fixture
def invite_tables(db):
    DriverInvitation.__table__.create(bind=db.engine, checkfirst=True)
    PortalGrant.__table__.create(bind=db.engine, checkfirst=True)
    return db


@pytest.fixture(autouse=True)
def _restore_flags(app):
    had_gate = "DRIVER_INVITATION_ACTIVATION" in app.config
    had_enforcement = "PORTAL_GRANT_ENFORCEMENT" in app.config
    had_mail = "MAIL_SUPPRESS_SEND" in app.config
    gate = app.config.get("DRIVER_INVITATION_ACTIVATION")
    enforcement = app.config.get("PORTAL_GRANT_ENFORCEMENT")
    mail_flag = app.config.get("MAIL_SUPPRESS_SEND")
    app.config["_DRIVER_INVITE_LINKS"] = []
    app.config["_DRIVER_INVITE_NOTICES"] = []
    yield
    if had_gate:
        app.config["DRIVER_INVITATION_ACTIVATION"] = gate
    else:
        app.config.pop("DRIVER_INVITATION_ACTIVATION", None)
    if had_enforcement:
        app.config["PORTAL_GRANT_ENFORCEMENT"] = enforcement
    else:
        app.config.pop("PORTAL_GRANT_ENFORCEMENT", None)
    if had_mail:
        app.config["MAIL_SUPPRESS_SEND"] = mail_flag
    else:
        app.config.pop("MAIL_SUPPRESS_SEND", None)


def _owner(db, *, approved: bool = True):
    owner = _user(db, UserRole.COMPANY)
    owner.email = f"resp-{uuid.uuid4().hex[:8]}@example.test"
    company = _company(db, owner, approved=approved)
    db.session.flush()
    return owner, company


def _company_grant(db, owner, company) -> None:
    """Droit entreprise nécessaire quand le drapeau d'habilitation est ouvert."""
    db.session.add(
        PortalGrant(
            user_id=owner.id,
            portal=PortalKind.COMPANY.value,
            company_id=company.id,
            status=PortalGrantStatus.ACTIVE.value,
        )
    )
    db.session.flush()


def _invite(client, app, db, email: str | None = None):
    owner, company = _owner(db)
    if app.config.get("PORTAL_GRANT_ENFORCEMENT"):
        _company_grant(db, owner, company)
    address = email or f"nouv-{uuid.uuid4().hex[:8]}@example.test"
    created = client.post(
        "/api/v1/companies/me/driver-invitations",
        json={"first_name": "Nora", "last_name": "Bernard", "email": address},
        headers=_headers(app, owner),
    )
    assert created.status_code == 201, created.get_json()
    return owner, company, address, created


def _token(app) -> str:
    return app.config["_DRIVER_INVITE_LINKS"][-1]["token"]


def _accept(client, token: str, **extra):
    headers = extra.pop("headers", None)
    body = {"token": token, "accept": True, **extra}
    return client.post(
        "/api/v1/driver/invitations/accept",
        json=body,
        headers=headers,
    )


def test_entreprise_non_approuvee_et_acteur_refuse(client, app, invite_tables):
    owner, _company_row = _owner(invite_tables, approved=False)
    denied = client.post(
        "/api/v1/companies/me/driver-invitations",
        json={
            "first_name": "Nora",
            "last_name": "Bernard",
            "email": "nora@example.test",
        },
        headers=_headers(app, owner),
    )
    assert denied.status_code == 403
    assert denied.get_json()["error"] == "entreprise_non_approuvee"
    client_user = _user(invite_tables, UserRole.CLIENT)
    forbidden = client.post(
        "/api/v1/companies/me/driver-invitations",
        json={
            "first_name": "Nora",
            "last_name": "Bernard",
            "email": "nora@example.test",
        },
        headers=_headers(app, client_user),
    )
    assert forbidden.status_code == 403
    stranger = _user(invite_tables, UserRole.COMPANY)
    missing = client.post(
        "/api/v1/companies/me/driver-invitations",
        json={
            "first_name": "Nora",
            "last_name": "Bernard",
            "email": "nora@example.test",
        },
        headers=_headers(app, stranger),
    )
    assert missing.status_code in {403, 404}
    created_for_refused = DriverInvitation.query.filter(
        DriverInvitation.email == "nora@example.test"
    ).count()
    assert created_for_refused == 0


def test_creation_liste_renvoi_annulation_sans_mot_de_passe(client, app, invite_tables):
    owner, company, address, created = _invite(client, app, invite_tables)
    assert created.status_code == 201
    body = created.get_json()
    assert body["status"] == "pending"
    assert body["email"] == address
    assert body["notification"]["delivered"] is False
    assert body["notification"]["state"] == "queued"
    assert "token" not in body
    assert app.config["_DRIVER_INVITE_NOTICES"][-1]["body"].find(_token(app)) == -1
    secret = _token(app)
    refused = client.post(
        "/api/v1/companies/me/driver-invitations",
        json={
            "first_name": "Nora",
            "last_name": "Bernard",
            "email": f"autre-{uuid.uuid4().hex[:6]}@example.test",
            "password": PASSWORD,
        },
        headers=_headers(app, owner),
    )
    assert refused.status_code == 400
    assert refused.get_json()["error"] == "droit_refuse"
    listed = client.get(
        "/api/v1/companies/me/driver-invitations",
        headers=_headers(app, owner),
    )
    assert listed.status_code == 200
    assert listed.get_json()["activation"]["enabled"] is False
    assert listed.get_json()["invitations"][0]["status_label"] == "En attente"
    assert secret not in listed.get_data(as_text=True)
    resent = client.post(
        f"/api/v1/companies/me/driver-invitations/{body['public_id']}/resend",
        headers=_headers(app, owner),
    )
    assert resent.status_code == 200
    assert resent.get_json()["notification"]["delivered"] is False
    assert resent.get_json()["resend_count"] == 1
    stale = client.post(
        "/api/v1/driver/invitations/open",
        json={"token": secret},
    )
    assert stale.status_code == 400
    assert stale.get_json()["error"] == "lien_invalide"
    opened = client.post(
        "/api/v1/driver/invitations/open",
        json={"token": _token(app)},
    )
    assert opened.status_code == 200
    assert opened.get_json()["company_name"] == company.name
    assert opened.get_json()["activation"]["enabled"] is False
    assert "@" in opened.get_json()["email_hint"]
    assert address not in opened.get_json()["email_hint"]
    cancelled = client.post(
        f"/api/v1/companies/me/driver-invitations/{body['public_id']}/cancel",
        headers=_headers(app, owner),
    )
    assert cancelled.status_code == 200
    assert cancelled.get_json()["status"] == "cancelled"
    again = client.post(
        f"/api/v1/companies/me/driver-invitations/{body['public_id']}/cancel",
        headers=_headers(app, owner),
    )
    assert again.status_code == 200
    blocked = _accept(client, _token(app))
    assert blocked.status_code == 400
    assert blocked.get_json()["error"] == "invitation_annulee"


def test_verrou_ferme_ne_cree_ni_chauffeur_ni_habilitation(client, app, invite_tables):
    users = invite_tables.session.query(User).count()
    drivers = invite_tables.session.query(Driver).count()
    grants = invite_tables.session.query(PortalGrant).count()
    _owner_row, _company_row, _address, created = _invite(client, app, invite_tables)
    assert created.status_code == 201
    denied = _accept(client, _token(app), password=PASSWORD)
    assert denied.status_code == 403
    assert denied.get_json()["error"] == "activation_bloquee"
    row = DriverInvitation.query.filter_by(
        public_id=created.get_json()["public_id"]
    ).one()
    assert row.status == "pending"
    assert invite_tables.session.query(User).count() == users + 1
    assert invite_tables.session.query(Driver).count() == drivers
    assert invite_tables.session.query(PortalGrant).count() == grants
    missing = client.post("/api/v1/driver/signup", json={"email": "x@example.test"})
    assert missing.status_code == 404


def test_acceptation_nouveau_chauffeur_ouvre_le_portail(client, app, invite_tables):
    app.config["DRIVER_INVITATION_ACTIVATION"] = True
    app.config["PORTAL_GRANT_ENFORCEMENT"] = True
    users = invite_tables.session.query(User).count()
    _owner_row, company, address, created = _invite(client, app, invite_tables)
    grants = (
        invite_tables.session.query(PortalGrant)
        .filter_by(portal=PortalKind.DRIVER.value)
        .count()
    )
    locks = []
    original = Query.with_for_update

    def spy(self, *args, **kwargs):
        locks.append(True)
        return original(self, *args, **kwargs)

    with patch.object(Query, "with_for_update", spy):
        accepted = _accept(client, _token(app), password=PASSWORD)
    assert accepted.status_code == 200
    assert accepted.get_json()["already_accepted"] is False
    assert accepted.get_json()["portal_open"] is True
    assert locks
    replay = _accept(client, _token(app), password=PASSWORD)
    assert replay.status_code == 200
    assert replay.get_json()["already_accepted"] is True
    user = User.query.filter(User.email == address).one()
    assert user.role == UserRole.CLIENT
    assert invite_tables.session.query(User).count() == users + 2
    drivers = Driver.query.filter_by(user_id=user.id).all()
    assert len(drivers) == 1
    assert drivers[0].is_active is True
    assert drivers[0].is_available is False
    assert int(drivers[0].company_id) == int(company.id)
    assert portal_company_allowed(user, PortalKind.DRIVER.value, int(company.id))
    _other_owner, other_company = _owner(invite_tables)
    assert not portal_company_allowed(
        user, PortalKind.DRIVER.value, int(other_company.id)
    )
    driver_grants = (
        invite_tables.session.query(PortalGrant)
        .filter_by(user_id=user.id, portal=PortalKind.DRIVER.value)
        .all()
    )
    assert len(driver_grants) == 1
    assert driver_grants[0].status == "active"
    assert int(driver_grants[0].company_id) == int(company.id)
    assert (
        invite_tables.session.query(PortalGrant)
        .filter_by(portal=PortalKind.DRIVER.value)
        .count()
        == grants + 1
    )
    profile = client.get(
        "/api/v1/driver/me/profile",
        headers=_headers(app, user),
    )
    assert profile.status_code == 200, profile.get_json()
    revoked = client.post(
        f"/api/v1/companies/me/driver-invitations/{created.get_json()['public_id']}/revoke",
        headers=_headers(app, _owner_row),
    )
    assert revoked.status_code == 200
    assert revoked.get_json()["driver_removed"] is True
    assert Driver.query.filter_by(user_id=user.id).count() == 0
    invite_tables.session.refresh(user)
    assert user.role == UserRole.CLIENT
    second = client.post(
        f"/api/v1/companies/me/driver-invitations/{created.get_json()['public_id']}/revoke",
        headers=_headers(app, _owner_row),
    )
    assert second.status_code == 200
    assert second.get_json()["driver_removed"] is False
    invite_tables.session.expire_all()
    kept_grant = PortalGrant.query.filter_by(
        user_id=user.id, portal=PortalKind.DRIVER.value
    ).one()
    assert kept_grant.status == "revoked"
    assert kept_grant.revoked_at is not None
    closed = client.get("/api/v1/driver/me/profile", headers=_headers(app, user))
    assert closed.status_code == 403


def test_activation_sans_habilitation_annule_l_acceptation(client, app, invite_tables):
    app.config["DRIVER_INVITATION_ACTIVATION"] = True
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False
    _owner_row, _company_row, address, created = _invite(client, app, invite_tables)
    users = invite_tables.session.query(User).count()
    drivers = invite_tables.session.query(Driver).count()
    grants = invite_tables.session.query(PortalGrant).count()
    denied = _accept(client, _token(app), password=PASSWORD)
    assert denied.status_code == 409
    assert denied.get_json()["error"] == "acces_non_ouvert"
    row = DriverInvitation.query.filter_by(
        public_id=created.get_json()["public_id"]
    ).one()
    assert row.status == "pending"
    assert User.query.filter(User.email == address).count() == 0
    assert invite_tables.session.query(User).count() == users
    assert invite_tables.session.query(Driver).count() == drivers
    assert invite_tables.session.query(PortalGrant).count() == grants


def test_compte_existant_exige_la_bonne_session(client, app, invite_tables):
    app.config["DRIVER_INVITATION_ACTIVATION"] = True
    app.config["PORTAL_GRANT_ENFORCEMENT"] = True
    owner, company, address, _created = _invite(client, app, invite_tables)
    account = _user(invite_tables, UserRole.CLIENT)
    account.email = address
    account.set_password("Ancien-Acces-Client-1!", force_change=False)
    digest = account.password
    invite_tables.session.flush()
    anonymous = _accept(client, _token(app))
    assert anonymous.status_code == 401
    assert anonymous.get_json()["error"] == "authentification_requise"
    assert Driver.query.filter_by(user_id=account.id).count() == 0
    other = _user(invite_tables, UserRole.CLIENT)
    other.email = f"autre-{uuid.uuid4().hex[:8]}@example.test"
    invite_tables.session.flush()
    mismatch = _accept(client, _token(app), headers=_headers(app, other))
    assert mismatch.status_code == 403
    assert mismatch.get_json()["error"] == "destinataire_different"
    accepted = _accept(client, _token(app), headers=_headers(app, account))
    assert accepted.status_code == 200
    invite_tables.session.refresh(account)
    assert account.role == UserRole.CLIENT
    assert account.password == digest
    driver = Driver.query.filter_by(user_id=account.id).one()
    assert int(driver.company_id) == int(company.id)
    assert driver.is_active is True
    assert driver.is_available is False
    assert portal_company_allowed(account, PortalKind.DRIVER.value, int(company.id))
    profile = client.get(
        "/api/v1/driver/me/profile",
        headers=_headers(app, account),
    )
    assert profile.status_code == 200, profile.get_json()
    kept = client.post(
        f"/api/v1/companies/me/driver-invitations/{_created.get_json()['public_id']}/revoke",
        headers=_headers(app, owner),
    )
    assert kept.get_json()["driver_removed"] is True
    assert User.query.filter_by(id=account.id).count() == 1


def test_autre_entreprise_et_chauffeur_deja_rattache(client, app, invite_tables):
    app.config["DRIVER_INVITATION_ACTIVATION"] = True
    owner_a, company_a, address, created = _invite(client, app, invite_tables)
    owner_b, company_b = _owner(invite_tables)
    hidden = client.get(
        "/api/v1/companies/me/driver-invitations",
        headers=_headers(app, owner_b),
    )
    assert hidden.status_code == 200
    assert hidden.get_json()["invitations"] == []
    foreign = client.post(
        f"/api/v1/companies/me/driver-invitations/{created.get_json()['public_id']}/cancel",
        headers=_headers(app, owner_b),
    )
    assert foreign.status_code == 404
    busy = _user(invite_tables, UserRole.DRIVER)
    busy.email = address
    current = Driver()
    current.user_id = busy.id
    current.company_id = company_b.id
    current.driver_type = DriverType.REGULAR
    current.is_active = True
    invite_tables.session.add(current)
    invite_tables.session.flush()
    conflict = _accept(client, _token(app), headers=_headers(app, busy))
    assert conflict.status_code == 409
    assert conflict.get_json()["error"] == "deja_rattache"
    invite_tables.session.refresh(current)
    assert int(current.company_id) == int(company_b.id)
    assert current.is_active is True
    assert busy.role == UserRole.DRIVER
    same = _user(invite_tables, UserRole.DRIVER)
    same_address = f"equipe-{uuid.uuid4().hex[:8]}@example.test"
    same.email = same_address
    existing = Driver()
    existing.user_id = same.id
    existing.company_id = company_a.id
    existing.driver_type = DriverType.REGULAR
    existing.is_active = True
    invite_tables.session.add(existing)
    invite_tables.session.flush()
    second = client.post(
        "/api/v1/companies/me/driver-invitations",
        json={"first_name": "Eli", "last_name": "Martin", "email": same_address},
        headers=_headers(app, owner_a),
    )
    assert second.status_code == 201
    joined = _accept(
        client,
        app.config["_DRIVER_INVITE_LINKS"][-1]["token"],
        headers=_headers(app, same),
    )
    assert joined.status_code == 200
    assert Driver.query.filter_by(user_id=same.id).count() == 1
    invite_tables.session.refresh(existing)
    assert existing.is_active is True
    assert int(existing.company_id) == int(company_a.id)
    kept = client.post(
        f"/api/v1/companies/me/driver-invitations/{second.get_json()['public_id']}/revoke",
        headers=_headers(app, owner_a),
    )
    assert kept.get_json()["driver_removed"] is False
    assert Driver.query.filter_by(user_id=same.id).count() == 1
    assert (
        PortalGrant.query.filter_by(
            user_id=same.id, portal=PortalKind.DRIVER.value
        ).count()
        == 0
    )


def test_creation_directe_refusee_et_chauffeur_existant_conserve(
    client, app, invite_tables
):
    owner, company = _owner(invite_tables)
    driver_user = _user(invite_tables, UserRole.DRIVER)
    driver_user.email = f"actif-{uuid.uuid4().hex[:8]}@example.test"
    current = Driver()
    current.user_id = driver_user.id
    current.company_id = company.id
    current.driver_type = DriverType.REGULAR
    current.is_active = True
    current.is_available = True
    invite_tables.session.add(current)
    invite_tables.session.flush()
    before = Driver.query.count()
    created = client.post(
        "/api/v1/companies/me/drivers/create",
        json={
            "username": f"neuf-{uuid.uuid4().hex[:8]}",
            "first_name": "Neuf",
            "last_name": "Chauffeur",
            "email": f"neuf-{uuid.uuid4().hex[:8]}@example.test",
            "password": PASSWORD,
        },
        headers=_headers(app, owner),
    )
    assert created.status_code == 403
    assert created.get_json()["error"] == "invitation_requise"
    assert Driver.query.count() == before
    profile = client.get(
        "/api/v1/driver/me/profile",
        headers=_headers(app, driver_user),
    )
    assert profile.status_code == 200, profile.get_json()
    invite_tables.session.refresh(current)
    assert current.is_active is True
    assert int(current.company_id) == int(company.id)


def test_lien_expire_tentatives_et_renvois_limites(client, app, invite_tables):
    app.config["DRIVER_INVITATION_ACTIVATION"] = True
    _owner_row, _company_row, _address, created = _invite(client, app, invite_tables)
    row = DriverInvitation.query.filter_by(
        public_id=created.get_json()["public_id"]
    ).one()
    row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    invite_tables.session.flush()
    expired = _accept(client, _token(app), password=PASSWORD)
    assert expired.status_code == 400
    assert expired.get_json()["error"] == "lien_expire"
    invite_tables.session.refresh(row)
    assert row.status == "expired"
    owner = _owner_row
    fresh = client.post(
        "/api/v1/companies/me/driver-invitations",
        json={
            "first_name": "Lina",
            "last_name": "Costa",
            "email": f"lina-{uuid.uuid4().hex[:6]}@example.test",
        },
        headers=_headers(app, owner),
    )
    assert fresh.status_code == 201
    public_id = fresh.get_json()["public_id"]
    token = _token(app)
    for _ in range(8):
        attempt = _accept(client, token, password="court")
        assert attempt.status_code in {400, 429}
    blocked = _accept(client, token, password=PASSWORD)
    assert blocked.status_code == 429
    assert blocked.get_json()["error"] == "tentatives_limitees"
    assert User.query.filter_by(email=fresh.get_json()["email"]).count() == 0
    for _ in range(5):
        assert (
            client.post(
                f"/api/v1/companies/me/driver-invitations/{public_id}/resend",
                headers=_headers(app, owner),
            ).status_code
            == 200
        )
    limited = client.post(
        f"/api/v1/companies/me/driver-invitations/{public_id}/resend",
        headers=_headers(app, owner),
    )
    assert limited.status_code == 429
    assert limited.get_json()["error"] == "renvoi_limite"


def test_envoi_retient_un_destinataire_hors_perimetre(app, invite_tables):
    _ = invite_tables
    with app.app_context():
        app.config["MAIL_SUPPRESS_SEND"] = False
        with patch("services.driver_invitations.mail.send") as send:
            withheld = _hand_off(
                "personne@gmail.com",
                "driver_invitation",
                "corps",
                "secret-non-journalise",
                "Transports du Lac",
            )
        assert withheld["state"] == "withheld"
        assert withheld["delivered"] is False
        send.assert_not_called()
        app.config["MAIL_SUPPRESS_SEND"] = True
        with patch("services.driver_invitations.mail.send") as send:
            queued = _hand_off(
                "nora@example.test",
                "driver_invitation",
                "corps",
                "secret-non-journalise",
                "Transports du Lac",
            )
        assert queued["state"] == "queued"
        assert queued["delivered"] is False
        send.assert_not_called()
