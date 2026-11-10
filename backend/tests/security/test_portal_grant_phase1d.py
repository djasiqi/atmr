"""Phase 1D — parcours réels, socket déjà ouverte, repli certifié."""

from __future__ import annotations

import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import pytest
from flask_jwt_extended import create_access_token
from werkzeug.exceptions import Forbidden

from ext import socketio
from models import Driver, Institution, UserRole
from models.enums import DriverType, PortalGrantStatus, PortalKind
from models.portal_grant import PortalGrant, RevokedPortalGrantProtected
from security.portal_grant_access import (
    adjust_refresh_access_claims,
    revoke_portal_grant,
)
from security.portal_scope import legacy_fallback_eligible, portal_company_allowed
from tests.security.test_portal_grant_phase1 import (
    _client_profile,
    _company,
    _grant,
    _require_company,
    _user,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def portal_grant_table(db):
    PortalGrant.__table__.create(bind=db.engine, checkfirst=True)
    return db


@pytest.fixture
def enforce_grants(app):
    had_key = "PORTAL_GRANT_ENFORCEMENT" in app.config
    previous = app.config.get("PORTAL_GRANT_ENFORCEMENT")
    app.config["PORTAL_GRANT_ENFORCEMENT"] = True
    yield
    if had_key:
        app.config["PORTAL_GRANT_ENFORCEMENT"] = previous
    else:
        app.config.pop("PORTAL_GRANT_ENFORCEMENT", None)


def _token(app, user, **claims) -> str:
    payload = {"role": user.role.value, "aud": "atmr-api"}
    payload.update(claims)
    with app.app_context():
        return create_access_token(
            identity=str(user.public_id),
            fresh=True,
            additional_claims=payload,
        )


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_parcours_http_drapeau_eteint(app, portal_grant_table):
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    token = _token(app, user, company_id=company.id)
    client = app.test_client()
    me = client.get("/api/v1/companies/me", headers=_auth(token))
    assert me.status_code == 200
    assert me.get_json()["data"]["id"] == company.id
    drivers = client.get("/api/v1/companies/me/drivers", headers=_auth(token))
    assert drivers.status_code == 200
    other_owner = _user(portal_grant_table, UserRole.COMPANY)
    other = _company(portal_grant_table, other_owner, approved=True)
    invoices = client.get(
        f"/api/v1/invoices/companies/{other.id}/invoices",
        headers=_auth(token),
    )
    assert invoices.status_code == 403
    logout = client.post("/api/v1/auth/logout", headers=_auth(token), json={})
    assert logout.status_code == 200


def test_client_et_entreprise_sur_la_meme_identite(app, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.CLIENT)
    _client_profile(portal_grant_table, user)
    company = _company(portal_grant_table, user, approved=True)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    http = app.test_client()
    client_token = _token(app, user, role="CLIENT")
    company_token = _token(app, user, role="COMPANY", company_id=company.id)
    client_me = http.get("/api/v1/clients/me", headers=_auth(client_token))
    company_me = http.get("/api/v1/companies/me", headers=_auth(company_token))
    back = http.get("/api/v1/clients/me", headers=_auth(client_token))
    assert client_me.status_code == 200
    assert company_me.status_code == 200
    assert company_me.get_json()["data"]["id"] == company.id
    assert back.status_code == 200
    assert user.role == UserRole.CLIENT


def test_chauffeur_et_institution(app, portal_grant_table, enforce_grants):
    owner = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, owner, approved=True)
    driver_user = _user(portal_grant_table, UserRole.DRIVER)
    driver = Driver()
    driver.user_id = driver_user.id
    driver.company_id = company.id
    driver.driver_type = DriverType.REGULAR
    portal_grant_table.session.add(driver)
    portal_grant_table.session.flush()
    _grant(
        portal_grant_table,
        driver_user,
        PortalKind.DRIVER.value,
        company_id=company.id,
    )
    assert portal_company_allowed(driver_user, "driver", company.id) is True
    other = _company(portal_grant_table, _user(portal_grant_table, UserRole.COMPANY), approved=True)
    assert portal_company_allowed(driver_user, "driver", other.id) is False

    institution_user = _user(portal_grant_table, UserRole.INSTITUTION)
    institution = Institution()
    institution.name = "Clinique locale"
    institution.public_id = str(uuid.uuid4())
    portal_grant_table.session.add(institution)
    portal_grant_table.session.flush()
    institution_user.institution_id = institution.id
    _grant(
        portal_grant_table,
        institution_user,
        PortalKind.INSTITUTION.value,
        institution_id=institution.id,
        status=PortalGrantStatus.PENDING_ACTIVATION.value,
        internal_role="institution_admin",
    )
    from security.portal_scope import resolve_portal_scope

    pending = resolve_portal_scope(
        institution_user, "institution", use_request_claim=False
    )
    assert pending.allowed is False
    assert pending.reason == "pending"
    active = _grant(
        portal_grant_table,
        _user(portal_grant_table, UserRole.INSTITUTION),
        PortalKind.INSTITUTION.value,
        institution_id=institution.id,
        internal_role="institution_admin",
    )
    opened = resolve_portal_scope(active.user, "institution", use_request_claim=False)
    assert opened.allowed is True
    assert opened.institution.id == institution.id


def test_date_seule_ne_donne_pas_acces(app, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.COMPANY)
    user.created_at = datetime.now(UTC) - timedelta(days=30)
    _company(portal_grant_table, user, approved=True)
    app.config["PORTAL_GRANT_LEGACY_CUTOFF"] = datetime.now(UTC).isoformat()
    app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)
    assert legacy_fallback_eligible(user, PortalKind.COMPANY.value) is False
    with pytest.raises(Forbidden):
        _require_company(app, user, context=None, claims={"role": "COMPANY"})
    refused = app.test_client().get(
        "/api/v1/companies/me",
        headers=_auth(_token(app, user, role="COMPANY")),
    )
    assert refused.status_code == 403
    app.config["PORTAL_GRANT_LEGACY_CERTIFIED"] = f"{user.id}:company"
    resolved, _same = _require_company(
        app, user, context=None, claims={"role": "COMPANY"}
    )
    assert resolved is not None
    created_after = _user(portal_grant_table, UserRole.COMPANY)
    created_after.created_at = datetime.now(UTC)
    _company(portal_grant_table, created_after, approved=True)
    app.config["PORTAL_GRANT_LEGACY_CERTIFIED"] = f"{created_after.id}:company"
    assert legacy_fallback_eligible(created_after, PortalKind.COMPANY.value) is False
    app.config.pop("PORTAL_GRANT_LEGACY_CUTOFF", None)
    app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)


def test_effacement_d_une_revocation_ne_rouvre_pas(app, portal_grant_table):
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    grant = _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    revoke_portal_grant(grant)
    portal_grant_table.session.flush()

    def _delete_revoked() -> None:
        with portal_grant_table.session.begin_nested():
            portal_grant_table.session.delete(grant)
            portal_grant_table.session.flush()

    with pytest.raises(RevokedPortalGrantProtected):
        _delete_revoked()
    portal_grant_table.session.refresh(grant)
    assert grant.status == PortalGrantStatus.REVOKED.value
    assert portal_company_allowed(user, "company", company.id) is False
    claims = adjust_refresh_access_claims(
        user,
        {"role": "COMPANY", "company_id": company.id, "aud": "atmr-api"},
        active_context_id=f"company:{company.id}",
    )
    assert claims is None or claims.get("role") != "COMPANY"


def test_observation_ne_change_pas_la_reponse(app, portal_grant_table, caplog):
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False
    app.config["PORTAL_GRANT_OBSERVATION"] = True
    app.config["PORTAL_GRANT_LEGACY_CUTOFF"] = datetime.now(UTC).isoformat()
    app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)
    token = _token(app, user, company_id=company.id)
    with caplog.at_level("INFO", logger="security.portal_scope"):
        response = app.test_client().get(
            "/api/v1/companies/me",
            headers=_auth(token),
        )
    assert response.status_code == 200
    assert response.get_json()["data"]["id"] == company.id
    assert any("portal_scope_divergence" in message for message in caplog.messages)
    assert not any("Bearer" in message or user.email in message for message in caplog.messages)
    app.config["PORTAL_GRANT_OBSERVATION"] = False
    app.config.pop("PORTAL_GRANT_LEGACY_CUTOFF", None)


def test_socket_reelle_coupee_a_la_revocation(app, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    other = _company(
        portal_grant_table,
        _user(portal_grant_table, UserRole.COMPANY),
        approved=True,
    )
    grant = _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    token = _token(app, user, role="COMPANY", company_id=company.id)
    client = socketio.test_client(
        app,
        namespace="/",
        auth={"token": token},
        headers=_auth(token),
    )
    try:
        assert client.is_connected()
        client.emit("join_company", {"company_id": other.id})
        joined = client.get_received()
        assert any(event.get("name") == "joined_company" for event in joined)
        joined_payload = next(
            event["args"][0]
            for event in joined
            if event.get("name") == "joined_company"
        )
        assert joined_payload["company_id"] == company.id
        assert joined_payload["company_id"] != other.id

        socketio.emit(
            "portal_probe",
            {"marker": "avant"},
            to=f"company_{company.id}",
            namespace="/",
        )
        before = client.get_received()
        assert any(event.get("name") == "portal_probe" for event in before)

        client.emit("team_chat_message", {"content": "sonde-locale-phase-1d"})
        sent = client.get_received()
        assert any(event.get("name") == "team_chat_message" for event in sent)
        assert client.is_connected()

        revoke_portal_grant(grant)
        portal_grant_table.session.flush()
        assert client.is_connected() is False

        socketio.emit(
            "portal_probe",
            {"marker": "apres"},
            to=f"company_{company.id}",
            namespace="/",
        )
        with pytest.raises(RuntimeError, match="not connected"):
            client.get_received()
        with pytest.raises(RuntimeError, match="not connected"):
            client.emit("team_chat_message", {"content": "ne-doit-pas-partir"})
        with pytest.raises(RuntimeError, match="not connected"):
            client.emit("join_company", {"company_id": other.id})

        again = socketio.test_client(
            app,
            namespace="/",
            auth={"token": token},
            headers=_auth(token),
        )
        assert again.is_connected() is False
    finally:
        with suppress(Exception):
            client.disconnect()
