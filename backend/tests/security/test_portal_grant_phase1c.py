"""Phase 1C — périmètre d'autorisation, drapeau éteint par défaut."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from werkzeug.exceptions import Forbidden

from application.users.get_current_company import GetCurrentCompanyUseCase
from models import UserRole
from models.enums import PortalGrantStatus, PortalKind
from models.portal_grant import PortalGrant
from routes.company_mobile_auth import _find_company_user_by_email
from routes.conversations import _resolve_conversation_actor
from security.portal_grant_access import adjust_refresh_access_claims
from security.portal_scope import (
    legacy_fallback_eligible,
    portal_company_allowed,
    resolve_portal_scope,
)
from tests.security.test_portal_grant_phase1 import (
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


def _legacy_company(app, db, *, created_before_cutoff: bool):
    user = _user(db, UserRole.COMPANY)
    now = datetime.now(UTC)
    user.created_at = now - timedelta(days=30 if created_before_cutoff else 0)
    company = _company(db, user, approved=True)
    app.config["PORTAL_GRANT_LEGACY_CUTOFF"] = (now - timedelta(days=1)).isoformat()
    app.config["PORTAL_GRANT_ENFORCEMENT"] = True
    if created_before_cutoff:
        app.config["PORTAL_GRANT_LEGACY_CERTIFIED"] = f"{user.id}:company"
    else:
        app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)
    return user, company


def test_client_seul_et_client_avec_entreprise(app, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.CLIENT)
    with pytest.raises(Forbidden):
        _require_company(app, user, context=None, claims={"role": "CLIENT"})
    company = _company(portal_grant_table, user, approved=True)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    resolved, same = _require_company(
        app,
        user,
        context=f"company:{company.id}",
        claims={"role": "COMPANY", "company_id": company.id},
    )
    assert resolved.id == company.id
    assert same.role == UserRole.CLIENT
    assert legacy_fallback_eligible(user, PortalKind.CLIENT.value) is False
    client_scope = resolve_portal_scope(
        user, PortalKind.CLIENT.value, use_request_claim=False
    )
    assert client_scope.allowed is True


def test_nouveau_compte_sans_repli_professionnel(app, portal_grant_table):
    user, _company_row = _legacy_company(
        app, portal_grant_table, created_before_cutoff=False
    )
    with pytest.raises(Forbidden):
        _require_company(app, user, context=None, claims={"role": "COMPANY"})
    app.config.pop("PORTAL_GRANT_LEGACY_CUTOFF", None)
    app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False


def test_ancien_compte_eligible_au_repli(app, portal_grant_table):
    user, company = _legacy_company(
        app, portal_grant_table, created_before_cutoff=True
    )
    resolved, _same = _require_company(
        app, user, context=None, claims={"role": "COMPANY"}
    )
    assert resolved.id == company.id
    assert legacy_fallback_eligible(user, PortalKind.COMPANY.value) is True
    app.config.pop("PORTAL_GRANT_LEGACY_CUTOFF", None)
    app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False


def test_entreprise_non_approuvee_en_repli(app, portal_grant_table):
    user = _user(portal_grant_table, UserRole.COMPANY)
    now = datetime.now(UTC)
    user.created_at = now - timedelta(days=30)
    company = _company(portal_grant_table, user, approved=False)
    app.config["PORTAL_GRANT_LEGACY_CUTOFF"] = now.isoformat()
    app.config["PORTAL_GRANT_LEGACY_CERTIFIED"] = f"{user.id}:company"
    app.config["PORTAL_GRANT_ENFORCEMENT"] = True
    with pytest.raises(Forbidden):
        _require_company(
            app,
            user,
            context=f"company:{company.id}",
            claims={"role": "COMPANY", "company_id": company.id},
        )
    app.config.pop("PORTAL_GRANT_LEGACY_CUTOFF", None)
    app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False


def test_jwt_entreprise_differente_et_lien_ambigu(app, portal_grant_table):
    user = _user(portal_grant_table, UserRole.COMPANY)
    own = _company(portal_grant_table, user, approved=True)
    other_owner = _user(portal_grant_table, UserRole.COMPANY)
    other = _company(portal_grant_table, other_owner, approved=True)
    use_case = GetCurrentCompanyUseCase(get_current_user_fn=lambda: user)
    with app.test_request_context("/api/v1/companies/me"):
        from unittest.mock import patch

        with patch(
            "security.portal_scope._jwt_claims",
            return_value={"role": "COMPANY", "company_id": other.id},
        ):
            mismatch = use_case.execute()
    assert mismatch.found is False
    assert mismatch.status_code == 403
    _company(portal_grant_table, user, approved=True)
    with app.test_request_context("/api/v1/companies/me"):
        from unittest.mock import patch

        with patch("security.portal_scope._jwt_claims", return_value={"role": "COMPANY"}):
            ambiguous = use_case.execute()
    assert ambiguous.found is False
    assert ambiguous.status_code == 403
    assert own.id != other.id


def test_revocation_survit_au_drapeau_eteint(app, portal_grant_table):
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    grant = _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
        status=PortalGrantStatus.REVOKED.value,
    )
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False
    with pytest.raises(Forbidden):
        _require_company(
            app,
            user,
            context=f"company:{company.id}",
            claims={"role": "COMPANY", "company_id": company.id},
        )
    refreshed = adjust_refresh_access_claims(
        user,
        {"role": "COMPANY", "company_id": company.id},
        active_context_id=f"company:{company.id}",
    )
    assert refreshed is None
    assert grant.status == PortalGrantStatus.REVOKED.value
    assert portal_company_allowed(user, "company", company.id) is False


def test_connexion_mobile_entreprise_suit_le_perimetre(app, portal_grant_table):
    client = _user(portal_grant_table, UserRole.CLIENT)
    company = _company(portal_grant_table, client, approved=True)
    _grant(
        portal_grant_table,
        client,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False
    hidden_user, hidden_company = _find_company_user_by_email(client.email)
    assert hidden_user is None
    assert hidden_company is None
    app.config["PORTAL_GRANT_ENFORCEMENT"] = True
    found_user, found_company = _find_company_user_by_email(client.email)
    assert found_user is not None
    assert found_company is not None
    assert found_company.id == company.id
    assert found_user.role == UserRole.CLIENT
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False


def test_conversation_autre_organisation_refusee(app, portal_grant_table):
    user = _user(portal_grant_table, UserRole.COMPANY)
    _company(portal_grant_table, user, approved=True)
    _company(portal_grant_table, user, approved=True)
    with app.test_request_context("/api/v1/conversations/inbox"):
        _actor, error = _resolve_conversation_actor(user)
    assert error is not None
    assert error[1] == 403


def test_observation_ne_change_pas_la_reponse(
    app, portal_grant_table, caplog: pytest.LogCaptureFixture
):
    user = _user(portal_grant_table, UserRole.CLIENT)
    company = _company(portal_grant_table, user, approved=True)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    app.config["PORTAL_GRANT_ENFORCEMENT"] = False
    app.config["PORTAL_GRANT_OBSERVATION"] = True
    with (
        caplog.at_level(logging.INFO, logger="security.portal_scope"),
        pytest.raises(Forbidden),
    ):
        _require_company(
            app,
            user,
            context=f"company:{company.id}",
            claims={"role": "COMPANY", "company_id": company.id},
        )
    assert any("portal_scope_divergence" in message for message in caplog.messages)
    app.config["PORTAL_GRANT_OBSERVATION"] = False


def test_institution_en_attente_conserve_le_client(app, portal_grant_table, enforce_grants):
    from models import Institution

    user = _user(portal_grant_table, UserRole.CLIENT)
    institution = Institution()
    institution.name = f"Clinique {uuid.uuid4().hex[:6]}"
    institution.public_id = str(uuid.uuid4())
    portal_grant_table.session.add(institution)
    portal_grant_table.session.flush()
    _grant(
        portal_grant_table,
        user,
        PortalKind.INSTITUTION.value,
        institution_id=institution.id,
        status=PortalGrantStatus.PENDING_ACTIVATION.value,
    )
    assert legacy_fallback_eligible(user, PortalKind.CLIENT.value) is True
    client_scope = resolve_portal_scope(user, PortalKind.CLIENT.value)
    institution_scope = resolve_portal_scope(
        user,
        PortalKind.INSTITUTION.value,
        claimed_institution_id=institution.id,
        use_request_claim=False,
    )
    assert client_scope.allowed is True
    assert institution_scope.allowed is False
    assert institution_scope.reason == "pending"
