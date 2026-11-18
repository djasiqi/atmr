"""Phase 1 — habilitations multiportails, drapeau éteint par défaut."""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from flask_jwt_extended import create_access_token
from werkzeug.exceptions import Forbidden

from ext import role_required
from models import Company, Driver, Institution, User, UserRole
from models.client import Client
from models.enums import DriverType, PortalGrantStatus, PortalKind
from models.portal_grant import PortalGrant
from security.authorization import AuthorizationService
from security.portal_grant_access import (
    adjust_refresh_access_claims,
    prepare_legacy_portal_grants,
    revoke_portal_grant,
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


def _user(db, role: UserRole) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"grant_{suffix}"
    user.email = f"grant-{suffix}@example.com"
    user.role = role
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def _company(db, owner: User, *, approved: bool) -> Company:
    suffix = uuid.uuid4().hex[:8]
    company = Company()
    company.name = f"Transport {suffix}"
    company.address = "Rue du Test 1"
    company.contact_phone = "0210000000"
    company.contact_email = f"co-{suffix}@example.com"
    company.user_id = owner.id
    company.is_approved = approved
    db.session.add(company)
    db.session.flush()
    return company


def _client_profile(db, user: User) -> Client:
    profile = Client()
    profile.user_id = user.id
    db.session.add(profile)
    db.session.flush()
    return profile


def _grant(
    db,
    user: User,
    portal: str,
    *,
    company_id: int | None = None,
    institution_id: int | None = None,
    status: str = PortalGrantStatus.ACTIVE.value,
    internal_role: str | None = None,
) -> PortalGrant:
    row = PortalGrant(
        user_id=user.id,
        portal=portal,
        company_id=company_id,
        institution_id=institution_id,
        internal_role=internal_role,
        status=status,
    )
    if status == PortalGrantStatus.REVOKED.value:
        from datetime import UTC, datetime

        row.revoked_at = datetime.now(UTC)
    db.session.add(row)
    db.session.flush()
    return row


def _headers(app, user: User, role: str | None = None) -> dict[str, str]:
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            fresh=True,
            additional_claims={
                "role": role or user.role.value,
                "aud": "atmr-api",
            },
        )
    return {"Authorization": f"Bearer {token}"}


def _require_company(app, user: User, *, context: str | None, claims: dict):
    headers = {}
    if context:
        headers["X-Active-Context-Id"] = context
    with (
        app.test_request_context("/api/v1/companies/me", headers=headers),
        patch(
            "security.authorization.get_jwt_identity",
            return_value=str(user.public_id),
        ),
        patch("flask_jwt_extended.get_jwt", return_value=claims),
    ):
        return AuthorizationService.require_company()


def test_client_existant_sans_droit_professionnel(
    app, portal_grant_table, enforce_grants
):
    user = _user(portal_grant_table, UserRole.CLIENT)
    _client_profile(portal_grant_table, user)
    with pytest.raises(Forbidden):
        _require_company(app, user, context=None, claims={"role": "CLIENT"})
    assert user.role == UserRole.CLIENT


def test_drapeau_eteint_ignore_un_droit_entreprise(app, portal_grant_table):
    user = _user(portal_grant_table, UserRole.CLIENT)
    company = _company(portal_grant_table, user, approved=True)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    with pytest.raises(Forbidden):
        _require_company(
            app,
            user,
            context=f"company:{company.id}",
            claims={"role": "COMPANY"},
        )


def test_client_avec_droit_entreprise_conserve_son_role(
    app, portal_grant_table, enforce_grants
):
    user = _user(portal_grant_table, UserRole.CLIENT)
    _client_profile(portal_grant_table, user)
    company = _company(portal_grant_table, user, approved=True)
    shell = _company(portal_grant_table, user, approved=True)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    resolved, _same = _require_company(
        app,
        user,
        context=f"company:{company.id}",
        claims={"role": "CLIENT"},
    )
    assert resolved.id == company.id
    assert resolved.id != shell.id
    assert user.role == UserRole.CLIENT


def test_bascule_client_entreprise_client(
    client, app, portal_grant_table, enforce_grants, monkeypatch
):
    import routes.auth as auth_routes

    user = _user(portal_grant_table, UserRole.CLIENT)
    company = _company(portal_grant_table, user, approved=True)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    saved: dict[str, str] = {}
    monkeypatch.setattr(auth_routes, "_load_user_for_bootstrap", lambda _pid: user)
    monkeypatch.setattr(auth_routes, "_prepare_user_for_bootstrap", lambda item: item)
    monkeypatch.setattr(
        auth_routes,
        "_save_active_context",
        lambda public_id, context_id: saved.__setitem__(public_id, context_id),
    )
    monkeypatch.setattr(
        auth_routes,
        "_get_saved_active_context",
        lambda public_id: saved.get(public_id),
    )
    headers = _headers(app, user)
    to_company = client.post(
        "/api/v1/auth/switch-context",
        json={"target_context_id": f"company:{company.id}"},
        headers=headers,
    )
    assert to_company.status_code == 200
    back = client.post(
        "/api/v1/auth/switch-context",
        json={"target_context_id": "client:self"},
        headers=headers,
    )
    assert back.status_code == 200
    assert user.role == UserRole.CLIENT


def test_entreprise_non_approuvee(app, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.CLIENT)
    company = _company(portal_grant_table, user, approved=False)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    with pytest.raises(Forbidden):
        _require_company(
            app,
            user,
            context=f"company:{company.id}",
            claims={"role": "CLIENT"},
        )


def test_institution_en_attente_activation(app, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.CLIENT)
    institution = Institution()
    institution.name = "Clinique test"
    institution.public_id = str(uuid.uuid4())
    portal_grant_table.session.add(institution)
    portal_grant_table.session.flush()
    _grant(
        portal_grant_table,
        user,
        PortalKind.INSTITUTION.value,
        institution_id=institution.id,
        status=PortalGrantStatus.PENDING_ACTIVATION.value,
        internal_role="institution_admin",
    )
    with (
        app.test_request_context(
            headers={"X-Active-Context-Id": f"institution:{institution.id}"}
        ),
        patch(
            "security.authorization.get_jwt_identity",
            return_value=str(user.public_id),
        ),
        patch("flask_jwt_extended.get_jwt", return_value={"role": "INSTITUTION"}),
        pytest.raises(Forbidden),
    ):
        AuthorizationService.require_institution()
    assert user.role == UserRole.CLIENT


def test_chauffeur_lie_a_une_entreprise(app, portal_grant_table, enforce_grants):
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
    with pytest.raises(Forbidden):
        _require_company(
            app,
            driver_user,
            context=f"company:{company.id}",
            claims={"role": "DRIVER"},
        )
    with (
        app.test_request_context(
            headers={"X-Active-Context-Id": f"driver:{driver.id}"}
        ),
        patch(
            "security.authorization.get_jwt_identity",
            return_value=str(driver_user.public_id),
        ),
        patch("flask_jwt_extended.get_jwt", return_value={"role": "DRIVER"}),
    ):
        resolved, _user_out = AuthorizationService.require_driver()
    assert resolved.company_id == company.id


def test_droit_revoque_pendant_la_session(app, portal_grant_table, enforce_grants):
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

    @role_required(UserRole.company)
    def protected():
        return {"ok": True}, 200

    with (
        app.test_request_context(
            headers={"X-Active-Context-Id": f"company:{company.id}"}
        ),
        patch("ext.get_jwt_identity", return_value=str(user.public_id)),
        patch(
            "flask_jwt_extended.get_jwt",
            return_value={"role": "COMPANY", "company_id": company.id},
        ),
        pytest.raises(Forbidden),
    ):
        protected()
    refreshed = adjust_refresh_access_claims(
        user,
        {"role": user.role.value, "company_id": company.id},
        active_context_id=f"company:{company.id}",
    )
    assert refreshed is None


def test_claim_forge_ou_perime(app, client, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.CLIENT)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    company = _company(portal_grant_table, user, approved=True)

    @role_required(UserRole.company)
    def protected():
        return {"ok": True}, 200

    with (
        app.test_request_context("/api/v1/companies/me"),
        patch("ext.get_jwt_identity", return_value=str(user.public_id)),
        patch(
            "flask_jwt_extended.get_jwt",
            return_value={"role": "COMPANY", "company_id": company.id},
        ),
        pytest.raises(Forbidden),
    ):
        protected()

    with app.app_context():
        expired = create_access_token(
            identity=str(user.public_id),
            expires_delta=timedelta(minutes=-5),
            additional_claims={"role": "CLIENT", "aud": "atmr-api"},
        )
    response = client.post(
        "/api/v1/auth/switch-context",
        json={"target_context_id": "client:self"},
        headers={"Authorization": f"Bearer {expired}"},
    )
    assert response.status_code == 401


def test_isolation_autre_entreprise(portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.CLIENT)
    own = _company(portal_grant_table, user, approved=True)
    other_owner = _user(portal_grant_table, UserRole.COMPANY)
    other = _company(portal_grant_table, other_owner, approved=True)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=own.id,
    )
    booking = SimpleNamespace(id=1, company_id=other.id, client_id=None)
    allowed, error = AuthorizationService.check_booking_ownership(booking, user)
    assert allowed is False
    assert error is not None
    resource_ok, resource_error = AuthorizationService.check_company_resource_access(
        other.id, user
    )
    assert resource_ok is False
    assert resource_error is not None


def test_comptes_historiques_sans_droit(app, portal_grant_table, enforce_grants):
    from datetime import UTC, datetime, timedelta

    user = _user(portal_grant_table, UserRole.COMPANY)
    user.created_at = datetime.now(UTC) - timedelta(days=30)
    company = _company(portal_grant_table, user, approved=True)
    app.config["PORTAL_GRANT_LEGACY_CUTOFF"] = datetime.now(UTC).isoformat()
    app.config["PORTAL_GRANT_LEGACY_CERTIFIED"] = f"{user.id}:company"
    resolved, _same = _require_company(
        app, user, context=None, claims={"role": "COMPANY"}
    )
    assert resolved.id == company.id
    app.config.pop("PORTAL_GRANT_LEGACY_CUTOFF", None)
    app.config.pop("PORTAL_GRANT_LEGACY_CERTIFIED", None)


def test_rattrapage_preserve_le_role_et_signale_les_ambiguites(
    app, portal_grant_table, enforce_grants
):
    historical = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, historical, approved=True)
    ambiguous = _user(portal_grant_table, UserRole.CLIENT)
    _client_profile(portal_grant_table, ambiguous)
    _company(portal_grant_table, ambiguous, approved=True)
    _company(portal_grant_table, ambiguous, approved=True)
    before = historical.role
    report = prepare_legacy_portal_grants(
        apply=True,
        user_ids=[historical.id, ambiguous.id],
    )
    assert historical.role == before == UserRole.COMPANY
    assert ambiguous.role == UserRole.CLIENT
    assert any(
        row["reason"] == "plusieurs_societes_meme_utilisateur"
        for row in report["ambiguous"]
    )
    assert not any(
        row["user_id"] == ambiguous.id and row["portal"] == PortalKind.COMPANY.value
        for row in report["created"]
    )
    resolved, _same = _require_company(
        app,
        historical,
        context=None,
        claims={"role": "COMPANY", "company_id": company.id},
    )
    assert resolved.id == company.id
    second = prepare_legacy_portal_grants(apply=True, user_ids=[historical.id])
    assert any(
        row["user_id"] == historical.id and row["portal"] == PortalKind.COMPANY.value
        for row in second["skipped_existing"]
    )


def test_droit_revoque_non_restaure_par_le_rattrapage(portal_grant_table):
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
    prepare_legacy_portal_grants(apply=True, user_ids=[user.id])
    portal_grant_table.session.refresh(grant)
    assert grant.status == PortalGrantStatus.REVOKED.value


def test_rafraichissement_ne_remet_pas_un_role_revoque(
    portal_grant_table, enforce_grants
):
    user = _user(portal_grant_table, UserRole.CLIENT)
    company = _company(portal_grant_table, user, approved=True)
    _grant(portal_grant_table, user, PortalKind.CLIENT.value)
    company_grant = _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    revoke_portal_grant(company_grant)
    portal_grant_table.session.flush()
    adjusted = adjust_refresh_access_claims(
        user,
        {"role": "COMPANY", "company_id": company.id},
        active_context_id=None,
    )
    assert adjusted is not None
    assert adjusted["role"] == "CLIENT"
    assert "company_id" not in adjusted


def test_migration_portail_reversible():
    versions = Path(__file__).resolve().parents[2] / "migrations" / "versions"
    matches = [
        path
        for path in versions.glob("*.py")
        if "portal_grant" in path.read_text(encoding="utf-8")
    ]
    assert matches, "migration portal_grant absente"
    source = matches[-1].read_text(encoding="utf-8")
    assert "create_table" in source
    assert "drop_table" in source
    assert "portal_grant" in source
