"""
Tests de contrôle d'accès et multi-tenant pour les endpoints push-privacy.

Preuve (via tests d'intégration avec PostgreSQL) :
- /company-settings/push-privacy ne modifie que le user lié à la company courante (token).
  Gardes : @jwt_required(), @role_required(UserRole.company), get_company_from_token() → user = User.query.get(company.user_id). Aucun user_id/company_id dans le body.
- /driver/me/push-privacy ne modifie que le user du driver authentifié (token).
  Gardes : @jwt_required(), @role_required(UserRole.driver), get_driver_from_token() → user = User.query.get(driver.user_id). Aucun user_id/driver_id dans le body.
- Validation stricte : seules les valeurs "detailed" | "discreet" (après .strip().lower()) → 400 sinon.

Les tests sont marqués @pytest.mark.integration et nécessitent PostgreSQL (DATABASE_URL).
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from http import HTTPStatus

import pytest

from models import Company, Driver, User, UserRole


def _make_company_user(db_session, role=UserRole.company):
    """Crée un User + Company (si company) et les persiste."""
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"pu_{suffix}",
        email=f"pu_{suffix}@test.ch",
        role=role,
        public_id=str(uuid.uuid4()),
    )
    user.set_password("password123", force_change=False)
    db_session.session.add(user)
    db_session.session.flush()
    if role == UserRole.company:
        company = Company(
            name=f"Company {suffix}",
            user_id=user.id,
            contact_email=user.email,
        )
        db_session.session.add(company)
        db_session.session.flush()
        return user, company
    return user, None


def _make_driver_user(db_session, company):
    """Crée un User + Driver pour une company et les persiste."""
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"driver_{suffix}",
        email=f"driver_{suffix}@test.ch",
        role=UserRole.driver,
        public_id=str(uuid.uuid4()),
    )
    user.set_password("password123", force_change=False)
    db_session.session.add(user)
    db_session.session.flush()
    driver = Driver(company_id=company.id, user_id=user.id, is_active=True)
    db_session.session.add(driver)
    db_session.session.flush()
    return user, driver


def _token_for_user(app, user, company_id=None, driver_id=None):
    from flask_jwt_extended import create_access_token

    claims = {
        "role": user.role.value,
        "company_id": company_id,
        "driver_id": getattr(driver_id, "id", driver_id)
        if driver_id is not None
        else None,
        "aud": "atmr-api",
    }
    with app.app_context():
        return create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
            expires_delta=timedelta(hours=24),
        )


@pytest.mark.integration
class TestPushPrivacyAccessControl:
    """Contrôle d'accès et multi-tenant push-privacy."""

    def test_company_push_privacy_only_modifies_current_company_user(
        self, app, client, db
    ):
        """PATCH /company-settings/push-privacy ne modifie que le user de la company du token."""
        user_a, company_a = _make_company_user(db)
        user_b, _ = _make_company_user(db)
        db.session.flush()

        # Valeur initiale : user_b en "detailed"
        if hasattr(user_b, "push_privacy_mode"):
            user_b.push_privacy_mode = "detailed"
        db.session.flush()
        db.session.refresh(user_b)

        token_a = _token_for_user(app, user_a, company_id=company_a.id)
        url = "/api/v1/company-settings/push-privacy"
        headers = {"Authorization": f"Bearer {token_a}"}
        resp = client.patch(
            url, json={"push_privacy_mode": "discreet"}, headers=headers
        )
        assert resp.status_code == 200, (resp.status_code, resp.get_data(as_text=True))

        db.session.expire_all()
        u_a = User.query.get(user_a.id)
        u_b = User.query.get(user_b.id)
        assert getattr(u_a, "push_privacy_mode", None) == "discreet"
        assert getattr(u_b, "push_privacy_mode", None) == "detailed"

    def test_driver_push_privacy_only_modifies_authenticated_driver_user(
        self, app, client, db
    ):
        """PATCH /driver/me/push-privacy ne modifie que le user du driver du token."""
        _, company = _make_company_user(db)
        user_driver_a, driver_a = _make_driver_user(db, company)
        user_driver_b, _ = _make_driver_user(db, company)
        db.session.flush()

        if hasattr(user_driver_b, "push_privacy_mode"):
            user_driver_b.push_privacy_mode = "detailed"
        db.session.flush()
        db.session.refresh(user_driver_b)

        token_a = _token_for_user(app, user_driver_a, driver_id=driver_a)
        url = "/api/v1/driver/me/push-privacy"
        headers = {"Authorization": f"Bearer {token_a}"}
        resp = client.patch(
            url, json={"push_privacy_mode": "discreet"}, headers=headers
        )
        assert resp.status_code == 200, (resp.status_code, resp.get_data(as_text=True))

        db.session.expire_all()
        u_a = User.query.get(user_driver_a.id)
        u_b = User.query.get(user_driver_b.id)
        assert getattr(u_a, "push_privacy_mode", None) == "discreet"
        assert getattr(u_b, "push_privacy_mode", None) == "detailed"

    def test_push_privacy_validation_rejects_invalid_mode(self, app, client, db):
        """PATCH avec push_privacy_mode invalide renvoie 400."""
        user_a, company_a = _make_company_user(db)
        db.session.flush()
        token_a = _token_for_user(app, user_a, company_id=company_a.id)
        url = "/api/v1/company-settings/push-privacy"
        headers = {"Authorization": f"Bearer {token_a}"}

        # "DETAILED" / "DISCREET" sont normalisés en minuscules et acceptés
        for invalid in ("invalid", "foo", "", "x"):
            resp = client.patch(
                url, json={"push_privacy_mode": invalid}, headers=headers
            )
            assert resp.status_code == HTTPStatus.BAD_REQUEST, (
                f"mode={invalid!r} should return 400, got {resp.status_code}"
            )

    def test_push_privacy_validation_accepts_detailed_and_discreet(
        self, app, client, db
    ):
        """PATCH avec 'detailed' ou 'discreet' (minuscules) renvoie 200."""
        user_a, company_a = _make_company_user(db)
        db.session.flush()
        token_a = _token_for_user(app, user_a, company_id=company_a.id)
        url = "/api/v1/company-settings/push-privacy"
        headers = {"Authorization": f"Bearer {token_a}"}

        for mode in ("detailed", "discreet"):
            resp = client.patch(url, json={"push_privacy_mode": mode}, headers=headers)
            assert resp.status_code == 200, (
                f"mode={mode!r} should return 200, got {resp.status_code}"
            )
            data = resp.get_json()
            assert data.get("push_privacy_mode") == mode
