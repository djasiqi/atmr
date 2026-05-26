"""Tests création automatique du chauffeur lié au compte entreprise mobile."""

from __future__ import annotations

import uuid

import pytest

from models import Company, Driver, User
from models.enums import UserRole


def _create_dispatch_company_user(db) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"mobile_ops_{suffix}"
    user.email = f"mobile_ops_{suffix}@test.ch"
    user.role = UserRole.COMPANY
    user.set_password("Password123!")
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.user_id = user.id
    company.name = f"Mobile Transport {suffix}"
    company.contact_email = user.email
    company.is_approved = True
    company.dispatch_enabled = True
    db.session.add(company)
    db.session.flush()
    return user


@pytest.mark.unit
def test_company_mobile_login_provisions_operator_driver_once(client, db):
    user = _create_dispatch_company_user(db)

    first = client.post(
        "/api/v1/company_mobile/auth/login",
        json={
            "method": "password",
            "email": user.email,
            "password": "Password123!",
        },
    )
    second = client.post(
        "/api/v1/company_mobile/auth/login",
        json={
            "method": "password",
            "email": user.email,
            "password": "Password123!",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    drivers = Driver.query.filter_by(user_id=user.id).all()
    assert len(drivers) == 1
    assert drivers[0].company_id == user.company.id
    assert drivers[0].is_active is True


@pytest.mark.unit
def test_company_mobile_driver_account_endpoint_provisions_before_lookup(client, db):
    user = _create_dispatch_company_user(db)
    login = client.post(
        "/api/v1/company_mobile/auth/login",
        json={
            "method": "password",
            "email": user.email,
            "password": "Password123!",
        },
    )
    token = login.get_json()["token"]
    Driver.query.filter_by(user_id=user.id).delete()
    db.session.commit()

    response = client.get(
        "/api/v1/company_mobile/auth/me/driver-account",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["has_driver_account"] is True
    assert Driver.query.filter_by(user_id=user.id).count() == 1
