"""Tests provisionnement chauffeur double casquette (compte entreprise)."""

from __future__ import annotations

import uuid

import pytest

from application.companies.drivers.ensure_company_operator_driver import (
    EnsureCompanyOperatorDriverUseCase,
)
from models import Company, Driver, User
from models.enums import UserRole
from routes.auth import _build_available_contexts


def _create_company_user(
    db,
    *,
    dispatch_enabled: bool = True,
    with_driver: bool = False,
) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"ops_{suffix}"
    user.email = f"ops_{suffix}@test.ch"
    user.role = UserRole.COMPANY
    user.set_password("Password123!")
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.user_id = user.id
    company.name = f"Transport {suffix}"
    company.contact_email = user.email
    company.is_approved = True
    company.dispatch_enabled = dispatch_enabled
    db.session.add(company)
    db.session.flush()

    if with_driver:
        driver = Driver(user_id=user.id, company_id=company.id, is_active=True)
        db.session.add(driver)
        db.session.flush()

    db.session.commit()
    return user


@pytest.mark.unit
def test_provisions_driver_for_company_with_dispatch(db):
    user = _create_company_user(db, dispatch_enabled=True)
    uc = EnsureCompanyOperatorDriverUseCase()

    result = uc.execute(user)

    assert result.created is True
    assert result.driver is not None
    assert result.driver.user_id == user.id
    assert result.driver.company_id == user.company.id


@pytest.mark.unit
def test_idempotent_when_driver_already_exists(db):
    user = _create_company_user(db, dispatch_enabled=True, with_driver=True)
    uc = EnsureCompanyOperatorDriverUseCase()

    result = uc.execute(user)

    assert result.created is False
    assert result.driver is not None
    assert Driver.query.filter_by(user_id=user.id).count() == 1


@pytest.mark.unit
def test_provisions_when_dispatch_disabled(db):
    user = _create_company_user(db, dispatch_enabled=False)
    uc = EnsureCompanyOperatorDriverUseCase()

    result = uc.execute(user)

    assert result.created is True
    assert result.driver is not None
    assert Driver.query.filter_by(user_id=user.id).count() == 1


@pytest.mark.unit
def test_bootstrap_contexts_include_driver_after_provision(db):
    user = _create_company_user(db, dispatch_enabled=True)
    EnsureCompanyOperatorDriverUseCase().execute(user)
    db.session.commit()

    user = User.query.filter_by(id=user.id).first()
    contexts = _build_available_contexts(user)
    types = {ctx["context_type"] for ctx in contexts}

    assert "company" in types
    assert "driver" in types
    driver_ctx = next(ctx for ctx in contexts if ctx["context_type"] == "driver")
    company_ctx = next(ctx for ctx in contexts if ctx["context_type"] == "company")
    assert driver_ctx["allow_mobile_context_switch"] is True
    assert company_ctx["allow_mobile_context_switch"] is True


@pytest.mark.unit
def test_bootstrap_contexts_include_driver_when_dispatch_disabled(db):
    user = _create_company_user(db, dispatch_enabled=False)
    EnsureCompanyOperatorDriverUseCase().execute(user)
    db.session.commit()

    user = User.query.filter_by(id=user.id).first()
    contexts = _build_available_contexts(user)
    types = {ctx["context_type"] for ctx in contexts}

    assert "company" in types
    assert "driver" in types
    driver_ctx = next(ctx for ctx in contexts if ctx["context_type"] == "driver")
    assert driver_ctx["allow_mobile_context_switch"] is True
