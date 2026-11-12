"""Contrat des deux déclarations de transport validées."""

from __future__ import annotations

import json
import uuid

import pytest
from flask_jwt_extended import create_access_token

from application.companies.transport_capabilities import (
    apply_transport_capabilities,
    declared_manual_wheelchair,
    declared_standard_assistance,
)
from models import Company, User, UserRole
from security.audit_log import AuditLog


def test_absent_false_and_true_have_distinct_meanings() -> None:
    company = type("Company", (), {})()
    company.provides_standard_assistance = None
    company.provides_manual_wheelchair = None
    company.manual_wheelchair_count = None
    assert declared_standard_assistance(company) is False
    assert declared_manual_wheelchair(company) is False

    company.provides_standard_assistance = False
    company.provides_manual_wheelchair = True
    company.manual_wheelchair_count = 0
    assert declared_standard_assistance(company) is False
    assert declared_manual_wheelchair(company) is False

    company.provides_standard_assistance = True
    company.manual_wheelchair_count = 2
    assert declared_standard_assistance(company) is True
    assert declared_manual_wheelchair(company) is True


def test_checkbox_without_stock_is_rejected() -> None:
    company = type("Company", (), {})()
    company.id = 1
    company.provides_standard_assistance = None
    company.provides_manual_wheelchair = None
    company.manual_wheelchair_count = None
    with pytest.raises(ValueError, match="réellement disponibles"):
        apply_transport_capabilities(
            company,
            {
                "provides_standard_assistance": None,
                "provides_manual_wheelchair": True,
                "manual_wheelchair_count": 0,
            },
            user_id=1,
        )
    assert company.provides_manual_wheelchair is None


def test_voluntary_declaration_is_audited(db) -> None:
    user = User()
    user.email = f"cap_{uuid.uuid4().hex[:8]}@test.com"
    user.username = user.email
    user.password = "test"
    user.role = UserRole.COMPANY.value
    db.session.add(user)
    db.session.flush()
    company = Company()
    company.name = "Transport Déclaré"
    company.user_id = user.id
    db.session.add(company)
    db.session.flush()
    assert company.provides_standard_assistance is None

    apply_transport_capabilities(
        company,
        {
            "provides_standard_assistance": True,
            "provides_manual_wheelchair": True,
            "manual_wheelchair_count": 1,
        },
        user_id=user.id,
    )
    db.session.flush()
    assert company.provides_standard_assistance is True
    assert company.manual_wheelchair_count == 1
    audit = (
        AuditLog.query.filter_by(
            company_id=company.id,
            action_type="company_transport_capabilities_updated",
        )
        .one()
    )
    details = json.loads(audit.action_details)
    assert details["before"]["provides_standard_assistance"] is None
    assert details["after"]["provides_standard_assistance"] is True
    assert details["after"]["manual_wheelchair_count"] == 1


def test_company_can_declare_and_driver_cannot(client, db) -> None:
    user = User()
    user.email = f"cap_route_{uuid.uuid4().hex[:8]}@test.com"
    user.username = user.email
    user.password = "test"
    user.role = UserRole.COMPANY.value
    db.session.add(user)
    db.session.flush()
    company = Company()
    company.name = "Transport Route"
    company.user_id = user.id
    db.session.add(company)
    db.session.flush()
    token = create_access_token(
        identity=str(user.public_id),
        additional_claims={
            "role": UserRole.COMPANY.value,
            "aud": "atmr-api",
            "user_id": user.id,
            "company_id": company.id,
        },
    )
    headers = {"Authorization": f"Bearer {token}"}
    denied = client.put(
        "/api/v1/companies/me/transport-capabilities",
        json={
            "provides_standard_assistance": True,
            "provides_manual_wheelchair": True,
            "manual_wheelchair_count": None,
        },
        headers=headers,
    )
    assert denied.status_code == 400

    saved = client.put(
        "/api/v1/companies/me/transport-capabilities",
        json={
            "provides_standard_assistance": False,
            "provides_manual_wheelchair": True,
            "manual_wheelchair_count": 3,
        },
        headers=headers,
    )
    assert saved.status_code == 200
    body = saved.get_json()["data"]
    assert body["provides_standard_assistance"] is False
    assert body["manual_wheelchair_count"] == 3
    assert "soins" in body["definitions"]["standard_assistance"]

    driver = User()
    driver.email = f"cap_driver_{uuid.uuid4().hex[:8]}@test.com"
    driver.username = driver.email
    driver.password = "test"
    driver.role = UserRole.DRIVER.value
    db.session.add(driver)
    db.session.flush()
    driver_token = create_access_token(
        identity=str(driver.public_id),
        additional_claims={
            "role": UserRole.DRIVER.value,
            "aud": "atmr-api",
            "user_id": driver.id,
        },
    )
    forbidden = client.put(
        "/api/v1/companies/me/transport-capabilities",
        json={
            "provides_standard_assistance": True,
            "provides_manual_wheelchair": None,
            "manual_wheelchair_count": None,
        },
        headers={"Authorization": f"Bearer {driver_token}"},
    )
    assert forbidden.status_code == 403
    db.session.delete(company)
    db.session.delete(user)
    db.session.delete(driver)
    db.session.commit()
