"""Isolation entreprise du temps de travail."""

from __future__ import annotations

import uuid

from models import Company, Driver, User
from models.enums import UserRole


def _other_driver(db) -> Driver:
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"owner_{suffix}"
    owner.email = f"owner_{suffix}@example.com"
    owner.public_id = str(uuid.uuid4())
    owner.role = UserRole.company
    owner.set_password("password123", force_change=False)
    db.session.add(owner)
    db.session.flush()

    company = Company()
    company.name = f"Autre {suffix}"
    company.address = "Rue Autre 1"
    company.contact_phone = "0210000000"
    company.contact_email = f"autre_{suffix}@example.com"
    company.user_id = owner.id
    db.session.add(company)
    db.session.flush()

    driver_user = User()
    driver_user.username = f"drv_{suffix}"
    driver_user.email = f"drv_{suffix}@example.com"
    driver_user.public_id = str(uuid.uuid4())
    driver_user.role = UserRole.driver
    driver_user.first_name = "Noah"
    driver_user.last_name = "Bernard"
    driver_user.set_password("password123", force_change=False)
    db.session.add(driver_user)
    db.session.flush()

    driver = Driver()
    driver.user = driver_user
    driver.company_id = company.id
    driver.is_active = True
    db.session.add(driver)
    db.session.flush()
    return driver


def test_chauffeur_dune_autre_entreprise_est_introuvable(
    client, auth_headers, db, sample_company
):
    driver = _other_driver(db)
    assert driver.company_id != sample_company.id
    response = client.get(
        f"/api/v1/companies/me/work-time/drivers/{driver.id}"
        "?from=2026-09-01&to=2026-09-30",
        headers=auth_headers,
    )
    assert response.status_code == 404


def test_resume_sans_jeton_est_refuse(client):
    response = client.get(
        "/api/v1/companies/me/work-time/summary?from=2026-09-01&to=2026-09-30"
    )
    assert response.status_code in {401, 403}
