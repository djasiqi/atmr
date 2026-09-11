"""Contrat d'éligibilité des transporteurs institution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from application.institutions.eligible_carriers import (
    explain_carrier_eligibility,
    is_synthetic_test_carrier,
    list_eligible_transport_companies,
)
from models import Company, Institution, User, UserRole
from models.enums import DispatchMode


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _make_carrier(
    db,
    *,
    name: str,
    email: str,
    address: str | None = None,
    contact_email: str | None = None,
    approved: bool = True,
    accepted: bool = True,
    dispatch_enabled: bool = False,
    platform_suspended: bool = False,
    owner_disabled: bool = False,
) -> Company:
    owner = User()
    owner.email = email
    owner.username = email
    owner.password = "test"
    owner.role = UserRole.COMPANY.value
    if owner_disabled:
        owner.disabled_at = datetime.now(UTC)
        owner.account_status = "disabled"
    db.session.add(owner)
    db.session.flush()

    company = Company()
    company.name = name
    company.user_id = owner.id
    company.address = address
    company.contact_email = contact_email or email
    company.is_approved = approved
    if accepted:
        company.accepted_at = datetime.now(UTC)
    company.dispatch_enabled = dispatch_enabled
    company.dispatch_mode = (
        DispatchMode.FULLY_AUTO if dispatch_enabled else DispatchMode.MANUAL
    )
    company.platform_suspended = platform_suspended
    db.session.add(company)
    db.session.flush()
    return company


def _institution(db) -> Institution:
    institution = Institution()
    institution.name = _unique("Clinique Eligible")
    institution.public_id = str(uuid.uuid4())
    db.session.add(institution)
    db.session.flush()
    return institution


def test_manual_approved_partner_is_eligible(db):
    institution = _institution(db)
    company = _make_carrier(
        db,
        name="Emmenez-moi",
        email=f"info+{uuid.uuid4().hex[:8]}@emmenez-moi.ch",
        address="Route de Chevrens 145, 1247 Anières",
        dispatch_enabled=False,
    )
    explained = explain_carrier_eligibility(company, institution)
    assert explained.approved == "PASS"
    assert explained.officially_accepted == "PASS"
    assert explained.not_suspended == "PASS"
    assert explained.owner_active == "PASS"
    assert explained.not_synthetic == "PASS"
    assert explained.institution_compatible == "PASS"
    assert explained.marketplace_billing == "PASS"
    assert explained.eligible is True
    assert company.id in {
        item.id for item in list_eligible_transport_companies(institution)
    }


def test_dispatch_enabled_false_does_not_exclude(db):
    """dispatch_enabled=false (MANUAL) n'est pas un critère d'exclusion."""
    institution = _institution(db)
    company = _make_carrier(
        db,
        name="Transport Manuel",
        email=f"{_unique('manuel')}@partenaire.ch",
        dispatch_enabled=False,
    )
    assert explain_carrier_eligibility(company, institution).eligible is True
    assert company.id in {
        item.id for item in list_eligible_transport_companies(institution)
    }


def test_inactive_owner_is_excluded(db):
    institution = _institution(db)
    company = _make_carrier(
        db,
        name="Transport Inactif",
        email=f"{_unique('inactif')}@partenaire.ch",
        owner_disabled=True,
    )
    explained = explain_carrier_eligibility(company, institution)
    assert explained.owner_active == "FAIL"
    assert explained.eligible is False
    assert company.id not in {
        item.id for item in list_eligible_transport_companies(institution)
    }


def test_synthetic_fixture_company_is_excluded(db):
    institution = _institution(db)
    company = _make_carrier(
        db,
        name="Co 104582",
        email=f"{_unique('co')}@test.com",
        dispatch_enabled=True,
    )
    assert is_synthetic_test_carrier(company) is True
    explained = explain_carrier_eligibility(company, institution)
    assert explained.not_synthetic == "FAIL"
    assert explained.eligible is False
    assert company.id not in {
        item.id for item in list_eligible_transport_companies(institution)
    }


def test_approved_without_accepted_at_is_excluded(db):
    institution = _institution(db)
    company = _make_carrier(
        db,
        name="Fixture Approuvee",
        email=f"{_unique('fixture')}@partenaire.ch",
        accepted=False,
        dispatch_enabled=True,
    )
    explained = explain_carrier_eligibility(company, institution)
    assert explained.officially_accepted == "FAIL"
    assert explained.eligible is False
    assert company.id not in {
        item.id for item in list_eligible_transport_companies(institution)
    }


def test_unapproved_company_is_excluded(db):
    institution = _institution(db)
    company = _make_carrier(
        db,
        name="Non Approuvee",
        email=f"{_unique('nope')}@partenaire.ch",
        approved=False,
    )
    explained = explain_carrier_eligibility(company, institution)
    assert explained.approved == "FAIL"
    assert explained.eligible is False
