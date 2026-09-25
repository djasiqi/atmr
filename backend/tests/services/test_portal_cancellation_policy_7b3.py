"""7B.3 — Bridge publication politique d'annulation PORTAL depuis billing."""

from __future__ import annotations

import uuid

import pytest

from ext import db
from models.company import Company
from models.company_portal_cancellation_policy import CompanyPortalCancellationPolicy
from models.enums import UserRole
from models.invoice import CompanyBillingSettings
from models.user import User
from services.legal.portal_cancellation_policy import (
    get_current_cancellation_policy,
    get_portal_policy_publication_status,
    publish_portal_policy_from_billing_settings,
    render_portal_cancellation_policy_text,
)


SAMPLE_POLICY = {
    "enabled": True,
    "apply_when_driver_assigned_only": True,
    "min_fee_chf": 0,
    "max_fee_chf": None,
    "basis": "booking_amount",
    "tiers": [
        {"type": "time", "hours_before": 24, "percent": 80, "label": ""},
        {"type": "status", "status": "EN_ROUTE", "percent": 100},
    ],
    "reason_overrides": {
        "LAST_MINUTE": {"billable": True},
        "NO_SHOW": {"billable": True},
        "CLIENT_REQUEST": {"billable": False},
        "COMPANY_ISSUE": {"billable": False},
        "MAJOR_DELAY": {"billable": False},
        "VEHICLE_ISSUE": {"billable": False},
        "OTHER": {"billable": False},
    },
}


def _company_with_billing(*, policy: dict | None) -> Company:
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"pol_{suffix}"
    owner.email = f"pol-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = "Emmenez-moi"
    company.user_id = owner.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    billing = CompanyBillingSettings()
    billing.company_id = company.id
    billing.cancellation_policy = policy
    db.session.add(billing)
    db.session.flush()
    return company


@pytest.mark.usefixtures("db_session")
class TestRenderPortalCancellationPolicyText:
    def test_disabled_policy_explicit_zero_fees(self):
        text = render_portal_cancellation_policy_text(
            company_name="Emmenez-moi",
            policy={"enabled": False},
        )
        assert "Conditions d'annulation de Emmenez-moi" in text
        assert "aucun frais" in text.lower()

    def test_none_policy_same_as_disabled(self):
        text = render_portal_cancellation_policy_text(
            company_name="Test Co",
            policy=None,
        )
        assert "aucun frais" in text.lower()

    def test_active_policy_reflects_tiers_and_overrides(self):
        text = render_portal_cancellation_policy_text(
            company_name="Emmenez-moi",
            policy=SAMPLE_POLICY,
        )
        assert "chauffeur" in text.lower()
        assert "assigné" in text.lower()
        assert "80 %" in text
        assert "100 %" in text
        assert "en route" in text.lower()
        assert "Client a demandé l'annulation" in text
        assert "non facturable" in text
        assert "Client ne s'est pas présenté" in text
        assert "facturable selon les paliers" in text


@pytest.mark.usefixtures("db_session")
class TestPublishPortalPolicyFromBilling:
    def test_publish_creates_v1_even_when_disabled(self):
        company = _company_with_billing(policy={"enabled": False})
        result = publish_portal_policy_from_billing_settings(
            company_id=company.id,
            company_name=company.name,
        )
        db.session.commit()
        assert result.ok
        assert result.policy is not None
        assert result.policy.version == "v1"
        assert result.policy.is_current is True
        assert "aucun frais" in result.policy.body_text.lower()
        assert "[Réf. config " in result.policy.body_text

    def test_publish_from_active_config(self):
        company = _company_with_billing(policy=SAMPLE_POLICY)
        result = publish_portal_policy_from_billing_settings(
            company_id=company.id,
            company_name=company.name,
        )
        db.session.commit()
        assert result.ok
        assert result.policy.version == "v1"
        assert "80 %" in result.policy.body_text
        current = get_current_cancellation_policy(company.id)
        assert current is not None
        assert current.id == result.policy.id

    def test_second_publish_bumps_version_and_supersedes(self):
        company = _company_with_billing(policy=SAMPLE_POLICY)
        r1 = publish_portal_policy_from_billing_settings(
            company_id=company.id,
            company_name=company.name,
        )
        db.session.flush()
        billing = CompanyBillingSettings.query.filter_by(
            company_id=company.id
        ).first()
        billing.cancellation_policy = {
            **SAMPLE_POLICY,
            "tiers": [
                {"type": "time", "hours_before": 24, "percent": 50, "label": ""},
                {"type": "status", "status": "EN_ROUTE", "percent": 100},
            ],
        }
        db.session.flush()
        r2 = publish_portal_policy_from_billing_settings(
            company_id=company.id,
            company_name=company.name,
        )
        db.session.commit()
        assert r1.ok and r2.ok
        assert r2.policy.version == "v2"
        assert "50 %" in r2.policy.body_text
        old = db.session.get(CompanyPortalCancellationPolicy, r1.policy.id)
        assert old is not None
        assert old.is_current is False
        assert "80 %" in old.body_text
        current = get_current_cancellation_policy(company.id)
        assert current is not None
        assert current.version == "v2"

    def test_publication_status_detects_unpublished_changes(self):
        company = _company_with_billing(policy=SAMPLE_POLICY)
        status0 = get_portal_policy_publication_status(
            company_id=company.id,
            company_name=company.name,
        )
        assert status0["policy"] is None
        assert status0["has_unpublished_changes"] is True
        assert status0["config_enabled"] is True

        publish_portal_policy_from_billing_settings(
            company_id=company.id,
            company_name=company.name,
        )
        db.session.flush()

        status1 = get_portal_policy_publication_status(
            company_id=company.id,
            company_name=company.name,
        )
        assert status1["policy"]["version"] == "v1"
        assert status1["has_unpublished_changes"] is False

        billing = CompanyBillingSettings.query.filter_by(
            company_id=company.id
        ).first()
        billing.cancellation_policy = {
            **SAMPLE_POLICY,
            "tiers": [
                {"type": "time", "hours_before": 12, "percent": 90, "label": ""},
            ],
        }
        db.session.flush()

        status2 = get_portal_policy_publication_status(
            company_id=company.id,
            company_name=company.name,
        )
        assert status2["policy"]["version"] == "v1"
        assert status2["has_unpublished_changes"] is True
        assert status2["next_version"] == "v2"
