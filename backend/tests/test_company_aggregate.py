"""Tests pour l'agrégat Company."""

from __future__ import annotations

from datetime import datetime

import pytest

from companies.domain.company import Company
from companies.domain.company_id import CompanyId
from companies.domain.value_objects import (
    BillingSettings,
    CompanySettings,
    DispatchMode,
    PlanningSettings,
)

# Constantes pour les tests
TEST_USER_ID = 1
TEST_COMPANY_NAME = "Test Company"


class TestCompanyAggregate:
    """Tests pour l'agrégat Company."""

    def test_create_company(self):
        """Test création d'une entreprise."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        assert company.id.value == 1
        assert company.name == TEST_COMPANY_NAME
        assert company.settings.dispatch_enabled is False

    def test_company_enable_dispatch(self):
        """Test activation du dispatch."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        company.enable_dispatch()
        assert company.settings.dispatch_enabled is True

    def test_company_disable_dispatch(self):
        """Test désactivation du dispatch."""
        settings = CompanySettings(
            dispatch_enabled=True,
            dispatch_mode=DispatchMode("semi_auto"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        company.disable_dispatch()
        assert company.settings.dispatch_enabled is False

    def test_company_set_dispatch_mode(self):
        """Test changement du mode de dispatch."""
        settings = CompanySettings(
            dispatch_enabled=True,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        company.set_dispatch_mode(DispatchMode("fully_auto"))
        assert company.settings.dispatch_mode.is_fully_auto() is True

    def test_company_set_dispatch_mode_disabled(self):
        """Test qu'on ne peut pas changer le mode si dispatch est désactivé."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        with pytest.raises(ValueError, match="dispatch is disabled"):
            company.set_dispatch_mode(DispatchMode("semi_auto"))

    def test_company_approve(self):
        """Test approbation d'une entreprise."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
            is_approved=False,
        )

        company.approve()
        assert company.is_approved is True
        assert company.accepted_at is not None

    def test_company_update_planning_settings(self):
        """Test mise à jour des paramètres de planification."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        new_planning = PlanningSettings(max_daily_bookings=100, service_area="Paris")
        company.update_planning_settings(new_planning)
        assert company.settings.planning_settings.max_daily_bookings == 100

    def test_company_update_billing_settings(self):
        """Test mise à jour des paramètres de facturation."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        new_billing = BillingSettings(
            billing_email="billing@example.com", billing_notes="Test notes"
        )
        company.update_billing_settings(new_billing)
        assert company.settings.billing_settings.billing_email == "billing@example.com"

    def test_company_validate(self):
        """Test validation des invariants."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=TEST_USER_ID,
            settings=settings,
        )

        assert company.validate() is True

    def test_company_validate_invalid_name(self):
        """Test validation échoue si nom vide."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name="",  # Nom vide
            user_id=TEST_USER_ID,
            settings=settings,
        )

        assert company.validate() is False

    def test_company_validate_invalid_user_id(self):
        """Test validation échoue si user_id invalide."""
        settings = CompanySettings(
            dispatch_enabled=False,
            dispatch_mode=DispatchMode("manual"),
            planning_settings=PlanningSettings(),
            billing_settings=BillingSettings(),
        )

        company = Company(
            id=CompanyId(1),
            name=TEST_COMPANY_NAME,
            user_id=0,  # user_id invalide
            settings=settings,
        )

        assert company.validate() is False

    def test_planning_settings_validation(self):
        """Test validation de PlanningSettings."""
        with pytest.raises(ValueError, match="must be non-negative"):
            PlanningSettings(max_daily_bookings=-1)

    def test_billing_settings_validation(self):
        """Test validation de BillingSettings."""
        with pytest.raises(ValueError, match="valid email format"):
            BillingSettings(billing_email="invalid-email")
