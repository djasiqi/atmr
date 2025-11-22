# pyright: reportAttributeAccessIssue=false
"""
Tests pour services/unified_dispatch/autonomous_manager.py
Coverage cible : 90%+
Tests pour les 3 modes de dispatch : MANUAL, SEMI_AUTO, FULLY_AUTO
"""

from unittest.mock import MagicMock, patch

import pytest

from models import DispatchMode
from services.unified_dispatch.autonomous_manager import (
    AutonomousDispatchManager,
    get_manager_for_company,
)
from services.unified_dispatch.reactive_suggestions import Suggestion
from tests.factories import CompanyFactory


def create_test_suggestion(**kwargs):
    """Helper pour créer une Suggestion avec valeurs par défaut."""
    defaults = {
        "action": "notify_customer",
        "priority": "medium",
        "message": "Test message",
        "auto_applicable": True,
        "booking_id": 1,
        "driver_id": 2,
    }
    defaults.update(kwargs)
    return Suggestion(**defaults)


class TestAutonomousManagerInit:
    """Tests pour l'initialisation du manager."""

    def test_init_with_valid_company(self, db):
        """Test initialisation avec company valide."""
        company = CompanyFactory(dispatch_mode=DispatchMode.SEMI_AUTO)

        manager = AutonomousDispatchManager(company.id)

        assert manager.company_id == company.id
        assert manager.company == company
        assert manager.mode == DispatchMode.SEMI_AUTO
        assert manager.config is not None, "Config devrait être chargée"

    def test_init_with_invalid_company(self, db):
        """Test initialisation avec company inexistante."""
        with pytest.raises(ValueError, match=r"Company .* not found"):
            AutonomousDispatchManager(company_id=0.99999)

    def test_get_manager_for_company_factory(self, db):
        """Test factory function get_manager_for_company."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)

        manager = get_manager_for_company(company.id)

        assert isinstance(manager, AutonomousDispatchManager)
        assert manager.company_id == company.id
        assert manager.mode == DispatchMode.FULLY_AUTO


class TestShouldRunAutorun:
    """Tests pour should_run_autorun()."""

    def test_should_run_autorun_manual_mode(self, db):
        """Test autorun en mode MANUAL : toujours False."""
        company = CompanyFactory(dispatch_mode=DispatchMode.MANUAL)
        manager = AutonomousDispatchManager(company.id)

        result = manager.should_run_autorun()

        assert result is False, "Autorun devrait être désactivé en mode MANUAL"

    def test_should_run_autorun_semi_auto_enabled(self, db):
        """Test autorun en mode SEMI_AUTO avec config activée."""
        company = CompanyFactory(dispatch_mode=DispatchMode.SEMI_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Activer auto_dispatch dans config
        manager.config["auto_dispatch"]["enabled"] = True

        result = manager.should_run_autorun()

        assert result is True, "Autorun devrait être activé si config enabled=True"

    def test_should_run_autorun_semi_auto_disabled(self, db):
        """Test autorun en mode SEMI_AUTO avec config désactivée."""
        company = CompanyFactory(dispatch_mode=DispatchMode.SEMI_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Désactiver auto_dispatch dans config
        manager.config["auto_dispatch"]["enabled"] = False

        result = manager.should_run_autorun()

        assert result is False, "Autorun devrait être désactivé si config enabled=False"

    def test_should_run_autorun_fully_auto(self, db):
        """Test autorun en mode FULLY_AUTO : toujours True."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        result = manager.should_run_autorun()

        assert result is True, "Autorun devrait toujours être activé en FULLY_AUTO"


class TestShouldRunRealtimeOptimizer:
    """Tests pour should_run_realtime_optimizer()."""

    def test_should_run_realtime_optimizer_manual(self, db):
        """Test realtime optimizer en mode MANUAL : False."""
        company = CompanyFactory(dispatch_mode=DispatchMode.MANUAL)
        manager = AutonomousDispatchManager(company.id)

        result = manager.should_run_realtime_optimizer()

        assert result is False, "Realtime optimizer désactivé en MANUAL"

    def test_should_run_realtime_optimizer_semi_auto(self, db):
        """Test realtime optimizer en mode SEMI_AUTO selon config."""
        company = CompanyFactory(dispatch_mode=DispatchMode.SEMI_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Tester avec enabled=True
        manager.config["realtime_optimizer"]["enabled"] = True
        assert manager.should_run_realtime_optimizer() is True

        # Tester avec enabled=False
        manager.config["realtime_optimizer"]["enabled"] = False
        assert manager.should_run_realtime_optimizer() is False

    def test_should_run_realtime_optimizer_fully_auto(self, db):
        """Test realtime optimizer en mode FULLY_AUTO selon config."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Devrait suivre la config même en FULLY_AUTO
        manager.config["realtime_optimizer"]["enabled"] = True
        assert manager.should_run_realtime_optimizer() is True


class TestCanAutoApplySuggestion:
    """Tests pour can_auto_apply_suggestion()."""

    def test_can_auto_apply_manual_mode(self, db):
        """Test auto-apply en mode MANUAL : jamais."""
        company = CompanyFactory(dispatch_mode=DispatchMode.MANUAL)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion()

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is False, "Jamais d'auto-apply en mode MANUAL"

    def test_can_auto_apply_semi_auto(self, db):
        """Test auto-apply en mode SEMI_AUTO : jamais."""
        company = CompanyFactory(dispatch_mode=DispatchMode.SEMI_AUTO)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion()

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is False, "Jamais d'auto-apply en mode SEMI_AUTO"

    def test_can_auto_apply_fully_auto_notification(self, db):
        """Test auto-apply notification en mode FULLY_AUTO."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion(message="Test notification")

        # Activer customer_notifications dans rules
        manager.config["auto_apply_rules"]["customer_notifications"] = True

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is True, "Notifications devraient être auto-applicable"

    def test_can_auto_apply_not_auto_applicable(self, db):
        """Test suggestion marquée non auto-applicable."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion(auto_applicable=False)

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is False, "Suggestion non auto-applicable devrait être rejetée"

    def test_can_auto_apply_time_adjustment_within_threshold(self, db):
        """Test auto-apply adjust_time avec retard acceptable."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion(
            action="adjust_time", additional_data={"delay_minutes": 5}
        )

        # Configurer
        manager.config["safety_limits"]["require_approval_delay_minutes"] = 15
        manager.config["auto_apply_rules"]["minor_time_adjustments"] = True

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is True, "Petit retard devrait être auto-applicable"

    def test_can_auto_apply_time_adjustment_exceeds_threshold(self, db):
        """Test auto-apply adjust_time avec retard trop important."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion(
            action="adjust_time", additional_data={"delay_minutes": 30}
        )

        # Configurer
        manager.config["safety_limits"]["require_approval_delay_minutes"] = 15
        manager.config["auto_apply_rules"]["minor_time_adjustments"] = True

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is False, "Gros retard devrait nécessiter validation manuelle"

    def test_can_auto_apply_reassignment_disabled(self, db):
        """Test auto-apply reassign (désactivé par défaut)."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion(action="reassign")

        # Par défaut, reassignments = False
        manager.config["auto_apply_rules"]["reassignments"] = False

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is False, "Réassignations devraient être désactivées par défaut"

    def test_can_auto_apply_redistribute_never(self, db):
        """Test auto-apply redistribute (jamais autorisé)."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion(action="redistribute")

        result = manager.can_auto_apply_suggestion(suggestion)

        assert result is False, "Redistribution jamais auto-applicable (trop critique)"


class TestShouldTriggerReoptimization:
    """Tests pour should_trigger_reoptimization()."""

    def test_should_trigger_reoptimization_manual_mode(self, db):
        """Test reoptimization en mode MANUAL : jamais."""
        company = CompanyFactory(dispatch_mode=DispatchMode.MANUAL)
        manager = AutonomousDispatchManager(company.id)

        result = manager.should_trigger_reoptimization("delay", {"delay_minutes": 30})

        assert result is False, "Jamais de reoptimization en MANUAL"

    def test_should_trigger_reoptimization_delay(self, db):
        """Test reoptimization sur retard important."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Configurer threshold = 15 min
        manager.config["re_optimize_triggers"]["delay_threshold_minutes"] = 15

        # Retard de 20 min
        result = manager.should_trigger_reoptimization("delay", {"delay_minutes": 20})

        assert result is True, "Retard > threshold devrait déclencher reoptimization"

        # Retard de 10 min
        result = manager.should_trigger_reoptimization("delay", {"delay_minutes": 10})

        assert result is False, "Retard < threshold ne devrait pas déclencher"

    def test_should_trigger_reoptimization_driver_unavailable(self, db):
        """Test reoptimization si chauffeur indisponible."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        manager.config["re_optimize_triggers"]["driver_became_unavailable"] = True

        result = manager.should_trigger_reoptimization(
            "driver_unavailable", {"driver_id": 123}
        )

        assert result is True, "Driver unavailable devrait déclencher reoptimization"

    def test_should_trigger_reoptimization_better_driver(self, db):
        """Test reoptimization si meilleur chauffeur disponible."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Threshold = 10 min de gain minimum
        manager.config["re_optimize_triggers"][
            "better_driver_available_gain_minutes"
        ] = 10

        # Gain de 15 min
        result = manager.should_trigger_reoptimization(
            "better_driver_available", {"gain_minutes": 15}
        )

        assert result is True, "Gain suffisant devrait déclencher reoptimization"

        # Gain de 5 min
        result = manager.should_trigger_reoptimization(
            "better_driver_available", {"gain_minutes": 5}
        )

        assert result is False, "Gain insuffisant ne devrait pas déclencher"


class TestCheckSafetyLimits:
    """Tests pour check_safety_limits()."""

    def test_check_safety_limits_returns_ok(self, db):
        """Test que check_safety_limits autorise les actions (implémentation actuelle)."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        can_proceed, reason = manager.check_safety_limits("notify_customer")

        assert can_proceed is True, "Devrait autoriser l'action"
        assert reason == "OK"

        # Tester différents types d'actions
        for action_type in ["reassign", "adjust_time", "redistribute"]:
            can_proceed, reason = manager.check_safety_limits(action_type)
            assert can_proceed is True, f"Devrait autoriser {action_type}"


class TestProcessOpportunities:
    """Tests pour process_opportunities()."""

    def test_process_opportunities_empty_list(self, db):
        """Test traitement d'une liste vide."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        stats = manager.process_opportunities([], dry_run=True)

        assert stats["total_opportunities"] == 0
        assert stats["auto_applied"] == 0
        assert stats["manual_required"] == 0

    def test_process_opportunities_dry_run(self, db):
        """Test traitement en mode dry_run."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Mock d'opportunité avec suggestion auto-applicable
        suggestion = create_test_suggestion()

        opportunity = MagicMock()
        opportunity.suggestions = [suggestion]

        # Configurer pour autoriser
        manager.config["auto_apply_rules"]["customer_notifications"] = True

        stats = manager.process_opportunities([opportunity], dry_run=True)

        assert stats["total_opportunities"] == 1
        assert stats["auto_applied"] == 1, "En dry_run, devrait compter comme appliqué"
        assert stats["manual_required"] == 0

    def test_process_opportunities_manual_required(self, db):
        """Test opportunités nécessitant validation manuelle."""
        company = CompanyFactory(dispatch_mode=DispatchMode.SEMI_AUTO)  # Mode semi-auto
        manager = AutonomousDispatchManager(company.id)

        suggestion = create_test_suggestion()

        opportunity = MagicMock()
        opportunity.suggestions = [suggestion]

        stats = manager.process_opportunities([opportunity], dry_run=True)

        assert stats["manual_required"] == 1, (
            "En SEMI_AUTO, devrait nécessiter validation"
        )
        assert stats["auto_applied"] == 0

    @patch("services.unified_dispatch.autonomous_manager.apply_suggestion")
    def test_process_opportunities_with_apply(self, mock_apply, db):
        """Test application réelle de suggestions."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Mock apply_suggestion pour retourner succès
        mock_apply.return_value = {"success": True}

        suggestion = create_test_suggestion()

        opportunity = MagicMock()
        opportunity.suggestions = [suggestion]

        manager.config["auto_apply_rules"]["customer_notifications"] = True

        stats = manager.process_opportunities([opportunity], dry_run=False)

        assert stats["auto_applied"] == 1
        assert stats["errors"] == 0
        mock_apply.assert_called_once()

    @patch("services.unified_dispatch.autonomous_manager.apply_suggestion")
    def test_process_opportunities_with_error(self, mock_apply, db):
        """Test gestion d'erreur lors de l'application."""
        company = CompanyFactory(dispatch_mode=DispatchMode.FULLY_AUTO)
        manager = AutonomousDispatchManager(company.id)

        # Mock apply_suggestion pour lever une exception
        mock_apply.side_effect = Exception("Test error")

        suggestion = create_test_suggestion()

        opportunity = MagicMock()
        opportunity.suggestions = [suggestion]

        manager.config["auto_apply_rules"]["customer_notifications"] = True

        stats = manager.process_opportunities([opportunity], dry_run=False)

        assert stats["errors"] == 1
        assert stats["auto_applied"] == 0
