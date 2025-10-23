# ruff: noqa: DTZ001, DTZ003, DTZ005, DTZ011
# pyright: reportAttributeAccessIssue=false
"""
Tests pour le système de safety limits et rate limiting.

Couvre:
- Rate limiting horaire et journalier
- Limites par type d'action
- Logging des actions dans AutonomousAction
- Vérification des limites de sécurité
"""
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models.autonomous_action import AutonomousAction
from models.company import Company
from models.enums import DispatchMode
from services.unified_dispatch.autonomous_manager import AutonomousDispatchManager
from tests.factories import CompanyFactory


@pytest.fixture
def company_fully_auto(db):
    """Crée une entreprise en mode FULLY_AUTO avec config de test."""
    config = {
        "auto_dispatch": {"enabled": True},
        "realtime_optimizer": {"enabled": True},
        "auto_apply_rules": {
            "customer_notifications": True,
            "minor_time_adjustments": True,
            "reassignments": False
        },
        "safety_limits": {
            "max_auto_actions_per_hour": 5,
            "max_auto_actions_per_day": 20,
            "require_approval_delay_minutes": 30,
            "action_type_limits": {
                "reassign": {"per_hour": 2, "per_day": 10},
                "notify_customer": {"per_hour": 10, "per_day": 50}
            }
        },
        "re_optimize_triggers": {
            "delay_threshold_minutes": 15,
            "driver_became_unavailable": True,
            "better_driver_available_gain_minutes": 10
        }
    }

    company = CompanyFactory.create(
        dispatch_mode=DispatchMode.FULLY_AUTO,
        autonomous_config=json.dumps(config)  # Convertir en JSON string
    )
    return company


class TestAutonomousActionModel:
    """Tests du modèle AutonomousAction."""

    def test_create_autonomous_action(self, db, company_fully_auto):
        """Test création d'une action autonome."""
        action = AutonomousAction(
            company_id=company_fully_auto.id,
            action_type="reassign",
            action_description="Test reassignment",
            success=True,
            execution_time_ms=123.45
        )
        db.session.add(action)
        db.session.commit()

        assert action.id is not None
        assert action.company_id == company_fully_auto.id
        assert action.action_type == "reassign"
        assert action.success is True
        assert action.reviewed_by_admin is False

    def test_count_actions_last_hour(self, db, company_fully_auto):
        """Test comptage des actions dans la dernière heure."""
        now = datetime.utcnow()

        # Créer 3 actions dans la dernière heure
        for i in range(3):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Action {i}",
                success=True,
                created_at=now - timedelta(minutes=i * 10)
            )
            db.session.add(action)

        # Créer 1 action il y a 2 heures (ne devrait pas compter)
        old_action = AutonomousAction(
            company_id=company_fully_auto.id,
            action_type="reassign",
            action_description="Old action",
            success=True,
            created_at=now - timedelta(hours=2)
        )
        db.session.add(old_action)
        db.session.commit()

        # Vérifier le comptage
        count = AutonomousAction.count_actions_last_hour(company_fully_auto.id)
        assert count == 3

    def test_count_actions_today(self, db, company_fully_auto):
        """Test comptage des actions aujourd'hui."""
        today = datetime.utcnow()
        yesterday = today - timedelta(days=1)

        # Créer 5 actions aujourd'hui
        for i in range(5):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="notify_customer",
                action_description=f"Action today {i}",
                success=True,
                created_at=today - timedelta(hours=i)
            )
            db.add(action)

        # Créer 2 actions hier (ne devraient pas compter)
        for i in range(2):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="notify_customer",
                action_description=f"Action yesterday {i}",
                success=True,
                created_at=yesterday
            )
            db.add(action)
        db.commit()

        # Vérifier le comptage
        count = AutonomousAction.count_actions_today(company_fully_auto.id)
        assert count == 5

    def test_count_actions_by_type(self, db, company_fully_auto):
        """Test comptage des actions par type."""
        # Créer actions de différents types
        for action_type in ["reassign", "reassign", "notify_customer"]:
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type=action_type,
                action_description=f"Action {action_type}",
                success=True
            )
            db.add(action)
        db.commit()

        # Vérifier comptage par type
        reassign_count = AutonomousAction.count_actions_last_hour(
            company_fully_auto.id, "reassign"
        )
        notify_count = AutonomousAction.count_actions_last_hour(
            company_fully_auto.id, "notify_customer"
        )

        assert reassign_count == 2
        assert notify_count == 1

    def test_to_dict(self, db, company_fully_auto):
        """Test sérialisation en dictionnaire."""
        action = AutonomousAction(
            company_id=company_fully_auto.id,
            action_type="reassign",
            action_description="Test action",
            success=True,
            execution_time_ms=100.5,
            confidence_score=0.95
        )
        db.add(action)
        db.commit()

        data = action.to_dict()
        assert data["company_id"] == company_fully_auto.id
        assert data["action_type"] == "reassign"
        assert data["success"] is True
        assert data["execution_time_ms"] == 100.5
        assert data["confidence_score"] == 0.95


class TestSafetyLimits:
    """Tests du système de safety limits."""

    def test_check_safety_limits_ok(self, db, company_fully_auto):
        """Test check_safety_limits quand tout est OK."""
        manager = AutonomousDispatchManager(company_fully_auto.id)

        can_proceed, reason = manager.check_safety_limits("reassign")

        assert can_proceed is True
        assert reason == "OK"

    def test_hourly_limit_reached(self, db, company_fully_auto):
        """Test limite horaire atteinte."""
        # Créer 5 actions (limite configurée)
        for i in range(5):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Action {i}",
                success=True
            )
            db.add(action)
        db.commit()

        manager = AutonomousDispatchManager(company_fully_auto.id)
        can_proceed, reason = manager.check_safety_limits("reassign")

        assert can_proceed is False
        assert "Limite horaire globale atteinte" in reason
        assert "5/5 actions/h" in reason

    def test_daily_limit_reached(self, db, company_fully_auto):
        """Test limite journalière atteinte."""
        # Créer 20 actions (limite configurée)
        for i in range(20):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="notify_customer",
                action_description=f"Action {i}",
                success=True,
                created_at=datetime.utcnow() - timedelta(hours=i % 12)
            )
            db.add(action)
        db.commit()

        manager = AutonomousDispatchManager(company_fully_auto.id)
        can_proceed, reason = manager.check_safety_limits("notify_customer")

        assert can_proceed is False
        assert "Limite journalière globale atteinte" in reason
        assert "20/20 actions/jour" in reason

    def test_action_type_hourly_limit(self, db, company_fully_auto):
        """Test limite horaire par type d'action."""
        # Créer 2 actions "reassign" (limite configurée pour ce type)
        for i in range(2):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Reassign {i}",
                success=True
            )
            db.add(action)
        db.commit()

        manager = AutonomousDispatchManager(company_fully_auto.id)
        can_proceed, reason = manager.check_safety_limits("reassign")

        assert can_proceed is False
        assert "Limite horaire pour 'reassign' atteinte" in reason
        assert "2/2 actions/h" in reason

    def test_action_type_daily_limit(self, db, company_fully_auto):
        """Test limite journalière par type d'action."""
        # Créer 10 actions "reassign" sur la journée
        for i in range(10):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Reassign {i}",
                success=True,
                created_at=datetime.utcnow() - timedelta(hours=i)
            )
            db.add(action)
        db.commit()

        manager = AutonomousDispatchManager(company_fully_auto.id)
        can_proceed, reason = manager.check_safety_limits("reassign")

        assert can_proceed is False
        assert "Limite journalière pour 'reassign' atteinte" in reason
        assert "10/10 actions/jour" in reason

    def test_failed_actions_not_counted(self, db, company_fully_auto):
        """Test que les actions échouées ne comptent pas dans les limites."""
        # Créer 5 actions échouées
        for i in range(5):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Failed action {i}",
                success=False,
                error_message="Test error"
            )
            db.add(action)
        db.commit()

        manager = AutonomousDispatchManager(company_fully_auto.id)
        can_proceed, reason = manager.check_safety_limits("reassign")

        # Les actions échouées ne comptent pas, donc devrait passer
        assert can_proceed is True
        assert reason == "OK"

    def test_different_companies_isolated(self, db, company_fully_auto):
        """Test que les limites sont isolées par entreprise."""
        # Créer une autre entreprise
        other_company = CompanyFactory.create(dispatch_mode=DispatchMode.FULLY_AUTO)

        # Créer 5 actions pour l'autre entreprise
        for i in range(5):
            action = AutonomousAction(
                company_id=other_company.id,
                action_type="reassign",
                action_description=f"Action {i}",
                success=True
            )
            db.add(action)
        db.commit()

        # Vérifier que company_fully_auto n'est pas affectée
        manager = AutonomousDispatchManager(company_fully_auto.id)
        can_proceed, reason = manager.check_safety_limits("reassign")

        assert can_proceed is True
        assert reason == "OK"


class TestSafetyLimitsIntegration:
    """Tests d'intégration pour le rate limiting."""

    def test_multiple_action_types_independent_limits(self, db, company_fully_auto):
        """Test que différents types d'actions ont des limites indépendantes."""
        # Atteindre la limite pour "reassign"
        for i in range(2):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Reassign {i}",
                success=True
            )
            db.add(action)
        db.commit()

        manager = AutonomousDispatchManager(company_fully_auto.id)

        # "reassign" devrait être bloqué
        can_proceed_reassign, _ = manager.check_safety_limits("reassign")
        assert can_proceed_reassign is False

        # "notify_customer" devrait toujours passer (limite différente)
        can_proceed_notify, reason = manager.check_safety_limits("notify_customer")
        assert can_proceed_notify is True
        assert reason == "OK"

    def test_hourly_limit_resets_after_hour(self, db, company_fully_auto):
        """Test que la limite horaire se réinitialise après une heure."""
        two_hours_ago = datetime.utcnow() - timedelta(hours=2)

        # Créer 5 actions il y a 2 heures
        for i in range(5):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Old action {i}",
                success=True,
                created_at=two_hours_ago
            )
            db.add(action)
        db.commit()

        manager = AutonomousDispatchManager(company_fully_auto.id)
        can_proceed, reason = manager.check_safety_limits("reassign")

        # Les actions il y a 2h ne comptent plus, donc devrait passer
        assert can_proceed is True
        assert reason == "OK"

    def test_action_logging_updates_limits(self, db, company_fully_auto):
        """Test que le logging des actions met à jour les limites en temps réel."""
        manager = AutonomousDispatchManager(company_fully_auto.id)

        # Première vérification : OK
        can_proceed, _ = manager.check_safety_limits("reassign")
        assert can_proceed is True

        # Logger une action
        action = AutonomousAction(
            company_id=company_fully_auto.id,
            action_type="reassign",
            action_description="Action 1",
            success=True
        )
        db.add(action)
        db.commit()

        # Deuxième vérification : encore OK (1/2)
        can_proceed, _ = manager.check_safety_limits("reassign")
        assert can_proceed is True

        # Logger une deuxième action
        action2 = AutonomousAction(
            company_id=company_fully_auto.id,
            action_type="reassign",
            action_description="Action 2",
            success=True
        )
        db.add(action2)
        db.commit()

        # Troisième vérification : limite atteinte (2/2)
        can_proceed, reason = manager.check_safety_limits("reassign")
        assert can_proceed is False
        assert "Limite horaire pour 'reassign' atteinte" in reason


class TestActionLogging:
    """Tests du logging automatique des actions."""

    @patch('services.unified_dispatch.autonomous_manager.apply_suggestion')
    def test_successful_action_logged(self, mock_apply, db, company_fully_auto):
        """Test qu'une action réussie est loggée."""
        from services.unified_dispatch.reactive_suggestions import Suggestion

        # Mock apply_suggestion pour retourner succès
        mock_apply.return_value = {"success": True}

        # Créer une suggestion
        suggestion = Suggestion(
            action="reassign",
            message="Test reassignment",
            booking_id=123,
            driver_id=456,
            auto_applicable=True
        )

        # Mock de l'opportunité
        mock_opportunity = MagicMock()
        mock_opportunity.suggestions = [suggestion]

        manager = AutonomousDispatchManager(company_fully_auto.id)
        stats = manager.process_opportunities([mock_opportunity])

        # Vérifier les stats
        assert stats["auto_applied"] == 1
        assert stats["errors"] == 0

        # Vérifier qu'une action a été loggée
        actions = AutonomousAction.query.filter_by(
            company_id=company_fully_auto.id
        ).all()

        assert len(actions) == 1
        assert actions[0].action_type == "reassign"
        assert actions[0].success is True
        assert actions[0].execution_time_ms is not None

    @patch('services.unified_dispatch.autonomous_manager.apply_suggestion')
    def test_failed_action_logged(self, mock_apply, db, company_fully_auto):
        """Test qu'une action échouée est loggée."""
        from services.unified_dispatch.reactive_suggestions import Suggestion

        # Mock apply_suggestion pour retourner échec
        mock_apply.return_value = {
            "success": False,
            "error": "Test error message"
        }

        suggestion = Suggestion(
            action="reassign",
            message="Test failed reassignment",
            booking_id=123,
            driver_id=456,
            auto_applicable=True
        )

        mock_opportunity = MagicMock()
        mock_opportunity.suggestions = [suggestion]

        manager = AutonomousDispatchManager(company_fully_auto.id)
        stats = manager.process_opportunities([mock_opportunity])

        # Vérifier les stats
        assert stats["errors"] == 1
        assert stats["auto_applied"] == 0

        # Vérifier qu'une action échouée a été loggée
        actions = AutonomousAction.query.filter_by(
            company_id=company_fully_auto.id,
            success=False
        ).all()

        assert len(actions) == 1
        assert actions[0].error_message == "Test error message"

    @patch('services.unified_dispatch.autonomous_manager.apply_suggestion')
    def test_action_blocked_by_limits_not_logged(self, mock_apply, db, company_fully_auto):
        """Test qu'une action bloquée par les limites n'est pas loggée."""
        from services.unified_dispatch.reactive_suggestions import Suggestion

        # Atteindre la limite
        for i in range(2):
            action = AutonomousAction(
                company_id=company_fully_auto.id,
                action_type="reassign",
                action_description=f"Limit action {i}",
                success=True
            )
            db.add(action)
        db.commit()

        # Essayer d'exécuter une nouvelle action
        mock_apply.return_value = {"success": True}

        suggestion = Suggestion(
            action="reassign",
            message="Should be blocked",
            booking_id=123,
            driver_id=456,
            auto_applicable=True
        )

        mock_opportunity = MagicMock()
        mock_opportunity.suggestions = [suggestion]

        manager = AutonomousDispatchManager(company_fully_auto.id)
        stats = manager.process_opportunities([mock_opportunity])

        # Vérifier qu'aucune nouvelle action n'a été loggée
        new_actions = AutonomousAction.query.filter_by(
            company_id=company_fully_auto.id,
            action_description="Should be blocked"
        ).all()

        assert len(new_actions) == 0
        assert stats["blocked_by_limits"] == 1
        assert stats["auto_applied"] == 0

