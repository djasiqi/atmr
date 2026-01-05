# backend/tests/services/unified_dispatch/orchestration/test_shadow_mode_manager.py
"""Tests unitaires pour ShadowModeManager.

Tests pour :
- should_apply_rl : Décision d'appliquer les suggestions RL
- generate_and_store_suggestions : Génération et stockage des suggestions shadow mode
"""

from __future__ import annotations  # noqa: I001

import pytest
from unittest.mock import MagicMock, patch

from factories import CompanyFactory, DispatchRunFactory
from services.unified_dispatch.orchestration.shadow_mode_manager import (
    ShadowModeManager,
)


class TestShouldApplyRl:
    """Tests pour la méthode should_apply_rl."""

    @patch(
        "services.unified_dispatch.orchestration.shadow_mode_manager.ShadowModeOrchestrator"
    )
    def test_should_apply_rl_returns_true_with_quality_score(
        self, mock_orchestrator_class
    ):
        """Test : Retourne True avec un quality_score."""
        settings = MagicMock()
        manager = ShadowModeManager(settings)

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.should_apply_rl_with_guards.return_value = (
            True,
            85.5,
        )
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        company = CompanyFactory()
        assignments = [MagicMock()]
        problem = {"bookings": [MagicMock()], "drivers": [MagicMock()]}

        should_apply, quality_score = manager.should_apply_rl(
            company_id=company.id,
            dispatch_run_id=42,
            final_assignments=assignments,
            problem=problem,
            company=company,
        )

        assert should_apply is True
        assert quality_score == 85.5
        mock_orchestrator_instance.should_apply_rl_with_guards.assert_called_once_with(
            company_id=company.id,
            dispatch_run_id=42,
            final_assignments=assignments,
            problem=problem,
            company=company,
        )

    @patch(
        "services.unified_dispatch.orchestration.shadow_mode_manager.ShadowModeOrchestrator"
    )
    def test_should_apply_rl_returns_false_with_none_quality_score(
        self, mock_orchestrator_class
    ):
        """Test : Retourne False avec quality_score None."""
        settings = MagicMock()
        manager = ShadowModeManager(settings)

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.should_apply_rl_with_guards.return_value = (
            False,
            None,
        )
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        company = CompanyFactory()
        assignments = []
        problem = {"bookings": [], "drivers": []}

        should_apply, quality_score = manager.should_apply_rl(
            company_id=company.id,
            dispatch_run_id=None,
            final_assignments=assignments,
            problem=problem,
            company=company,
        )

        assert should_apply is False
        assert quality_score is None

    @patch(
        "services.unified_dispatch.orchestration.shadow_mode_manager.ShadowModeOrchestrator"
    )
    def test_should_apply_rl_handles_none_dispatch_run_id(
        self, mock_orchestrator_class
    ):
        """Test : Gère dispatch_run_id None."""
        settings = MagicMock()
        manager = ShadowModeManager(settings)

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.should_apply_rl_with_guards.return_value = (
            False,
            None,
        )
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        company = CompanyFactory()
        assignments = [MagicMock()]
        problem = {"bookings": [MagicMock()]}

        should_apply, quality_score = manager.should_apply_rl(
            company_id=company.id,
            dispatch_run_id=None,
            final_assignments=assignments,
            problem=problem,
            company=company,
        )

        assert should_apply is False
        assert quality_score is None


class TestGenerateAndStoreSuggestions:
    """Tests pour la méthode generate_and_store_suggestions."""

    @patch(
        "services.unified_dispatch.orchestration.shadow_mode_manager.ShadowModeOrchestrator"
    )
    def test_generate_and_store_suggestions_returns_true_when_stored(
        self, mock_orchestrator_class
    ):
        """Test : Retourne True quand des suggestions sont stockées."""
        settings = MagicMock()
        manager = ShadowModeManager(settings)

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.generate_and_store_shadow_suggestions.return_value = 5  # 5 suggestions stockées
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        problem = {"bookings": [MagicMock()], "drivers": [MagicMock()]}
        assignments = [MagicMock()]

        result = manager.generate_and_store_suggestions(
            dispatch_run_id=42,
            problem=problem,
            final_assignments=assignments,
            used_heuristic=True,
            used_solver=False,
        )

        assert result is True
        mock_orchestrator_instance.generate_and_store_shadow_suggestions.assert_called_once_with(
            dispatch_run_id=42,
            problem=problem,
            final_assignments=assignments,
            used_heuristic=True,
            used_solver=False,
        )

    @patch(
        "services.unified_dispatch.orchestration.shadow_mode_manager.ShadowModeOrchestrator"
    )
    def test_generate_and_store_suggestions_returns_false_when_not_stored(
        self, mock_orchestrator_class
    ):
        """Test : Retourne False quand aucune suggestion n'est stockée."""
        settings = MagicMock()
        manager = ShadowModeManager(settings)

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.generate_and_store_shadow_suggestions.return_value = 0  # Aucune suggestion stockée
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        problem = {"bookings": []}
        assignments = []

        result = manager.generate_and_store_suggestions(
            dispatch_run_id=42,
            problem=problem,
            final_assignments=assignments,
            used_heuristic=False,
            used_solver=False,
        )

        assert result is False

    @patch(
        "services.unified_dispatch.orchestration.shadow_mode_manager.ShadowModeOrchestrator"
    )
    def test_generate_and_store_suggestions_handles_none_dispatch_run_id(
        self, mock_orchestrator_class
    ):
        """Test : Gère dispatch_run_id None."""
        settings = MagicMock()
        manager = ShadowModeManager(settings)

        mock_orchestrator_instance = MagicMock()
        mock_orchestrator_instance.generate_and_store_shadow_suggestions.return_value = 0
        mock_orchestrator_class.return_value = mock_orchestrator_instance

        problem = {"bookings": [MagicMock()]}
        assignments = [MagicMock()]

        result = manager.generate_and_store_suggestions(
            dispatch_run_id=None,
            problem=problem,
            final_assignments=assignments,
            used_heuristic=True,
            used_solver=True,
        )

        assert result is False
        mock_orchestrator_instance.generate_and_store_shadow_suggestions.assert_called_once_with(
            dispatch_run_id=None,
            problem=problem,
            final_assignments=assignments,
            used_heuristic=True,
            used_solver=True,
        )
