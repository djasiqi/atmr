# backend/tests/services/unified_dispatch/orchestration/test_pipeline_executor.py
"""Tests unitaires pour PipelineExecutor.

Tests pour :
- _extend_unique : Ajout d'assignations uniques
- _remaining_ids_from : Calcul des IDs restants
- _filter_problem : Filtrage du problème
- execute : Exécution complète du pipeline
"""

from __future__ import annotations  # noqa: I001

import pytest
from unittest.mock import MagicMock, patch

from factories import CompanyFactory, DispatchRunFactory
from services.unified_dispatch.orchestration.pipeline_executor import (
    PipelineExecutor,
)


class TestExtendUnique:
    """Tests pour la méthode _extend_unique."""

    def test_adds_unique_assignments(self):
        """Test : Ajoute des assignations uniques."""
        executor = PipelineExecutor()
        final_assignments = []
        assigned_set = set()

        assignment1 = MagicMock()
        assignment1.booking_id = 1
        assignment2 = MagicMock()
        assignment2.booking_id = 2

        executor._extend_unique(
            [assignment1, assignment2], final_assignments, assigned_set
        )

        assert len(final_assignments) == 2
        assert 1 in assigned_set
        assert 2 in assigned_set

    def test_skips_duplicate_assignments(self):
        """Test : Ignore les assignations en double."""
        executor = PipelineExecutor()
        final_assignments = []
        assigned_set = {1}  # booking_id 1 déjà assigné

        assignment1 = MagicMock()
        assignment1.booking_id = 1
        assignment2 = MagicMock()
        assignment2.booking_id = 2

        executor._extend_unique(
            [assignment1, assignment2], final_assignments, assigned_set
        )

        assert len(final_assignments) == 1
        assert final_assignments[0].booking_id == 2
        assert 1 in assigned_set
        assert 2 in assigned_set

    def test_handles_none_booking_id(self):
        """Test : Gère les assignations sans booking_id."""
        executor = PipelineExecutor()
        final_assignments = []
        assigned_set = set()

        assignment1 = MagicMock()
        assignment1.booking_id = None
        assignment2 = MagicMock()
        assignment2.booking_id = 2

        executor._extend_unique(
            [assignment1, assignment2], final_assignments, assigned_set
        )

        assert len(final_assignments) == 1
        assert final_assignments[0].booking_id == 2


class TestRemainingIdsFrom:
    """Tests pour la méthode _remaining_ids_from."""

    def test_returns_unassigned_booking_ids(self):
        """Test : Retourne les IDs de bookings non assignés."""
        executor = PipelineExecutor()
        assigned_set = {1}  # booking_id 1 déjà assigné

        booking1 = MagicMock()
        booking1.id = 1
        booking2 = MagicMock()
        booking2.id = 2
        booking3 = MagicMock()
        booking3.id = 3

        problem = {"bookings": [booking1, booking2, booking3]}

        result = executor._remaining_ids_from(problem, assigned_set)

        assert result == [2, 3]

    def test_returns_empty_list_when_all_assigned(self):
        """Test : Retourne liste vide si tous assignés."""
        executor = PipelineExecutor()
        assigned_set = {1, 2, 3}

        booking1 = MagicMock()
        booking1.id = 1
        booking2 = MagicMock()
        booking2.id = 2

        problem = {"bookings": [booking1, booking2]}

        result = executor._remaining_ids_from(problem, assigned_set)

        assert result == []

    def test_handles_missing_bookings_key(self):
        """Test : Gère l'absence de clé bookings."""
        executor = PipelineExecutor()
        assigned_set = set()

        problem = {}

        result = executor._remaining_ids_from(problem, assigned_set)

        assert result == []


class TestFilterProblem:
    """Tests pour la méthode _filter_problem."""

    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.data.build_vrptw_problem"
    )
    def test_filters_problem_by_booking_ids(self, mock_build, db):
        """Test : Filtre le problème par IDs de bookings."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        booking1 = MagicMock()
        booking1.id = 1
        booking2 = MagicMock()
        booking2.id = 2
        booking3 = MagicMock()
        booking3.id = 3

        driver = MagicMock()

        problem_dict = {
            "bookings": [booking1, booking2, booking3],
            "drivers": [driver],
            "base_time": 1000,
            "for_date": "2025-01-14",
            "dispatch_run_id": 42,
        }

        settings = MagicMock()

        mock_build.return_value = {
            "bookings": [booking1, booking2],
            "drivers": [driver],
        }

        executor = PipelineExecutor()
        result = executor._filter_problem(problem_dict, [1, 2], company, settings)

        mock_build.assert_called_once_with(
            company,
            [booking1, booking2],
            [driver],
            settings=settings,
            base_time=1000,
            for_date="2025-01-14",
        )
        assert result["dispatch_run_id"] == 42


class TestExecute:
    """Tests pour la méthode execute."""

    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ShadowModeOrchestrator"
    )
    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.data.get_available_drivers_split"
    )
    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ClusteringManager.should_use_clustering"
    )
    def test_execute_without_clustering(
        self, mock_should_cluster, mock_get_drivers, mock_shadow, db
    ):
        """Test : Exécution sans clustering (pipeline direct)."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        mock_should_cluster.return_value = False
        mock_get_drivers.return_value = ([], [])

        booking = MagicMock()
        booking.id = 1
        driver = MagicMock()

        problem = {
            "bookings": [booking],
            "drivers": [driver],
        }

        settings = MagicMock()
        settings.features.enable_heuristics = False
        settings.features.enable_solver = False

        mock_shadow_instance = MagicMock()
        mock_shadow_instance.should_apply_rl_with_guards.return_value = (False, None)
        mock_shadow_instance.generate_and_store_shadow_suggestions.return_value = False
        mock_shadow.return_value = mock_shadow_instance

        executor = PipelineExecutor()
        (
            final_assignments,
            unassigned_ids,
            used_heuristic,
            used_solver,
            used_fallback,
            used_emergency_pass,
            should_apply_rl,
            quality_score_pre_apply,
            error_result,
        ) = executor.execute(
            company=company,
            company_id=company.id,
            problem=problem,
            dispatch_run=dispatch_run,
            settings=settings,
            mode="auto",
            regular_first=False,
            allow_emg=False,
            is_fast_mode=False,
            perf_collector=None,
        )

        assert final_assignments == []
        assert unassigned_ids == [1]
        assert used_heuristic is False
        assert used_solver is False
        assert used_fallback is False
        assert used_emergency_pass is False
        assert should_apply_rl is False
        assert quality_score_pre_apply is None
        assert error_result is None

    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ShadowModeOrchestrator"
    )
    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ClusteringManager.dispatch_zones"
    )
    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ClusteringManager.create_zones"
    )
    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ClusteringManager.should_use_clustering"
    )
    def test_execute_with_clustering(
        self,
        mock_should_cluster,
        mock_create_zones,
        mock_dispatch_zones,
        mock_shadow,
        db,
    ):
        """Test : Exécution avec clustering activé."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        mock_should_cluster.return_value = True
        mock_create_zones.return_value = [MagicMock(), MagicMock()]  # 2 zones

        assignment = MagicMock()
        assignment.booking_id = 1
        mock_dispatch_zones.return_value = {
            "assignments": [assignment],
            "unassigned": [],
        }

        booking = MagicMock()
        booking.id = 1
        problem = {
            "bookings": [booking],
            "drivers": [MagicMock()],
        }

        settings = MagicMock()
        settings.features.enable_clustering = True

        mock_shadow_instance = MagicMock()
        mock_shadow_instance.should_apply_rl_with_guards.return_value = (False, None)
        mock_shadow_instance.generate_and_store_shadow_suggestions.return_value = False
        mock_shadow.return_value = mock_shadow_instance

        executor = PipelineExecutor()
        (
            final_assignments,
            _unassigned_ids,
            used_heuristic,
            used_solver,
            used_fallback,
            _used_emergency_pass,
            _should_apply_rl,
            _quality_score_pre_apply,
            error_result,
        ) = executor.execute(
            company=company,
            company_id=company.id,
            problem=problem,
            dispatch_run=dispatch_run,
            settings=settings,
            mode="auto",
            regular_first=False,
            allow_emg=False,
            is_fast_mode=False,
            perf_collector=None,
        )

        assert len(final_assignments) == 1
        assert final_assignments[0].booking_id == 1
        assert used_heuristic is True
        assert used_solver is True
        assert used_fallback is True
        assert error_result is None
