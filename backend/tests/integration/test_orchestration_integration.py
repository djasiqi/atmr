# backend/tests/integration/test_orchestration_integration.py
"""Tests d'intégration pour les modules d'orchestration du dispatch.

Tests l'intégration complète de chaque module avec des données réelles
pour valider que le refactoring fonctionne correctement.
"""

from __future__ import annotations  # noqa: I001

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from factories import (
    BookingFactory,
    CompanyFactory,
    DispatchRunFactory,
    DriverFactory,
)
from services.unified_dispatch.orchestration.assignment_applier_wrapper import (
    AssignmentApplierWrapper,
)
from services.unified_dispatch.orchestration.clustering_manager import (
    ClusteringManager,
)
from services.unified_dispatch.orchestration.dispatch_orchestrator import (
    DispatchOrchestrator,
)
from services.unified_dispatch.orchestration.dispatch_run_manager import (
    DispatchRunManager,
)
from services.unified_dispatch.orchestration.initializer import DispatchInitializer
from services.unified_dispatch.orchestration.metrics_finalizer import (
    MetricsFinalizer,
)
from services.unified_dispatch.orchestration.pipeline_executor import (
    PipelineExecutor,
)
from services.unified_dispatch.orchestration.problem_builder import ProblemBuilder
from services.unified_dispatch.orchestration.result_builder import ResultBuilder
from services.unified_dispatch.orchestration.shadow_mode_manager import (
    ShadowModeManager,
)


class TestDispatchInitializerIntegration:
    """Tests d'intégration pour DispatchInitializer."""

    def test_find_and_validate_company_with_existing_company(self, db):
        """Test : Trouve et valide une Company existante."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        initializer = DispatchInitializer()
        result_company, error_result = initializer.find_and_validate_company(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
            raise_on_company_not_found=False,
        )

        assert result_company is not None
        assert result_company.id == company.id
        assert error_result is None

    def test_find_and_validate_company_not_found(self, db):
        """Test : Company inexistante retourne error_result."""
        initializer = DispatchInitializer()
        result_company, error_result = initializer.find_and_validate_company(
            company_id=999_999,
            for_date="2025-01-14",
            mode="auto",
            raise_on_company_not_found=False,
        )

        assert result_company is None
        assert error_result is not None
        assert error_result.get("meta", {}).get("reason") == "company_not_found"

    def test_configure_settings_with_overrides(self, db):
        """Test : Configuration avec overrides."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        initializer = DispatchInitializer()
        overrides = {"mode": "solver_only", "enable_clustering": False}

        settings, mode, _allow_emg, _is_fast_mode = initializer.configure_settings(
            company=company,
            mode="auto",
            custom_settings=None,
            allow_emergency=True,
            overrides=overrides,
        )

        assert settings is not None
        assert mode == "solver_only"


class TestDispatchRunManagerIntegration:
    """Tests d'intégration pour DispatchRunManager."""

    def test_create_or_reuse_creates_new_dispatch_run(self, db):
        """Test : Crée un nouveau DispatchRun."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        manager = DispatchRunManager()
        dispatch_run, error_result = manager.create_or_reuse(
            company=company,
            company_id=company.id,
            day_str="2025-01-14",
            mode="auto",
            regular_first=True,
            allow_emg=True,
            for_date="2025-01-14",
            existing_id=None,
        )

        assert error_result is None
        assert dispatch_run is not None
        assert dispatch_run.company_id == company.id
        assert dispatch_run.day.isoformat() == "2025-01-14"

    def test_create_or_reuse_reuses_existing_dispatch_run(self, db):
        """Test : Réutilise un DispatchRun existant."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        existing_run = DispatchRunFactory(company_id=company.id)
        db.session.add(existing_run)
        db.session.commit()

        manager = DispatchRunManager()
        dispatch_run, error_result = manager.create_or_reuse(
            company=company,
            company_id=company.id,
            day_str="2025-01-14",
            mode="auto",
            regular_first=True,
            allow_emg=True,
            for_date="2025-01-14",
            existing_id=existing_run.id,
        )

        assert error_result is None
        assert dispatch_run is not None
        assert dispatch_run.id == existing_run.id

    def test_update_status(self, db):
        """Test : Met à jour le statut d'un DispatchRun."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        manager = DispatchRunManager()
        manager.update_status(dispatch_run, "RUNNING")

        db.session.refresh(dispatch_run)
        assert dispatch_run.status == "RUNNING"


class TestProblemBuilderIntegration:
    """Tests d'intégration pour ProblemBuilder."""

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.get_available_bookings"
    )
    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.get_available_drivers_split"
    )
    def test_build_problem_with_real_data(
        self, mock_get_drivers, mock_get_bookings, db
    ):
        """Test : Construction du problème avec données réelles."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        booking1 = BookingFactory(company_id=company.id)
        booking1.pickup_lat = 46.2
        booking1.pickup_lon = 6.1
        booking1.dropoff_lat = 46.3
        booking1.dropoff_lon = 6.2
        booking1.scheduled_time = datetime.now(UTC) + timedelta(hours=1)

        booking2 = BookingFactory(company_id=company.id)
        booking2.pickup_lat = 46.2
        booking2.pickup_lon = 6.1
        booking2.dropoff_lat = 46.3
        booking2.dropoff_lon = 6.2
        booking2.scheduled_time = datetime.now(UTC) + timedelta(hours=2)

        driver = DriverFactory(company_id=company.id)
        driver.current_lat = 46.2
        driver.current_lon = 6.1

        db.session.add_all([booking1, booking2, driver])
        db.session.commit()

        mock_get_bookings.return_value = [booking1, booking2]
        mock_get_drivers.return_value = ([driver], [])

        settings = MagicMock()
        settings.features.enable_geographic_validation = True

        builder = ProblemBuilder()
        problem, error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=None,
            settings=settings,
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=None,
        )

        assert error_result is None
        assert problem is not None
        assert "bookings" in problem
        assert "drivers" in problem
        assert len(problem["bookings"]) == 2
        assert len(problem["drivers"]) == 1


class TestClusteringManagerIntegration:
    """Tests d'intégration pour ClusteringManager."""

    def test_should_use_clustering_with_threshold(self, db):
        """Test : Décision de clustering basée sur le seuil."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        manager = ClusteringManager()
        settings = MagicMock()
        settings.features.enable_clustering = True
        settings.clustering.min_bookings_threshold = 10

        problem_small = {"bookings": [MagicMock() for _ in range(5)]}
        problem_large = {"bookings": [MagicMock() for _ in range(15)]}

        assert manager.should_use_clustering(problem_small, settings) is False
        assert manager.should_use_clustering(problem_large, settings) is True


class TestPipelineExecutorIntegration:
    """Tests d'intégration pour PipelineExecutor."""

    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ShadowModeManager"
    )
    @patch(
        "services.unified_dispatch.orchestration.pipeline_executor.ClusteringManager.should_use_clustering"
    )
    def test_execute_pipeline_without_clustering(
        self, mock_should_cluster, mock_shadow, db
    ):
        """Test : Exécution du pipeline sans clustering."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        mock_should_cluster.return_value = False

        mock_shadow_instance = MagicMock()
        mock_shadow_instance.should_apply_rl.return_value = (False, None)
        mock_shadow_instance.generate_and_store_suggestions.return_value = False
        mock_shadow.return_value = mock_shadow_instance

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

        executor = PipelineExecutor()
        (
            final_assignments,
            unassigned_ids,
            _used_heuristic,
            _used_solver,
            _used_fallback,
            _used_emergency_pass,
            should_apply_rl,
            _quality_score_pre_apply,
            error_result,
        ) = executor.execute(
            company=company,
            company_id=company.id,
            problem=problem,
            dispatch_run=None,
            settings=settings,
            mode="auto",
            regular_first=False,
            allow_emg=False,
            is_fast_mode=False,
            perf_collector=None,
        )

        assert error_result is None
        assert isinstance(final_assignments, list)
        assert isinstance(unassigned_ids, list)
        assert isinstance(should_apply_rl, bool)


class TestShadowModeManagerIntegration:
    """Tests d'intégration pour ShadowModeManager."""

    @patch(
        "services.unified_dispatch.orchestration.shadow_mode_manager.ShadowModeOrchestrator"
    )
    def test_should_apply_rl_integration(self, mock_orchestrator_class):
        """Test : Décision d'appliquer RL."""
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


class TestAssignmentApplierWrapperIntegration:
    """Tests d'intégration pour AssignmentApplierWrapper."""

    @patch(
        "services.unified_dispatch.orchestration.assignment_applier_wrapper.AssignmentApplier"
    )
    def test_apply_assignments_integration(self, mock_applier_class):
        """Test : Application des assignations."""
        wrapper = AssignmentApplierWrapper()

        mock_applier_instance = MagicMock()
        mock_applier_class.return_value = mock_applier_instance

        company = CompanyFactory()
        assignments = [MagicMock()]

        wrapper.apply(
            company=company,
            final_assignments=assignments,
            dispatch_run_id=42,
            perf_collector=None,
        )

        mock_applier_instance.apply_and_emit.assert_called_once_with(
            company, assignments, 42
        )


class TestMetricsFinalizerIntegration:
    """Tests d'intégration pour MetricsFinalizer."""

    @patch(
        "services.unified_dispatch.orchestration.metrics_finalizer.collect_quality_metrics"
    )
    @patch(
        "services.unified_dispatch.orchestration.metrics_finalizer.collect_analytics_metrics"
    )
    def test_finalize_metrics_integration(self, mock_analytics, mock_quality, db):
        """Test : Finalisation complète des métriques."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        mock_quality.return_value = {}
        mock_analytics.return_value = {}

        finalizer = MetricsFinalizer()

        problem = {
            "bookings": [MagicMock()],
            "drivers": [MagicMock()],
        }

        settings = MagicMock()

        result = finalizer.finalize(
            company_id=company.id,
            problem=problem,
            final_assignments=[],
            unassigned_ids=[1, 2],
            dispatch_run=dispatch_run,
            settings=settings,
            mode="auto",
            regular_first=True,
            allow_emg=True,
            for_date="2025-01-14",
            day_str="2025-01-14",
            used_heuristic=False,
            used_solver=False,
            used_fallback=False,
            used_emergency_pass=False,
            h_res=None,
            s_res=None,
            perf_collector=None,
            should_apply_rl=False,
            kpi_monitor=None,
        )

        assert result is not None
        assert "assignments" in result
        assert "unassigned" in result
        assert "meta" in result
        assert "debug" in result


class TestResultBuilderIntegration:
    """Tests d'intégration pour ResultBuilder."""

    @patch("services.unified_dispatch.orchestration.result_builder.DispatchResult")
    def test_build_result_integration(self, mock_dispatch_result_class):
        """Test : Construction du résultat complet."""
        builder = ResultBuilder()

        mock_result_instance = MagicMock()
        mock_result_instance.to_dict.return_value = {
            "dispatch_run_id": 42,
            "assignments": [{"booking_id": 1, "driver_id": 2}],
            "unassigned": [3],
            "bookings": [{"id": 1}],
            "drivers": [{"id": 1}],
            "meta": {"assignments_count": 1},
            "debug": {"used_heuristic": True},
        }
        mock_dispatch_result_class.return_value = mock_result_instance

        assignment = MagicMock()
        assignment.booking_id = 1
        assignment.driver_id = 2

        booking = MagicMock()
        booking.id = 1

        driver = MagicMock()
        driver.id = 1

        result = builder.build(
            dispatch_run_id=42,
            assignments=[assignment],
            unassigned_ids=[3],
            bookings=[booking],
            drivers=[driver],
            meta={"assignments_count": 1},
            debug={"used_heuristic": True},
        )

        assert result is not None
        assert result["dispatch_run_id"] == 42
        assert len(result["assignments"]) == 1


class TestDispatchOrchestratorFullIntegration:
    """Tests d'intégration complets pour DispatchOrchestrator."""

    @patch(
        "services.unified_dispatch.orchestration.dispatch_orchestrator.RedisLockManager"
    )
    @patch(
        "services.unified_dispatch.orchestration.dispatch_orchestrator.performance_metrics"
    )
    def test_execute_full_flow(self, mock_perf_metrics, mock_lock_manager, db):
        """Test : Flux complet de bout en bout."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        # Mock Redis lock
        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire_lock.return_value = True
        mock_lock_manager.return_value = mock_lock_instance

        # Mock performance metrics
        mock_perf_collector = MagicMock()
        mock_perf_metrics.DispatchMetricsCollector.return_value = mock_perf_collector

        orchestrator = DispatchOrchestrator()

        # Exécuter avec des données minimales
        result = orchestrator.execute(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
            regular_first=True,
            allow_emergency=True,
        )

        # Vérifier que le résultat est cohérent
        assert result is not None
        assert "assignments" in result
        assert "unassigned" in result
        assert "meta" in result
        assert "debug" in result

        # Vérifier que le lock a été libéré
        mock_lock_instance.release_lock.assert_called_once()
