# backend/tests/services/unified_dispatch/orchestration/test_metrics_finalizer.py
"""Tests unitaires pour MetricsFinalizer.

Tests pour :
- _analyze_unassigned_reasons : Analyse des raisons de non-assignation
- _serialize_assignment, _serialize_booking, _serialize_driver : Sérialisation
- _record_prometheus_metrics : Enregistrement Prometheus
- finalize : Finalisation complète des métriques
"""

from __future__ import annotations  # noqa: I001

import pytest
from unittest.mock import MagicMock, patch

from factories import CompanyFactory, DispatchRunFactory
from services.unified_dispatch.orchestration.metrics_finalizer import (
    MetricsFinalizer,
)


class TestAnalyzeUnassignedReasons:
    """Tests pour la méthode _analyze_unassigned_reasons."""

    @patch(
        "services.unified_dispatch.orchestration.metrics_finalizer.UnassignedAnalyzer"
    )
    def test_analyze_unassigned_reasons_success(self, mock_analyzer_class):
        """Test : Analyse réussie avec UnassignedAnalyzer."""
        finalizer = MetricsFinalizer()

        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {
            1: ["no_driver_available"],
            2: ["time_conflict"],
        }
        mock_analyzer_class.return_value = mock_analyzer_instance

        problem = {"bookings": [MagicMock()]}
        assignments = [MagicMock()]
        unassigned_ids = [1, 2]

        result = finalizer._analyze_unassigned_reasons(
            problem, assignments, unassigned_ids
        )

        assert result == {"1": ["no_driver_available"], "2": ["time_conflict"]}
        mock_analyzer_instance.analyze.assert_called_once_with(
            problem, assignments, unassigned_ids
        )

    @patch(
        "services.unified_dispatch.orchestration.metrics_finalizer.UnassignedAnalyzer",
        side_effect=ImportError("Module not found"),
    )
    def test_analyze_unassigned_reasons_import_error(self):
        """Test : Gestion de ImportError (retourne dict vide)."""
        finalizer = MetricsFinalizer()

        problem = {"bookings": []}
        assignments = []
        unassigned_ids = []

        result = finalizer._analyze_unassigned_reasons(
            problem, assignments, unassigned_ids
        )

        assert result == {}


class TestSerialize:
    """Tests pour les méthodes de sérialisation."""

    def test_serialize_assignment_with_to_dict(self):
        """Test : Sérialisation avec to_dict()."""
        finalizer = MetricsFinalizer()

        assignment = MagicMock()
        assignment.to_dict.return_value = {"booking_id": 1, "driver_id": 2}

        result = finalizer._serialize_assignment(assignment)

        assert result == {"booking_id": 1, "driver_id": 2}

    def test_serialize_assignment_without_to_dict(self):
        """Test : Sérialisation sans to_dict() (fallback)."""
        finalizer = MetricsFinalizer()

        assignment = MagicMock()
        del assignment.to_dict
        assignment.booking_id = 1
        assignment.driver_id = 2
        assignment.dispatch_run_id = 3

        result = finalizer._serialize_assignment(assignment)

        assert result == {
            "booking_id": 1,
            "driver_id": 2,
            "dispatch_run_id": 3,
        }

    def test_serialize_booking_with_to_dict(self):
        """Test : Sérialisation booking avec to_dict()."""
        finalizer = MetricsFinalizer()

        booking = MagicMock()
        booking.to_dict.return_value = {"id": 1, "pickup_lat": 45.0}

        result = finalizer._serialize_booking(booking)

        assert result == {"id": 1, "pickup_lat": 45.0}

    def test_serialize_booking_without_to_dict(self):
        """Test : Sérialisation booking sans to_dict() (fallback)."""
        finalizer = MetricsFinalizer()

        booking = MagicMock()
        del booking.to_dict
        booking.id = 1
        booking.pickup_lat = 45.0
        booking.pickup_lon = -73.0
        booking.dropoff_lat = 46.0
        booking.dropoff_lon = -74.0

        result = finalizer._serialize_booking(booking)

        assert result == {
            "id": 1,
            "pickup_lat": 45.0,
            "pickup_lon": -73.0,
            "dropoff_lat": 46.0,
            "dropoff_lon": -74.0,
        }

    def test_serialize_driver_with_to_dict(self):
        """Test : Sérialisation driver avec to_dict()."""
        finalizer = MetricsFinalizer()

        driver = MagicMock()
        driver.to_dict.return_value = {"id": 1, "current_lat": 45.0}

        result = finalizer._serialize_driver(driver)

        assert result == {"id": 1, "current_lat": 45.0}

    def test_serialize_driver_without_to_dict(self):
        """Test : Sérialisation driver sans to_dict() (fallback)."""
        finalizer = MetricsFinalizer()

        driver = MagicMock()
        del driver.to_dict
        driver.id = 1
        driver.current_lat = 45.0
        driver.current_lon = -73.0

        result = finalizer._serialize_driver(driver)

        assert result == {
            "id": 1,
            "current_lat": 45.0,
            "current_lon": -73.0,
        }


class TestRecordPrometheusMetrics:
    """Tests pour la méthode _record_prometheus_metrics."""

    @patch(
        "services.unified_dispatch.orchestration.metrics_finalizer.record_assignments_created"
    )
    @patch(
        "services.unified_dispatch.orchestration.metrics_finalizer.record_unassigned_count"
    )
    def test_record_prometheus_metrics_success(
        self, mock_record_unassigned, mock_record_assignments
    ):
        """Test : Enregistrement réussi avec Prometheus disponible."""
        finalizer = MetricsFinalizer()

        perf_metrics = MagicMock()
        perf_metrics.quality_score = 85.0
        perf_metrics.assignment_rate = 90.0
        perf_metrics.temporal_conflicts_count = 0
        perf_metrics.db_conflicts_count = 0
        perf_metrics.total_time = 1000
        perf_metrics.bookings_processed = 10
        perf_metrics.drivers_available = 5
        perf_metrics.data_collection_time = 100
        perf_metrics.heuristics_time = 200
        perf_metrics.solver_time = 0
        perf_metrics.persistence_time = 50

        problem = {"drivers": [MagicMock(), MagicMock()]}
        final_assignments = [MagicMock(), MagicMock()]
        unassigned_ids = [1, 2]

        finalizer._record_prometheus_metrics(
            company_id=1,
            dispatch_run_id=42,
            problem=problem,
            final_assignments=final_assignments,
            unassigned_ids=unassigned_ids,
            perf_metrics=perf_metrics,
            mode="auto",
        )

        # Vérifier que les fonctions record_* sont appelées
        mock_record_assignments.assert_called()
        mock_record_unassigned.assert_called()

    @patch(
        "services.unified_dispatch.orchestration.metrics_finalizer.record_assignments_created",
        side_effect=ImportError("Module not found"),
    )
    def test_record_prometheus_metrics_import_error(self):
        """Test : Gestion de ImportError (retourne silencieusement)."""
        finalizer = MetricsFinalizer()

        perf_metrics = MagicMock()
        perf_metrics.quality_score = 0
        perf_metrics.assignment_rate = 0

        problem = {}
        final_assignments = []
        unassigned_ids = []

        # Ne doit pas lever d'exception
        finalizer._record_prometheus_metrics(
            company_id=1,
            dispatch_run_id=42,
            problem=problem,
            final_assignments=final_assignments,
            unassigned_ids=unassigned_ids,
            perf_metrics=perf_metrics,
            mode="auto",
        )


class TestFinalize:
    """Tests pour la méthode finalize."""

    @patch("services.unified_dispatch.orchestration.metrics_finalizer.DispatchResult")
    @patch.object(MetricsFinalizer, "_record_prometheus_metrics")
    @patch.object(MetricsFinalizer, "_analyze_unassigned_reasons")
    def test_finalize_success(
        self, mock_analyze, mock_record_prometheus, mock_dispatch_result_class
    ):
        """Test : Finalisation réussie avec toutes les métriques."""
        finalizer = MetricsFinalizer()

        mock_analyze.return_value = {}
        mock_result_instance = MagicMock()
        mock_result_instance.to_dict.return_value = {"dispatch_run_id": 42}
        mock_dispatch_result_class.return_value = mock_result_instance

        company = CompanyFactory()
        dispatch_run = DispatchRunFactory()
        settings = MagicMock()
        settings.features.enable_heuristics = True
        settings.features.enable_solver = True
        settings.features.enable_rl = False
        settings.features.enable_clustering = False
        settings.features.enable_parallel_heuristics = False
        settings.to_dict.return_value = {}

        problem = {
            "bookings": [MagicMock(id=1)],
            "drivers": [MagicMock(id=1)],
        }
        final_assignments = [MagicMock(booking_id=1, driver_id=1)]
        unassigned_ids = [2]

        perf_collector = MagicMock()
        perf_metrics = MagicMock()
        perf_metrics.quality_score = 0
        perf_metrics.assignment_rate = 0
        perf_metrics.temporal_conflicts_count = 0
        perf_metrics.db_conflicts_count = 0
        perf_metrics.total_time = 0
        perf_metrics.bookings_processed = 0
        perf_metrics.drivers_available = 0
        perf_metrics.data_collection_time = 0
        perf_metrics.heuristics_time = 0
        perf_metrics.solver_time = 0
        perf_metrics.persistence_time = 0
        perf_collector.finalize.return_value = perf_metrics

        result = finalizer.finalize(
            company_id=company.id,
            problem=problem,
            final_assignments=final_assignments,
            unassigned_ids=unassigned_ids,
            dispatch_run=dispatch_run,
            settings=settings,
            mode="auto",
            regular_first=True,
            allow_emg=True,
            for_date="2025-01-14",
            day_str="2025-01-14",
            used_heuristic=True,
            used_solver=False,
            used_fallback=False,
            used_emergency_pass=False,
            h_res=None,
            s_res=None,
            perf_collector=perf_collector,
            should_apply_rl=False,
            kpi_monitor=None,
        )

        assert result == {"dispatch_run_id": 42}
        mock_analyze.assert_called_once()
        mock_record_prometheus.assert_called_once()

    def test_finalize_with_dispatch_run_none(self):
        """Test : Gestion de dispatch_run None."""
        finalizer = MetricsFinalizer()

        with (
            patch.object(finalizer, "_analyze_unassigned_reasons", return_value={}),
            patch.object(finalizer, "_record_prometheus_metrics"),
            patch(
                "services.unified_dispatch.orchestration.metrics_finalizer.DispatchResult"
            ) as mock_dispatch_result_class,
        ):
            mock_result_instance = MagicMock()
            mock_result_instance.to_dict.return_value = {"dispatch_run_id": None}
            mock_dispatch_result_class.return_value = mock_result_instance

            settings = MagicMock()
            settings.features.enable_heuristics = True
            settings.features.enable_solver = True
            settings.features.enable_rl = False
            settings.features.enable_clustering = False
            settings.features.enable_parallel_heuristics = False
            settings.to_dict.return_value = {}

            problem = {"bookings": [], "drivers": []}
            final_assignments = []
            unassigned_ids = []

            perf_collector = MagicMock()
            perf_metrics = MagicMock()
            perf_metrics.quality_score = 0
            perf_metrics.assignment_rate = 0
            perf_metrics.temporal_conflicts_count = 0
            perf_metrics.db_conflicts_count = 0
            perf_metrics.total_time = 0
            perf_metrics.bookings_processed = 0
            perf_metrics.drivers_available = 0
            perf_metrics.data_collection_time = 0
            perf_metrics.heuristics_time = 0
            perf_metrics.solver_time = 0
            perf_metrics.persistence_time = 0
            perf_collector.finalize.return_value = perf_metrics

            result = finalizer.finalize(
                company_id=1,
                problem=problem,
                final_assignments=final_assignments,
                unassigned_ids=unassigned_ids,
                dispatch_run=None,
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
                perf_collector=perf_collector,
                should_apply_rl=False,
                kpi_monitor=None,
            )

            assert result["dispatch_run_id"] is None
