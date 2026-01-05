"""Tests pour l'agrégat DispatchRun."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from dispatch.domain.dispatch_run import DispatchRun
from dispatch.domain.dispatch_run_id import DispatchRunId
from dispatch.domain.value_objects import DispatchMetrics, DispatchStatus

# Constantes pour les tests
TEST_COMPANY_ID = 1
TEST_DAY = date(2025, 1, 15)


class TestDispatchRunAggregate:
    """Tests pour l'agrégat DispatchRun."""

    def test_create_dispatch_run(self):
        """Test création d'un dispatch run."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("PENDING"),
        )

        assert dispatch_run.id.value == 1
        assert dispatch_run.company_id == TEST_COMPANY_ID
        assert dispatch_run.status.is_pending() is True

    def test_dispatch_run_start(self):
        """Test démarrage d'un dispatch run."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("PENDING"),
        )

        dispatch_run.start()
        assert dispatch_run.status.is_running() is True
        assert dispatch_run.started_at is not None

    def test_dispatch_run_start_invalid_status(self):
        """Test qu'on ne peut pas démarrer un dispatch déjà en cours."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("RUNNING"),
        )

        with pytest.raises(ValueError, match="Cannot start dispatch"):
            dispatch_run.start()

    def test_dispatch_run_complete(self):
        """Test complétion d'un dispatch run."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("RUNNING"),
            started_at=datetime.now(),
        )

        metrics = DispatchMetrics(
            assignments_count=10,
            unassigned_count=2,
            total_distance_km=150.5,
            total_duration_minutes=120,
            average_wait_time_minutes=5.5,
        )

        dispatch_run.complete(metrics=metrics)
        assert dispatch_run.status.is_completed() is True
        assert dispatch_run.completed_at is not None
        assert dispatch_run.metrics == metrics

    def test_dispatch_run_complete_invalid_status(self):
        """Test qu'on ne peut pas compléter un dispatch en attente."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("PENDING"),
        )

        with pytest.raises(ValueError, match="Cannot complete dispatch"):
            dispatch_run.complete()

    def test_dispatch_run_fail(self):
        """Test échec d'un dispatch run."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("RUNNING"),
            started_at=datetime.now(),
        )

        dispatch_run.fail(reason="Timeout")
        assert dispatch_run.status.is_failed() is True
        assert dispatch_run.completed_at is not None
        assert dispatch_run.config is not None
        assert dispatch_run.config.get("error") == "Timeout"

    def test_dispatch_run_fail_final_status(self):
        """Test qu'on ne peut pas faire échouer un dispatch déjà final."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("COMPLETED"),
            completed_at=datetime.now(),
        )

        with pytest.raises(ValueError, match="Cannot fail dispatch"):
            dispatch_run.fail()

    def test_dispatch_run_validate(self):
        """Test validation des invariants."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("PENDING"),
        )

        assert dispatch_run.validate() is True

    def test_dispatch_run_validate_invalid_completed_at(self):
        """Test validation échoue si completed_at présent sans statut final."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("RUNNING"),
            completed_at=datetime.now(),  # Invalide : RUNNING mais completed_at présent
        )

        assert dispatch_run.validate() is False

    def test_dispatch_run_validate_invalid_started_at(self):
        """Test validation échoue si started_at présent avec statut PENDING."""
        dispatch_run = DispatchRun(
            id=DispatchRunId(1),
            company_id=TEST_COMPANY_ID,
            day=TEST_DAY,
            status=DispatchStatus("PENDING"),
            started_at=datetime.now(),  # Invalide : PENDING mais started_at présent
        )

        assert dispatch_run.validate() is False

    def test_dispatch_metrics(self):
        """Test création et validation de DispatchMetrics."""
        metrics = DispatchMetrics(
            assignments_count=10,
            unassigned_count=2,
            total_distance_km=150.5,
            total_duration_minutes=120,
            average_wait_time_minutes=5.5,
        )

        assert metrics.total_bookings() == 12
        assert metrics.assignment_rate() == pytest.approx(10 / 12, rel=0.01)

    def test_dispatch_metrics_invalid_negative(self):
        """Test que DispatchMetrics rejette les valeurs négatives."""
        with pytest.raises(ValueError, match="must be non-negative"):
            DispatchMetrics(
                assignments_count=-1,
                unassigned_count=0,
                total_distance_km=0.0,
                total_duration_minutes=0,
                average_wait_time_minutes=0.0,
            )

    def test_dispatch_status_transitions(self):
        """Test les transitions de statut valides."""
        status_pending = DispatchStatus("PENDING")
        status_running = DispatchStatus("RUNNING")
        status_completed = DispatchStatus("COMPLETED")
        status_failed = DispatchStatus("FAILED")

        assert status_pending.can_start() is True
        assert status_running.can_complete() is True
        assert status_completed.is_final() is True
        assert status_failed.is_final() is True
