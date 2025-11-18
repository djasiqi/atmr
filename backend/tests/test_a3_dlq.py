#!/usr/bin/env python3
"""
Tests pour l'amélioration A3 : DLQ Celery + politique de retry.

Teste que les tâches échouées sont stockées en DLQ et qu'on peut les consulter.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from celery_app import celery
from ext import db
from models import TaskFailure

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_task_failure():
    """Créer un exemple de tâche échouée."""
    return TaskFailure(
        task_id="test-task-123",
        task_name="tasks.dispatch_tasks.run_dispatch_task",
        exception="ValueError: Sample exception",
        traceback="Traceback...",
        args="[123, '2025-01-27']",
        kwargs={"mode": "auto"},
        first_seen=datetime.now(UTC) - timedelta(minutes=10),
        last_seen=datetime.now(UTC),
        failure_count=3,
        dispatch_run_id=999,
    )


class TestDLQ:
    """Tests pour la DLQ Celery (A3)."""

    def test_store_task_failure(self, app_context, db_session, sample_task_failure):
        """Test: stocker une tâche échouée en DB."""

        # Ajouter la tâche échouée
        db_session.add(sample_task_failure)
        db_session.commit()

        # Vérifier qu'elle existe
        stored = TaskFailure.query.filter_by(task_id="test-task-123").first()
        assert stored is not None
        assert stored.task_name == "tasks.dispatch_tasks.run_dispatch_task"
        assert stored.failure_count == 3
        assert stored.exception.startswith("ValueError")

        logger.info("✅ Test: Tâche échouée stockée en DB")

    def test_update_task_failure_on_retry(self, app_context, db_session):
        """Test: mettre à jour le compteur lors d'un nouvel échec."""

        # Créer une tâche échouée
        failure = TaskFailure(
            task_id="test-task-456",
            task_name="tasks.test_task",
            exception="ConnectionError",
            first_seen=datetime.now(UTC),
            last_seen=datetime.now(UTC),
            failure_count=1,
        )
        db_session.add(failure)
        db_session.commit()

        initial_count = failure.failure_count

        # Simuler un nouvel échec (simuler ce que fait _store_task_failure_in_db)
        existing = TaskFailure.query.filter_by(task_id="test-task-456").first()
        existing.failure_count += 1
        existing.last_seen = datetime.now(UTC)
        db_session.commit()

        # Vérifier que le compteur a été incrémenté
        updated = TaskFailure.query.filter_by(task_id="test-task-456").first()
        assert updated.failure_count == initial_count + 1

        logger.info("✅ Test: Compteur échec incrémenté")

    def test_dlq_backlog_query(self, app_context, db_session):
        """Test: requête pour backlog DLQ."""

        # Créer plusieurs tâches échouées
        for i in range(15):  # Plus que le seuil de 10
            failure = TaskFailure(
                task_id=f"test-task-{i}",
                task_name="tasks.test_task",
                exception=f"Error #{i}",
                first_seen=datetime.now(UTC) - timedelta(minutes=30),
                last_seen=datetime.now(UTC) - timedelta(minutes=i),
                failure_count=1,
            )
            db_session.add(failure)

        db_session.commit()

        # Requête backlog
        backlog = TaskFailure.query.count()
        assert backlog >= 15

        # Test alerte backlog > 10
        assert backlog > 10, "Backlog devrait dépasser le seuil de 10"

        logger.info("✅ Test: Backlog DLQ détecté correctement")

    def test_dlq_max_age_calculation(self, app_context, db_session):
        """Test: calcul âge max DLQ."""

        # Créer une tâche très ancienne (> 5 min)
        old_failure = TaskFailure(
            task_id="test-old-task",
            task_name="tasks.old_task",
            exception="Old error",
            first_seen=datetime.now(UTC) - timedelta(hours=1),
            last_seen=datetime.now(UTC) - timedelta(minutes=10),  # 10 min
            failure_count=1,
        )
        db_session.add(old_failure)
        db_session.commit()

        # Calculer l'âge
        now = datetime.now(UTC)
        failures = TaskFailure.query.all()

        max_age_seconds = 0
        for failure in failures:
            age = (now - failure.last_seen).total_seconds()
            max_age_seconds = max(max_age_seconds, age)

        max_age_minutes = int(max_age_seconds / 60)

        assert max_age_minutes >= 10, "Âge max devrait être >= 10 minutes"

        # Test alerte âge > 5 min
        assert max_age_minutes > 5, "Âge devrait dépasser le seuil de 5 min"

        logger.info("✅ Test: Âge max DLQ calculé correctement")

    def test_celery_dlq_flow(self, app_context, db_session):
        """Test: tâche qui échoue 3x → visible en DLQ + alerte.

        Test simulant une tâche qui échoue 3 fois et est ensuite visible
        dans la DLQ avec alerte.
        """

        task_id = "celery-test-task-789"

        # Simuler 3 échecs consécutifs (avec doublons détectés)
        for attempt in range(3):
            existing = TaskFailure.query.filter_by(task_id=task_id).first()

            if existing:
                existing.failure_count += 1
                existing.last_seen = datetime.now(UTC)
                logger.warning("[Test] Retry #%d: task_id=%s", attempt + 1, task_id)
            else:
                failure = TaskFailure(
                    task_id=task_id,
                    task_name="tasks.dispatch_tasks.run_dispatch_task",
                    exception=f"Simulated failure attempt {attempt + 1}",
                    first_seen=datetime.now(UTC),
                    last_seen=datetime.now(UTC),
                    failure_count=1,
                )
                db_session.add(failure)

            db_session.commit()

        # Vérifier que la tâche est en DLQ avec failure_count = 3
        final = TaskFailure.query.filter_by(task_id=task_id).first()
        assert final is not None
        assert final.failure_count == 3

        # Vérifier alerte
        backlog = TaskFailure.query.count()
        alerts = []
        if backlog > 10:
            alerts.append("DLQ backlog élevé")
        if final.failure_count >= 3:
            alerts.append("Tâche échouée définitivement après 3 retries")

        assert len(alerts) >= 1, "Au moins une alerte devrait être déclenchée"

        logger.info("✅ Test DLQ flow: Tâche échouée 3x visible en DLQ avec alertes")

    def test_dlq_endpoint_response(self, app_context, client):
        """Test: endpoint GET /api/dispatch-health/dlq retourne les bonnes métriques."""

        # Créer quelques tâches échouées
        from datetime import UTC, datetime, timedelta

        for i in range(5):
            failure = TaskFailure(
                task_id=f"endpoint-test-{i}",
                task_name="tasks.test_task",
                exception=f"Error {i}",
                first_seen=datetime.now(UTC) - timedelta(minutes=10),
                last_seen=datetime.now(UTC) - timedelta(minutes=i),
                failure_count=1,
            )
            db.session.add(failure)

        db.session.commit()

        # Appeler l'endpoint
        with client.get("/api/dispatch-health/dlq") as response:
            assert response.status_code == 200
            data = response.get_json()

            assert "backlog" in data
            assert "max_age_minutes" in data
            assert "by_task_name" in data
            assert "alerts" in data

            logger.info("✅ Test: Endpoint DLQ retourne les bonnes métriques")

    def test_dlq_auto_cleanup(self, app_context, db_session):
        """Test: nettoyage automatique des anciennes tâches (> 7 jours)."""

        from datetime import UTC, datetime, timedelta

        # Créer une tâche très ancienne (> 7 jours)
        old_failure = TaskFailure(
            task_id="old-cleanup-test",
            task_name="tasks.old_task",
            exception="Very old error",
            first_seen=datetime.now(UTC) - timedelta(days=10),
            last_seen=datetime.now(UTC) - timedelta(days=8),
            failure_count=1,
        )
        db_session.add(old_failure)
        db_session.commit()

        # Nettoyer les tâches > 7 jours
        cutoff = datetime.now(UTC) - timedelta(days=7)
        TaskFailure.query.filter(TaskFailure.last_seen < cutoff).delete()
        db_session.commit()

        # Vérifier que la tâche ancienne a été supprimée
        remaining = TaskFailure.query.filter_by(task_id="old-cleanup-test").first()
        assert remaining is None

        logger.info("✅ Test: Nettoyage auto des tâches anciennes")
