# backend/tests/services/test_notification_targets.py
"""Tests anti-régression pour compute_notification_targets (exclude_actor).

P0: Chauffeur qui change le statut ne doit JAMAIS recevoir de push.
"""

from __future__ import annotations

import pytest

from services.notifications.notification_targets import (
    NotificationTargets,
    compute_notification_targets,
)


class TestComputeNotificationTargets:
    """Tests du routage centralisé booking_updated."""

    def test_actor_driver_status_in_progress_company_only(self) -> None:
        """Cas 1: actor=driver, status=in_progress -> company notified, driver NOT push."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role="driver",
            actor_id=33,
            status="in_progress",
        )
        assert targets.notify_company_socket is True
        assert targets.notify_company_push is True
        assert targets.notify_driver_socket is False
        assert targets.notify_driver_push is False
        assert targets.exclude_driver_id == 33

    def test_actor_driver_status_en_route_company_only(self) -> None:
        """actor=driver, status=en_route -> company only."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role="driver",
            actor_id=33,
            status="en_route",
        )
        assert targets.notify_company_socket is True
        assert targets.notify_driver_push is False

    def test_actor_driver_status_completed_company_only(self) -> None:
        """actor=driver, status=completed -> company only."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role="driver",
            actor_id=33,
            status="completed",
        )
        assert targets.notify_company_push is True
        assert targets.notify_driver_push is False

    def test_actor_company_status_assigned_driver_notified(self) -> None:
        """Cas 2: actor=company, status=assigned -> driver notified (socket+push)."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role="company",
            actor_id=42,
            status="assigned",
        )
        assert targets.notify_driver_socket is True
        assert targets.notify_driver_push is True
        assert targets.notify_company_socket is True
        assert targets.notify_company_push is False
        assert targets.exclude_driver_id is None

    def test_actor_company_status_en_route_driver_notified(self) -> None:
        """actor=company modifie horaire/adresse -> driver notified."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role="company",
            actor_id=42,
            status="en_route",
        )
        assert targets.notify_driver_socket is True
        assert targets.notify_driver_push is True

    def test_actor_absent_status_in_progress_no_driver_push(self) -> None:
        """Cas 3: actor absent, status=in_progress -> pas push chauffeur (conservateur)."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role=None,
            actor_id=None,
            status="in_progress",
        )
        assert targets.notify_company_socket is True
        assert targets.notify_company_push is True
        assert targets.notify_driver_socket is True
        assert targets.notify_driver_push is False
        assert targets.exclude_driver_id == 33

    def test_source_driver_api_company_only(self) -> None:
        """source=driver_api sans actor_role -> company only (fallback)."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role=None,
            actor_id=None,
            status="in_progress",
            source="driver_api",
        )
        assert targets.notify_company_socket is True
        assert targets.notify_company_push is True
        assert targets.notify_driver_socket is False
        assert targets.notify_driver_push is False

    def test_source_company_api_driver_notified(self) -> None:
        """source=company_api sans actor_role -> driver notified."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role=None,
            actor_id=None,
            status="assigned",
            source="company_api",
        )
        assert targets.notify_driver_socket is True
        assert targets.notify_driver_push is True

    def test_actor_absent_status_cancelled_driver_push(self) -> None:
        """actor absent, status=cancelled -> driver peut recevoir push (changement important)."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role=None,
            actor_id=None,
            status="cancelled",
        )
        assert targets.notify_driver_push is True
        assert targets.exclude_driver_id is None

    def test_actor_driver_different_driver_not_excluded(self) -> None:
        """actor=driver A, booking driver=B -> B reçoit (changement par autre chauffeur)."""
        targets = compute_notification_targets(
            driver_id=33,
            company_id=42,
            actor_role="driver",
            actor_id=99,  # autre chauffeur
            status="in_progress",
        )
        # actor_id != driver_id -> driver 33 est le destinataire, pas l'acteur
        assert targets.notify_driver_socket is True
        assert targets.notify_driver_push is True
        assert targets.exclude_driver_id is None
