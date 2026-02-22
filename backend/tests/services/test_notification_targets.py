# backend/tests/services/test_notification_targets.py
"""Tests anti-regression pour le routage de notifications.

- compute_notification_targets : legacy, booking_updated only.
- compute_all_notification_targets : contrat unifie multi-acteurs.
"""

from __future__ import annotations

import pytest

from services.notifications.notification_targets import (
    BookingNotificationContext,
    FullNotificationTargets,
    NotificationTargets,
    compute_all_notification_targets,
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


# ---------------------------------------------------------------------------
# Tests table-driven : compute_all_notification_targets (contrat unifie)
# ---------------------------------------------------------------------------

def _ctx(
    inst: bool = False,
    sub: bool = False,
    driver: bool = True,
) -> BookingNotificationContext:
    return BookingNotificationContext(
        booking_id=1,
        owner_company_id=10,
        executing_company_id=20 if sub else None,
        driver_id=100 if driver else None,
        institution_id=5 if inst else None,
        request_id=50 if inst else None,
        request_public_id="abc-123" if inst else None,
        is_institution_sourced=inst,
        is_subcontracted=sub,
    )


_MATRIX = [
    # ---- booking_assigned ----
    {
        "id": "assigned_simple",
        "event": "booking_assigned", "inst": False, "sub": False, "actor": None, "status": None,
        "expect": {
            "notify_driver_push": True, "notify_driver_socket": True,
            "notify_owner_socket": True, "notify_owner_push": False,
            "notify_institution_persist": False, "notify_executing_socket": False,
        },
    },
    {
        "id": "assigned_institution",
        "event": "booking_assigned", "inst": True, "sub": False, "actor": None, "status": None,
        "expect": {
            "notify_driver_push": True, "notify_owner_socket": True,
            "notify_institution_persist": True, "notify_institution_socket": True,
            "notify_executing_socket": False,
        },
    },
    {
        "id": "assigned_subcontracted",
        "event": "booking_assigned", "inst": True, "sub": True, "actor": None, "status": None,
        "expect": {
            "notify_driver_push": True, "notify_owner_socket": True,
            "notify_institution_persist": True,
            "notify_executing_socket": True, "notify_executing_push": False,
        },
    },
    {
        "id": "assigned_no_driver",
        "event": "booking_assigned", "inst": False, "sub": False, "actor": None, "status": None,
        "driver": False,
        "expect": {
            "notify_driver_push": False, "notify_driver_socket": False,
            "notify_owner_socket": True,
        },
    },

    # ---- booking_reassigned ----
    {
        "id": "reassigned_institution_blocked",
        "event": "booking_reassigned", "inst": True, "sub": True, "actor": None, "status": None,
        "expect": {
            "notify_driver_push": True, "notify_driver_socket": True,
            "notify_owner_socket": True, "notify_owner_push": False,
            "notify_institution_persist": False, "notify_institution_socket": False,
            "notify_executing_socket": True, "notify_executing_push": False,
        },
    },
    {
        "id": "reassigned_simple",
        "event": "booking_reassigned", "inst": False, "sub": False, "actor": None, "status": None,
        "expect": {
            "notify_driver_push": True,
            "notify_institution_persist": False, "notify_executing_socket": False,
        },
    },

    # ---- booking_updated ----
    {
        "id": "updated_en_route_driver_actor",
        "event": "booking_updated", "inst": True, "sub": True, "actor": "driver", "actor_id": 100, "status": "en_route",
        "expect": {
            "notify_owner_push": True, "notify_owner_socket": True,
            "notify_driver_push": False, "notify_driver_socket": False,
            "notify_institution_persist": True, "notify_institution_socket": True,
            "notify_executing_socket": True, "notify_executing_push": False,
        },
    },
    {
        "id": "updated_in_progress_driver_actor",
        "event": "booking_updated", "inst": True, "sub": True, "actor": "driver", "actor_id": 100, "status": "in_progress",
        "expect": {
            "notify_owner_push": True,
            "notify_driver_push": False,
            "notify_institution_persist": False, "notify_institution_socket": False,
            "notify_executing_socket": True, "notify_executing_push": False,
        },
    },
    {
        "id": "updated_completed_driver_actor",
        "event": "booking_updated", "inst": True, "sub": True, "actor": "driver", "actor_id": 100, "status": "completed",
        "expect": {
            "notify_owner_push": True,
            "notify_driver_push": False,
            "notify_institution_persist": True,
            "notify_executing_socket": True, "notify_executing_push": True, "notify_executing_persist": True,
        },
    },
    {
        "id": "updated_completed_company_actor",
        "event": "booking_updated", "inst": True, "sub": False, "actor": "company", "status": "completed",
        "expect": {
            "notify_owner_push": False,
            "notify_driver_push": True,
            "notify_institution_persist": True,
        },
    },
    {
        "id": "updated_en_route_no_institution",
        "event": "booking_updated", "inst": False, "sub": False, "actor": "driver", "status": "en_route",
        "expect": {
            "notify_institution_persist": False, "notify_institution_socket": False,
            "notify_executing_socket": False,
        },
    },
    {
        "id": "updated_cancelled_status",
        "event": "booking_updated", "inst": True, "sub": True, "actor": "company", "status": "cancelled",
        "expect": {
            "notify_institution_persist": True,
            "notify_executing_push": True, "notify_executing_persist": True,
            "notify_driver_push": True, "notify_owner_push": False,
        },
    },

    # ---- booking_cancelled by company ----
    {
        "id": "cancelled_by_company_simple",
        "event": "booking_cancelled", "inst": False, "sub": False, "actor": "company", "status": None,
        "expect": {
            "notify_driver_push": True, "notify_driver_socket": True,
            "notify_owner_push": False, "notify_owner_socket": False,
            "notify_institution_persist": False,
            "notify_executing_push": False, "notify_executing_socket": False,
        },
    },
    {
        "id": "cancelled_by_company_full",
        "event": "booking_cancelled", "inst": True, "sub": True, "actor": "company", "status": None,
        "expect": {
            "notify_driver_push": True,
            "notify_owner_push": False, "notify_owner_socket": False,
            "notify_institution_persist": True, "notify_institution_socket": True,
            "notify_executing_push": True, "notify_executing_socket": True, "notify_executing_persist": True,
        },
    },
    {
        "id": "cancelled_by_company_no_driver",
        "event": "booking_cancelled", "inst": True, "sub": False, "actor": "company", "status": None,
        "driver": False,
        "expect": {
            "notify_driver_push": False, "notify_driver_socket": False,
            "notify_institution_persist": True,
        },
    },

    # ---- booking_cancelled by driver ----
    {
        "id": "cancelled_by_driver_full",
        "event": "booking_cancelled", "inst": True, "sub": True, "actor": "driver", "status": None,
        "expect": {
            "notify_owner_push": True, "notify_owner_socket": True,
            "notify_driver_push": False, "notify_driver_socket": False,
            "notify_institution_persist": True, "notify_institution_socket": True,
            "notify_executing_push": True, "notify_executing_persist": True,
        },
    },
    {
        "id": "cancelled_by_driver_simple",
        "event": "booking_cancelled", "inst": False, "sub": False, "actor": "driver", "status": None,
        "expect": {
            "notify_owner_push": True,
            "notify_driver_push": False,
            "notify_institution_persist": False,
            "notify_executing_push": False,
        },
    },

    # ---- booking_cancelled by institution (futur) ----
    {
        "id": "cancelled_by_institution",
        "event": "booking_cancelled", "inst": True, "sub": True, "actor": "institution", "status": None,
        "expect": {
            "notify_owner_push": True, "notify_owner_socket": True,
            "notify_driver_push": True, "notify_driver_socket": True,
            "notify_institution_persist": False, "notify_institution_socket": False,
            "notify_executing_push": True, "notify_executing_persist": True,
        },
    },

    # ---- booking_cancelled by system ----
    {
        "id": "cancelled_by_system",
        "event": "booking_cancelled", "inst": True, "sub": True, "actor": "system", "status": None,
        "expect": {
            "notify_owner_push": True, "notify_owner_socket": True,
            "notify_driver_push": True,
            "notify_institution_persist": True,
            "notify_executing_push": True,
        },
    },

    # ---- actor_role=None fallback → system ----
    {
        "id": "cancelled_actor_none_fallback_system",
        "event": "booking_cancelled", "inst": True, "sub": True, "actor": None, "status": None,
        "expect": {
            "notify_owner_push": True, "notify_owner_socket": True,
            "notify_driver_push": True, "notify_driver_socket": True,
            "notify_institution_persist": True,
            "notify_executing_push": True, "notify_executing_socket": True,
        },
    },
    {
        "id": "updated_actor_none_treats_as_system",
        "event": "booking_updated", "inst": True, "sub": False, "actor": None, "status": "completed",
        "expect": {
            "notify_owner_push": True,
            "notify_driver_push": True,
            "notify_institution_persist": True,
        },
    },
]


@pytest.mark.parametrize("case", _MATRIX, ids=lambda c: c["id"])
def test_notification_matrix(case: dict) -> None:
    """Test table-driven de la matrice de routage unifiee."""
    ctx = _ctx(
        inst=case.get("inst", False),
        sub=case.get("sub", False),
        driver=case.get("driver", True),
    )
    targets = compute_all_notification_targets(
        case["event"],
        ctx,
        actor_role=case.get("actor"),
        actor_id=case.get("actor_id"),
        status=case.get("status"),
    )
    for key, expected_val in case["expect"].items():
        actual = getattr(targets, key)
        assert actual == expected_val, (
            f"[{case['id']}] {key}: expected {expected_val}, got {actual}"
        )
