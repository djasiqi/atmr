from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from models.enums import AssignmentStatus
from services.geolocation.location import (
    TRIP_TRACKING_ASSIGNMENT_STATUSES,
    LocationService,
)


def test_trip_tracking_statuses_include_en_route_phases() -> None:
    """Historique trajet doit couvrir toute la mission, pas seulement ONBOARD."""
    assert AssignmentStatus.EN_ROUTE_DROPOFF in TRIP_TRACKING_ASSIGNMENT_STATUSES
    assert AssignmentStatus.EN_ROUTE_PICKUP in TRIP_TRACKING_ASSIGNMENT_STATUSES
    assert AssignmentStatus.ONBOARD in TRIP_TRACKING_ASSIGNMENT_STATUSES
    assert AssignmentStatus.SCHEDULED not in TRIP_TRACKING_ASSIGNMENT_STATUSES


def test_should_append_trip_history_true_for_older_than_canonical() -> None:
    svc = LocationService(redis_client_instance=None)
    now = datetime.now(UTC)
    assert (
        svc._should_append_trip_history(
            location_mode="mission_live",
            accept_status="accepted_observability_only",
            accept_reason="older_than_canonical",
            recorded_at=now - timedelta(minutes=1),
        )
        is True
    )


def test_should_append_trip_history_false_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import services.geolocation.location as loc

    monkeypatch.setattr(loc, "TRIP_HISTORY_REPLAY_ENABLED", False)
    svc = LocationService(redis_client_instance=None)
    now = datetime.now(UTC)
    assert (
        svc._should_append_trip_history(
            location_mode="mission_live",
            accept_status="accepted_observability_only",
            accept_reason="older_than_canonical",
            recorded_at=now,
        )
        is False
    )


def test_should_append_trip_history_false_for_availability() -> None:
    svc = LocationService(redis_client_instance=None)
    now = datetime.now(UTC)
    assert (
        svc._should_append_trip_history(
            location_mode="availability_presence",
            accept_status="accepted_observability_only",
            accept_reason="older_than_canonical",
            recorded_at=now,
        )
        is False
    )
