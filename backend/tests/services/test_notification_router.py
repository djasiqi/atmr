# backend/tests/services/test_notification_router.py
"""Tests unitaires du routage des notifications (NotificationRouter).

Vérifie (type, actor, recipient) -> push yes/no + socket emit.
Vérifie P0: DRIVER_EN_ROUTE / ONBOARD / COMPLETED -> jamais push chauffeur.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from services.notifications.domain_event_canonical import DomainEventCanonical
from services.notifications.router import route, should_skip_push_for_driver
from services.notifications.router_config import (
    DRIVER_PROGRESS_STATUS_VALUES,
    DRIVER_STATUS_PROGRESS_TYPES,
    is_driver_progress_status,
    is_driver_status_progress_type,
)

# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------


def _canonical(
    event_type: str,
    *,
    driver_id: int = 101,
    company_id: int = 42,
    booking_id: int = 1,
    actor_role: str | None = "driver",
    actor_id: int | None = 101,
    title: str = "Titre",
    body: str = "Corps",
) -> DomainEventCanonical:
    return DomainEventCanonical(
        type=event_type,
        booking_id=booking_id,
        company_id=company_id,
        driver_id=driver_id,
        actor_role=actor_role,
        actor_id=actor_id,
        title=title,
        body=body,
        ts=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Router : (type, actor, recipient) -> push yes/no + socket emit
# ---------------------------------------------------------------------------


@patch(
    "services.notifications.router.check_dedup_and_throttle",
    return_value=(False, None),
)
def test_route_driver_en_route_no_push_to_driver(mock_dedup):
    """P0: DRIVER_EN_ROUTE avec acteur=chauffeur -> push uniquement company, jamais driver."""
    ev = _canonical("DRIVER_EN_ROUTE", actor_role="driver", actor_id=101)
    res = route(ev, presence_driver=None, presence_company=None)
    mock_dedup.assert_called()
    push_driver = [r for r in res.push_requests if r.role == "driver"]
    push_company = [r for r in res.push_requests if r.role == "company"]
    assert len(push_driver) == 0, "jamais de push au chauffeur pour DRIVER_EN_ROUTE"
    assert len(push_company) >= 1, "push entreprise attendue"


@patch(
    "services.notifications.router.check_dedup_and_throttle",
    return_value=(False, None),
)
def test_route_driver_completed_no_push_to_driver(mock_dedup):
    """P0: DRIVER_COMPLETED -> push company only, jamais driver."""
    ev = _canonical("DRIVER_COMPLETED", actor_role="driver", actor_id=101)
    res = route(ev)
    push_driver = [r for r in res.push_requests if r.role == "driver"]
    assert len(push_driver) == 0


@patch(
    "services.notifications.router.check_dedup_and_throttle",
    return_value=(False, None),
)
def test_route_booking_assigned_push_to_driver_if_inactive(mock_dedup):
    """BOOKING_ASSIGNED -> push driver si acteur != driver (entreprise), exclude_actor."""
    ev = _canonical(
        "BOOKING_ASSIGNED",
        actor_role="company",
        actor_id=42,
    )
    res = route(ev, presence_driver=False)
    push_driver = [r for r in res.push_requests if r.role == "driver"]
    assert len(push_driver) >= 1


@patch(
    "services.notifications.router.check_dedup_and_throttle",
    return_value=(False, None),
)
def test_route_booking_assigned_exclude_actor_driver_no_push_to_driver(mock_dedup):
    """BOOKING_ASSIGNED avec acteur=driver (cas edge) -> exclude_actor skip push driver."""
    ev = _canonical("BOOKING_ASSIGNED", actor_role="driver", actor_id=101)
    res = route(ev)
    push_driver = [r for r in res.push_requests if r.role == "driver"]
    skip_driver = [s for s in res.skip_reasons if s[0] == "driver"]
    assert len(push_driver) == 0
    assert any(s[2] == "exclude_actor" for s in skip_driver)


@patch(
    "services.notifications.router.check_dedup_and_throttle",
    return_value=(False, None),
)
def test_route_socket_emits_both_rooms_when_both(mock_dedup):
    """Pour recipients=both, socket_emits contient driver et company."""
    ev = _canonical("BOOKING_UPDATED", actor_role="company", actor_id=42)
    res = route(ev)
    roles = {e.role for e in res.socket_emits}
    assert "driver" in roles
    assert "company" in roles


# ---------------------------------------------------------------------------
# should_skip_push_for_driver (helper P0)
# ---------------------------------------------------------------------------


def test_should_skip_push_for_driver_when_driver_progress_status():
    """Si status en_route/in_progress/completed -> skip push driver."""
    assert should_skip_push_for_driver(
        "BOOKING_UPDATED", "en_route", "driver", 101, 101
    )
    assert should_skip_push_for_driver(
        "BOOKING_UPDATED", "in_progress", "company", 42, 101
    )
    assert should_skip_push_for_driver("BOOKING_UPDATED", "completed", None, None, 101)


def test_should_skip_push_for_driver_when_actor_is_driver():
    """Si actor_role=driver et actor_id=driver_id -> skip push driver."""
    assert should_skip_push_for_driver(
        "BOOKING_UPDATED", "assigned", "driver", 101, 101
    )


def test_should_not_skip_push_for_driver_when_company_actor():
    """Si acteur=company, on peut push au driver (booking_updated depuis entreprise)."""
    assert not should_skip_push_for_driver(
        "BOOKING_UPDATED", "assigned", "company", 42, 101
    )


# ---------------------------------------------------------------------------
# router_config helpers
# ---------------------------------------------------------------------------


def test_is_driver_status_progress_type():
    """Types driver progress reconnus."""
    assert is_driver_status_progress_type("DRIVER_EN_ROUTE")
    assert is_driver_status_progress_type("DRIVER_ONBOARD")
    assert is_driver_status_progress_type("DRIVER_COMPLETED")
    assert not is_driver_status_progress_type("BOOKING_UPDATED")


def test_is_driver_progress_status():
    """Statuts booking qui indiquent action chauffeur."""
    assert is_driver_progress_status("en_route")
    assert is_driver_progress_status("in_progress")
    assert is_driver_progress_status("completed")
    assert is_driver_progress_status("return_completed")
    assert not is_driver_progress_status("assigned")
    assert not is_driver_progress_status(None)


def test_driver_progress_constants():
    """Cohérence des constantes."""
    assert "DRIVER_EN_ROUTE" in DRIVER_STATUS_PROGRESS_TYPES
    assert "en_route" in DRIVER_PROGRESS_STATUS_VALUES


# ---------------------------------------------------------------------------
# Dedup / Throttle (Redis mock)
# ---------------------------------------------------------------------------


@patch("services.notifications.dedup_throttle._get_redis")
def test_dedup_skip_when_key_exists(mock_get_redis):
    """Si la clé dedup existe déjà -> should_skip_dedup True."""
    from services.notifications.dedup_throttle import should_skip_dedup

    mock_redis = pytest.importorskip("unittest.mock").MagicMock()
    mock_redis.get.return_value = b"1"
    mock_get_redis.return_value = mock_redis
    assert should_skip_dedup("driver", 101, "booking:1:type:x:v1") is True
    mock_redis.get.assert_called_once()
    mock_redis.setex.assert_not_called()


@patch("services.notifications.dedup_throttle._get_redis")
def test_dedup_no_skip_when_key_absent(mock_get_redis):
    """Si la clé dedup n'existe pas -> should_skip_dedup False, setex appelé."""
    from services.notifications.dedup_throttle import should_skip_dedup

    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_get_redis.return_value = mock_redis
    assert should_skip_dedup("driver", 101, "booking:1:type:x:v1") is False
    mock_redis.setex.assert_called_once()


@patch("services.notifications.dedup_throttle._get_redis")
def test_check_dedup_and_throttle_returns_deduped(mock_get_redis):
    """check_dedup_and_throttle retourne (True, 'deduped') quand dedup hit."""
    from services.notifications.dedup_throttle import check_dedup_and_throttle

    mock_redis = pytest.importorskip("unittest.mock").MagicMock()
    mock_redis.get.return_value = b"1"
    mock_get_redis.return_value = mock_redis
    skip, reason = check_dedup_and_throttle("driver", 101, "dedupe_key", "scope", 60, 1)
    assert skip is True
    assert reason == "deduped"


@patch("services.notifications.dedup_throttle._get_redis")
def test_check_dedup_and_throttle_returns_throttled(mock_get_redis):
    """check_dedup_and_throttle retourne (True, 'throttled') quand throttle dépassé."""
    from services.notifications.dedup_throttle import check_dedup_and_throttle

    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.incr.return_value = 2
    mock_get_redis.return_value = mock_redis
    skip, reason = check_dedup_and_throttle(
        "company", 42, "dedupe_key", "booking_1", 60, 1
    )
    assert skip is True
    assert reason == "throttled"
