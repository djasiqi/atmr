"""V2 LIRIE : cohérence emit_assignment_* avec dispatch_state_patch et hint dashboard."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.realtime.socketio import (
    DEFAULT_NAMESPACE,
    emit_assignment_cancelled,
    emit_assignment_created,
    emit_assignment_updated,
)


@pytest.fixture
def patch_legacy_emits():
    """Évite les émissions Socket réelles pendant les tests."""
    with (
        patch("services.realtime.socketio.emit_company_event") as m_company,
        patch("services.realtime.socketio.emit_driver_event") as m_driver,
    ):
        yield m_company, m_driver


def test_emit_assignment_created_calls_dispatch_state_patch_and_dashboard_hint(
    patch_legacy_emits,
):
    with (
        patch(
            "services.realtime.socketio._emit_dispatch_state_patch_if_enabled"
        ) as m_patch,
        patch(
            "services.realtime.socketio._maybe_emit_dispatch_dashboard_snapshot"
        ) as m_snap,
    ):
        emit_assignment_created(
            company_id=12,
            booking_id=9001,
            driver_id=44,
            assignment_id="asg-1",
        )
    m_patch.assert_called_once_with(
        company_id=12,
        op="assignment_created",
        reservation_id=9001,
        driver_id=44,
        assignment_id="asg-1",
        namespace=DEFAULT_NAMESPACE,
    )
    m_snap.assert_called_once_with(12, namespace=DEFAULT_NAMESPACE)


def test_emit_assignment_updated_calls_dispatch_state_patch_and_dashboard_hint(
    patch_legacy_emits,
):
    fields = {"status": "assigned"}
    with (
        patch(
            "services.realtime.socketio._emit_dispatch_state_patch_if_enabled"
        ) as m_patch,
        patch(
            "services.realtime.socketio._maybe_emit_dispatch_dashboard_snapshot"
        ) as m_snap,
    ):
        emit_assignment_updated(
            company_id=3,
            assignment_id="u1",
            booking_id=100,
            driver_id=5,
            fields=fields,
        )
    m_patch.assert_called_once_with(
        company_id=3,
        op="assignment_updated",
        reservation_id=100,
        driver_id=5,
        assignment_id="u1",
        namespace=DEFAULT_NAMESPACE,
        fields=fields,
    )
    m_snap.assert_called_once_with(3, namespace=DEFAULT_NAMESPACE)


def test_emit_assignment_cancelled_calls_dispatch_state_patch_and_dashboard_hint(
    patch_legacy_emits,
):
    with (
        patch(
            "services.realtime.socketio._emit_dispatch_state_patch_if_enabled"
        ) as m_patch,
        patch(
            "services.realtime.socketio._maybe_emit_dispatch_dashboard_snapshot"
        ) as m_snap,
    ):
        emit_assignment_cancelled(
            company_id=9,
            assignment_id="c1",
            booking_id=200,
            driver_id=6,
        )
    m_patch.assert_called_once_with(
        company_id=9,
        op="assignment_cancelled",
        reservation_id=200,
        driver_id=6,
        assignment_id="c1",
        namespace=DEFAULT_NAMESPACE,
    )
    m_snap.assert_called_once_with(9, namespace=DEFAULT_NAMESPACE)
