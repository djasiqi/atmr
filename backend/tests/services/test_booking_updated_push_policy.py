"""Tests politique booking_updated push conditionnel (handler)."""

from __future__ import annotations

from services.events.handlers.booking_push_policy import (
    DRIVER_PUSH_FIELDS,
    should_send_driver_push_on_booking_updated,
    status_change_triggers_driver_push,
)


def test_driver_push_only_on_significant_changes() -> None:
    changes = {"status": "EN_ROUTE", "internal_ref": "x"}
    assert status_change_triggers_driver_push(changes) is False
    assert (
        should_send_driver_push_on_booking_updated(
            notify_driver_push=True,
            changes=changes,
        )
        is False
    )


def test_driver_push_true_on_scheduled_time_change() -> None:
    changes = {"scheduled_time": "2026-06-17T10:00:00Z"}
    assert (
        should_send_driver_push_on_booking_updated(
            notify_driver_push=True,
            changes=changes,
        )
        is True
    )


def test_driver_push_true_on_canceled_status() -> None:
    changes = {"status": "CANCELED"}
    assert status_change_triggers_driver_push(changes) is True
    assert (
        should_send_driver_push_on_booking_updated(
            notify_driver_push=True,
            changes=changes,
        )
        is True
    )


def test_driver_push_true_on_assigned_status() -> None:
    changes = {"status": "ASSIGNED"}
    assert status_change_triggers_driver_push(changes) is True


def test_driver_push_false_on_return_completed_status() -> None:
    changes = {"status": "RETURN_COMPLETED"}
    assert status_change_triggers_driver_push(changes) is False


def test_driver_push_status_nested_change_dict() -> None:
    changes = {"status": {"old": "PENDING", "new": "CANCELED"}}
    assert status_change_triggers_driver_push(changes) is True


def test_driver_push_false_when_notify_disabled() -> None:
    changes = {"scheduled_time": "2026-06-17T10:00:00Z"}
    assert (
        should_send_driver_push_on_booking_updated(
            notify_driver_push=False,
            changes=changes,
        )
        is False
    )


def test_driver_push_fields_contains_expected_keys() -> None:
    assert "scheduled_time" in DRIVER_PUSH_FIELDS
    assert "pickup_location" in DRIVER_PUSH_FIELDS
