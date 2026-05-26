"""Tests fallback notify_driver_new_booking dans handle_booking_assigned."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_handle_booking_assigned_calls_notify_driver_when_push_target():
    from services.events.handlers.booking_handlers import handle_booking_assigned

    booking = MagicMock()
    booking.id = 34071

    targets = MagicMock()
    targets.notify_driver_push = True
    targets.notify_institution_persist = False
    targets.notify_executing_socket = False

    ctx = MagicMock()
    ctx.driver_id = 6855

    with (
        patch("ext.db.session.get", return_value=booking),
        patch(
            "services.notifications.notification_targets.resolve_booking_notification_context",
            return_value=ctx,
        ),
        patch(
            "services.notifications.notification_targets.compute_all_notification_targets",
            return_value=targets,
        ),
        patch("services.notifications.core.notify_booking_assigned"),
        patch("services.notifications.core.notify_driver_new_booking") as mock_notify_driver,
    ):
        handle_booking_assigned(
            {
                "booking_id": 34071,
                "event_id": "evt-test",
                "correlation_id": "corr-test",
            }
        )

    mock_notify_driver.assert_called_once()
    args, kwargs = mock_notify_driver.call_args
    assert args[0] == 6855
    assert args[1] is booking
    assert kwargs.get("event_id") == "evt-test"
