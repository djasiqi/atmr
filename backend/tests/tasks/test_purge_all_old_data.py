"""Tests purge_all_old_data : ne pas passer self aux sous-tâches bind=True."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tasks.purge_tasks import purge_all_old_data


@patch("tasks.purge_tasks.get_flask_app")
@patch("tasks.purge_tasks.purge_old_task_failures")
@patch("tasks.purge_tasks.purge_old_autonomous_actions")
@patch("tasks.purge_tasks.purge_old_realtime_events")
@patch("tasks.purge_tasks.purge_old_messages")
@patch("tasks.purge_tasks.purge_old_bookings")
def test_purge_all_old_data_calls_subtasks_without_self(
    mock_bookings,
    mock_messages,
    mock_events,
    mock_actions,
    mock_failures,
    mock_get_flask_app,
):
    mock_app = MagicMock()
    mock_get_flask_app.return_value = mock_app
    mock_app.app_context.return_value.__enter__ = MagicMock(return_value=None)
    mock_app.app_context.return_value.__exit__ = MagicMock(return_value=False)

    for mock_fn in (
        mock_bookings,
        mock_messages,
        mock_events,
        mock_actions,
        mock_failures,
    ):
        mock_fn.return_value = {"status": "success", "deleted_count": 1, "errors": []}

    task_self = MagicMock()
    result = purge_all_old_data.run()

    for mock_fn in (
        mock_bookings,
        mock_messages,
        mock_events,
        mock_actions,
        mock_failures,
    ):
        mock_fn.assert_called_once_with()
        assert mock_fn.call_args.args == ()

    assert result["status"] == "success"
    assert result["summary"]["total_deleted"] == 5
    assert result["summary"]["total_errors"] == 0
