"""Tests tâches Celery maintenance push tokens."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tasks.notification_tasks import (
    deactivate_stale_device_tokens_task,
    refresh_push_coverage_gauges_task,
)


def test_deactivate_stale_device_tokens_task_commits_and_refreshes() -> None:
    mock_app = MagicMock()
    mock_ctx = MagicMock()
    mock_app.app_context.return_value = mock_ctx
    mock_ctx.__enter__ = MagicMock(return_value=None)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch("celery_app.get_flask_app", return_value=mock_app),
        patch(
            "services.notifications.device_token_lifecycle.deactivate_stale_device_tokens",
            return_value=2,
        ) as deactivate,
        patch("ext.db.session.commit"),
        patch(
            "services.monitoring.prometheus.refresh_push_active_owners_gauges"
        ) as refresh,
    ):
        result = deactivate_stale_device_tokens_task()

    assert result == {"deactivated": 2}
    deactivate.assert_called_once()
    refresh.assert_called_once()


def test_refresh_push_coverage_gauges_task() -> None:
    mock_app = MagicMock()
    mock_ctx = MagicMock()
    mock_app.app_context.return_value = mock_ctx
    mock_ctx.__enter__ = MagicMock(return_value=None)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch("celery_app.get_flask_app", return_value=mock_app),
        patch(
            "services.monitoring.prometheus.refresh_push_active_owners_gauges"
        ) as refresh,
    ):
        result = refresh_push_coverage_gauges_task()

    assert result == {"status": "ok"}
    refresh.assert_called_once()
