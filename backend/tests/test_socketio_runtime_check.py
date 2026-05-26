"""Tests diagnostics Socket.IO / Redis multi-workers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.infrastructure.socketio_runtime_check import (
    collect_socketio_runtime_diagnostics,
)


def test_single_worker_safe_without_message_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUNICORN_WORKERS", "1")
    diag = collect_socketio_runtime_diagnostics(message_queue=None, redis_url="")
    assert diag.multi_worker_safe is True
    assert diag.warnings == ()


def test_multi_worker_without_queue_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUNICORN_WORKERS", "4")
    diag = collect_socketio_runtime_diagnostics(
        message_queue=None,
        redis_url="redis://redis:6379/0",
    )
    assert diag.multi_worker_safe is False
    assert any("Invalid session" in w for w in diag.warnings)


def test_multi_worker_with_queue_and_redis_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUNICORN_WORKERS", "2")
    with patch(
        "services.infrastructure.socketio_runtime_check._ping_redis",
        return_value=(True, None),
    ):
        diag = collect_socketio_runtime_diagnostics(
            message_queue="redis://redis:6379/0",
            redis_url="redis://redis:6379/0",
        )
    assert diag.message_queue_enabled is True
    assert diag.multi_worker_safe is True
    assert diag.warnings == ()
