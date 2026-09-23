"""Tests diagnostics Socket.IO : file Redis distincte du polling Engine.IO."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.infrastructure.socketio_runtime_check import (
    collect_socketio_runtime_diagnostics,
)


def test_single_worker_polling_safe_without_message_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUNICORN_WORKERS", "1")
    monkeypatch.delenv("SOCKETIO_WORKER_AFFINITY_GUARANTEED", raising=False)
    diag = collect_socketio_runtime_diagnostics(message_queue=None, redis_url="")
    assert diag.message_queue_enabled is False
    assert diag.engineio_polling_safe is True
    assert diag.multi_worker_safe is True
    assert diag.worker_affinity_guaranteed is False
    assert diag.warnings == ()


def test_multi_worker_without_queue_is_not_polling_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUNICORN_WORKERS", "4")
    monkeypatch.delenv("SOCKETIO_WORKER_AFFINITY_GUARANTEED", raising=False)
    diag = collect_socketio_runtime_diagnostics(
        message_queue=None,
        redis_url="redis://redis:6379/0",
    )
    assert diag.message_queue_enabled is False
    assert diag.engineio_polling_safe is False
    assert diag.multi_worker_safe is False
    assert any("Invalid session" in w for w in diag.warnings)
    assert any("émissions" in w for w in diag.warnings)


def test_six_workers_redis_ok_queue_on_without_affinity_is_not_polling_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis et la file ne rendent pas le long-polling sûr sous Gunicorn."""
    monkeypatch.setenv("GUNICORN_WORKERS", "6")
    monkeypatch.delenv("SOCKETIO_WORKER_AFFINITY_GUARANTEED", raising=False)
    with patch(
        "services.infrastructure.socketio_runtime_check._ping_redis",
        return_value=(True, None),
    ):
        diag = collect_socketio_runtime_diagnostics(
            message_queue="redis://redis:6379/0",
            redis_url="redis://redis:6379/0",
        )
    assert diag.message_queue_enabled is True
    assert diag.redis_ping_ok is True
    assert diag.worker_affinity_guaranteed is False
    assert diag.engineio_polling_safe is False
    assert diag.multi_worker_safe is False
    assert any("ne partage pas ces sid" in w for w in diag.warnings)
    assert not any(
        "émissions Socket.IO ne sont pas coordonnées" in w for w in diag.warnings
    )


def test_multi_worker_with_explicit_affinity_is_polling_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GUNICORN_WORKERS", "6")
    monkeypatch.setenv("SOCKETIO_WORKER_AFFINITY_GUARANTEED", "true")
    with patch(
        "services.infrastructure.socketio_runtime_check._ping_redis",
        return_value=(True, None),
    ):
        diag = collect_socketio_runtime_diagnostics(
            message_queue="redis://redis:6379/0",
            redis_url="redis://redis:6379/0",
        )
    assert diag.worker_affinity_guaranteed is True
    assert diag.engineio_polling_safe is True
    assert diag.message_queue_enabled is True
    assert diag.warnings == ()
