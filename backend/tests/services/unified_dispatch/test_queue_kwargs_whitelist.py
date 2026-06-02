"""Régression : ne pas transmettre les kwargs non supportés à Celery."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from services.unified_dispatch.core import queue as ud_queue
from services.unified_dispatch.core.queue import (
    ALLOWED_RUN_KWARGS,
    CompanyDispatchState,
)


@contextmanager
def _fake_app_context():
    app = MagicMock()
    app.app_context.return_value.__enter__ = MagicMock(return_value=None)
    app.app_context.return_value.__exit__ = MagicMock(return_value=False)
    yield app


def _make_state(
    company_id: int = 42, params: dict | None = None
) -> CompanyDispatchState:
    st = CompanyDispatchState(company_id=company_id)
    st.params = dict(params or {})
    st.backlog.append("2026-06-02T12:00:00+00:00 booking_assign")
    return st


class TestQueueKwargsWhitelist:
    def test_allowed_kwargs_pass_through(self):
        st = _make_state(
            params={
                "for_date": "2026-06-02",
                "mode": "manual",
                "regular_first": False,
                "allow_emergency": True,
            }
        )
        sent_kwargs: dict = {}

        with (
            _fake_app_context() as app,
            patch.object(st, "app_ref", app),
            patch("services.unified_dispatch.core.queue._APP", app),
            patch("ext.redis_client", None),
            patch("celery_app.celery") as mock_celery,
        ):
            mock_result = MagicMock()
            mock_result.id = "task-1"
            mock_celery.send_task.return_value = mock_result
            mock_celery.conf.broker_url = "redis://redis:6379/0"

            def capture_send(_name, kwargs=None, **_kw):
                sent_kwargs.update(kwargs or {})
                return mock_result

            mock_celery.send_task.side_effect = capture_send

            ud_queue._enqueue_celery_task(st, mode="auto")

        assert sent_kwargs["company_id"] == 42
        assert sent_kwargs["for_date"] == "2026-06-02"
        assert sent_kwargs["mode"] == "manual"
        assert sent_kwargs["regular_first"] is False
        assert sent_kwargs["allow_emergency"] is True
        assert set(sent_kwargs.keys()).issubset(ALLOWED_RUN_KWARGS)

    def test_action_kwarg_filtered_with_warning(self, caplog):
        st = _make_state(params={"action": "assign", "for_date": "2026-06-02"})
        sent_kwargs: dict = {}

        with (
            _fake_app_context() as app,
            patch.object(st, "app_ref", app),
            patch("services.unified_dispatch.core.queue._APP", app),
            patch("ext.redis_client", None),
            patch("celery_app.celery") as mock_celery,
            caplog.at_level("WARNING"),
        ):
            mock_result = MagicMock()
            mock_result.id = "task-2"
            mock_celery.send_task.return_value = mock_result
            mock_celery.conf.broker_url = "redis://redis:6379/0"

            def capture_send(_name, kwargs=None, **_kw):
                sent_kwargs.update(kwargs or {})
                return mock_result

            mock_celery.send_task.side_effect = capture_send

            ud_queue._enqueue_celery_task(st, mode="auto")

        assert "action" not in sent_kwargs
        assert sent_kwargs["company_id"] == 42
        assert sent_kwargs.get("mode") == "auto"
        assert any(
            "Ignoring unsupported run kwargs" in r.message and "action" in r.message
            for r in caplog.records
        )

    def test_company_id_and_mode_forced(self):
        st = _make_state(params={"for_date": "2026-06-02"})
        sent_kwargs: dict = {}

        with (
            _fake_app_context() as app,
            patch.object(st, "app_ref", app),
            patch("services.unified_dispatch.core.queue._APP", app),
            patch("ext.redis_client", None),
            patch("celery_app.celery") as mock_celery,
        ):
            mock_result = MagicMock()
            mock_result.id = "task-3"
            mock_celery.send_task.return_value = mock_result
            mock_celery.conf.broker_url = "redis://redis:6379/0"

            def capture_send(_name, kwargs=None, **_kw):
                sent_kwargs.update(kwargs or {})
                return mock_result

            mock_celery.send_task.side_effect = capture_send

            ud_queue._enqueue_celery_task(st, mode="semi_auto")

        assert sent_kwargs["company_id"] == 42
        assert sent_kwargs["mode"] == "semi_auto"
