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


def _patch_run_dispatch_apply_async(capture: dict):
    """Helper: patch apply_async and capture kwargs."""

    def _apply_async(*, kwargs=None, **_kw):
        capture.update(kwargs or {})
        mock_result = MagicMock()
        mock_result.id = "task-test"
        mock_result.state = "PENDING"
        return mock_result

    return patch(
        "tasks.dispatch_tasks.run_dispatch_task.apply_async",
        side_effect=_apply_async,
    )


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
            _patch_run_dispatch_apply_async(sent_kwargs),
        ):
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
            _patch_run_dispatch_apply_async(sent_kwargs),
            caplog.at_level("WARNING"),
        ):
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
            _patch_run_dispatch_apply_async(sent_kwargs),
        ):
            ud_queue._enqueue_celery_task(st, mode="semi_auto")

        assert sent_kwargs["company_id"] == 42
        assert sent_kwargs["mode"] == "semi_auto"

    def test_for_date_defaulted_when_missing(self):
        st = _make_state(params={})
        sent_kwargs: dict = {}

        with (
            _fake_app_context() as app,
            patch.object(st, "app_ref", app),
            patch("services.unified_dispatch.core.queue._APP", app),
            patch("ext.redis_client", None),
            _patch_run_dispatch_apply_async(sent_kwargs),
        ):
            ud_queue._enqueue_celery_task(st, mode="auto")

        assert sent_kwargs["company_id"] == 42
        assert sent_kwargs.get("for_date")
