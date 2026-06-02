"""Intégration : trigger_on_booking_change → enqueue Celery sans kwarg parasite."""

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


class TestDispatchEnqueueFromBooking:
    def test_trigger_on_booking_change_records_reason_in_backlog(self):
        company_id = 9001
        ud_queue._STATE.pop(company_id, None)

        with patch.object(ud_queue, "_schedule_run", lambda st, mode: None):
            ud_queue.trigger_on_booking_change(
                company_id, reason="booking_assign", mode="auto"
            )

        st = ud_queue._get_state(company_id)
        assert any("booking_assign" in entry for entry in st.backlog)

    def test_enqueue_from_booking_path_filters_action(self):
        company_id = 9002
        st = CompanyDispatchState(company_id=company_id)
        st.params = {"action": "assign", "for_date": "2026-06-02"}
        st.backlog.append("2026-06-02T12:00:00+00:00 booking_assign")
        sent_kwargs: dict = {}

        with (
            _fake_app_context() as app,
            patch.object(st, "app_ref", app),
            patch("services.unified_dispatch.core.queue._APP", app),
            patch("ext.redis_client", None),
            patch(
                "tasks.dispatch_tasks.run_dispatch_task.apply_async",
                side_effect=lambda *, kwargs=None, **_kw: (
                    sent_kwargs.update(kwargs or {}),
                    MagicMock(id="task-booking", state="PENDING"),
                )[1],
            ),
        ):
            ud_queue._enqueue_celery_task(st, mode="auto")

        assert "action" not in sent_kwargs
        assert sent_kwargs["company_id"] == company_id
        assert "mode" in sent_kwargs
        assert set(sent_kwargs.keys()).issubset(ALLOWED_RUN_KWARGS)
