"""Couverture critique ``services.unified_dispatch.core.queue`` (seuil 95 %)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from models import DispatchStatus, DispatchTriggerOrigin
from services.unified_dispatch.core import queue as ud_queue


@pytest.fixture(autouse=True)
def _reset_queue_globals():
    ud_queue._STOP_EVENT.clear()
    ud_queue._STATE.clear()
    ud_queue._LAST_RESULT.clear()
    ud_queue._LAST_ERROR.clear()
    ud_queue._RUNNING.clear()
    ud_queue._PROGRESS.clear()
    ud_queue._CELERY_STATE.clear()
    yield
    ud_queue._STOP_EVENT.clear()
    ud_queue._STATE.clear()


class _FakeTimer:
    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def test_serialize_datetimes_et_init_app():
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    payload = {
        "when": now,
        "day": date(2026, 8, 13),
        "items": [now, "ok"],
        "n": 3,
    }
    out = ud_queue._serialize_datetimes(payload)
    assert out["when"] == now.isoformat()
    assert out["day"] == "2026-08-13"
    assert out["items"][0] == now.isoformat()
    assert out["n"] == 3
    assert ud_queue._serialize_datetimes("x") == "x"

    app = object()
    ud_queue.init_app(app)
    assert ud_queue._APP is app
    ud_queue._APP = None


def test_get_state_cree_et_reutilise():
    a = ud_queue._get_state(101)
    b = ud_queue._get_state(101)
    assert a is b
    assert a.company_id == 101


def test_get_redis_for_status_branches(monkeypatch):
    ping_ok = MagicMock()
    ping_ok.ping.return_value = True
    monkeypatch.setattr("ext.redis_client", ping_ok)
    assert ud_queue._get_redis_for_status() is ping_ok

    ping_ok.ping.side_effect = RuntimeError("down")
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert ud_queue._get_redis_for_status() is None

    fallback = MagicMock()
    fallback.ping.return_value = True
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr("redis.from_url", lambda *a, **k: fallback)
    assert ud_queue._get_redis_for_status() is fallback

    monkeypatch.setattr("redis.from_url", MagicMock(side_effect=RuntimeError("no")))
    assert ud_queue._get_redis_for_status() is None


def test_get_status_cache_redis_et_local(monkeypatch):
    cid = 201
    fake = MagicMock()
    fake.get.return_value = (
        b'{"last_result": {"bookings": [1], "drivers": [], "assignments": []},'
        b' "last_error": null}'
    )
    monkeypatch.setattr(ud_queue, "_get_redis_for_status", lambda: fake)
    status = ud_queue.get_status(cid)
    assert status["reason"] == "no_drivers"
    assert status["counters"]["bookings"] == 1

    fake.get.return_value = '{"last_result": {"bookings": [], "drivers": [], "assignments": []}, "last_error": "x"}'
    status = ud_queue.get_status(cid)
    assert status["reason"] == "no_bookings_for_day"

    fake.get.return_value = b"not-json"
    ud_queue._LAST_RESULT[cid] = {
        "bookings": [1],
        "drivers": [1],
        "assignments": [],
        "meta": {"dispatch_run_id": 9},
    }
    ud_queue._LAST_ERROR[cid] = "apply boom"
    status = ud_queue.get_status(cid)
    assert status["reason"] == "apply_failed"
    assert status["dispatch_run_id"] == 9

    fake.get.return_value = 12345
    ud_queue._LAST_ERROR[cid] = None
    ud_queue._LAST_RESULT[cid] = {
        "bookings": [1],
        "drivers": [1],
        "assignments": [],
    }
    status = ud_queue.get_status(cid)
    assert status["reason"] == "unknown"

    fake.get.side_effect = RuntimeError("cache fail")
    ud_queue._LAST_RESULT[cid] = {
        "bookings": [1],
        "drivers": [1],
        "assignments": [1],
    }
    status = ud_queue.get_status(cid)
    assert status["reason"] is None


def test_get_status_dispatch_run_et_celery(monkeypatch):
    cid = 202
    run = SimpleNamespace(
        id=44,
        status=SimpleNamespace(value="RUNNING"),
        assignments=[1, 2],
        day=date(2026, 8, 13),
        created_at=datetime(2026, 8, 13, 8, 0, tzinfo=UTC),
        started_at=None,
        completed_at=None,
    )
    repo = MagicMock()
    repo.find_by_company_and_day.return_value = SimpleNamespace(id=44)
    monkeypatch.setattr(ud_queue, "DispatchRunRepository", lambda: repo)
    fake_dr = MagicMock()
    fake_dr.query.get.return_value = run
    monkeypatch.setattr(ud_queue, "DispatchRun", fake_dr)
    monkeypatch.setattr(ud_queue, "_get_redis_for_status", lambda: None)

    status = ud_queue.get_status(cid, for_date="2026-08-13")
    assert status["active_dispatch_run"]["id"] == 44
    assert status["active_dispatch_run"]["assignments_count"] == 2
    assert status["dispatch_run_id"] == 44

    run_plain = SimpleNamespace(
        id=45,
        status="PENDING",
        day=None,
        created_at=None,
        started_at=None,
        completed_at=None,
    )
    fake_dr.query.get.return_value = run_plain
    status = ud_queue.get_status(cid, for_date="2026-08-13")
    assert status["active_dispatch_run"]["status"] == "PENDING"
    assert status["active_dispatch_run"]["assignments_count"] == 0

    repo.find_by_company_and_day.side_effect = RuntimeError("db")
    status = ud_queue.get_status(cid, for_date="2026-08-13")
    assert status["active_dispatch_run"] is None

    bad = ud_queue.get_status(cid, for_date="pas-une-date")
    assert bad["active_dispatch_run"] is None

    st = ud_queue._get_state(cid)
    st.last_task_id = "task-1"
    task = MagicMock()
    task.state = "PENDING"
    task.failed.return_value = False
    task.ready.return_value = False
    monkeypatch.setattr(ud_queue, "AsyncResult", lambda *a, **k: task)
    monkeypatch.setattr("celery_app.celery", MagicMock())
    pending = ud_queue.get_status(cid)
    assert pending["is_running"] is True

    task.state = "FAILURE"
    task.failed.return_value = True
    task.result = "celery boom"
    failed = ud_queue.get_status(cid)
    assert failed["last_error"] == "celery boom"

    task.state = "SUCCESS"
    task.failed.return_value = False
    task.ready.return_value = True
    task.get.return_value = {
        "bookings": [1, 2],
        "drivers": [1],
        "assignments": [1],
    }
    redis = MagicMock()
    monkeypatch.setattr(ud_queue, "_get_redis_for_status", lambda: redis)
    ok = ud_queue.get_status(cid)
    assert ok["counters"]["assignments"] == 1
    redis.setex.assert_called()

    redis.setex.side_effect = RuntimeError("setex")
    ud_queue.get_status(cid)

    task.get.side_effect = RuntimeError("get fail")
    ud_queue.get_status(cid)

    monkeypatch.setattr(
        ud_queue, "AsyncResult", MagicMock(side_effect=RuntimeError("celery down"))
    )
    ud_queue.get_status(cid)


def test_trigger_job_creation_et_overrides(monkeypatch):
    cid = 301
    monkeypatch.setattr(ud_queue, "trigger", lambda *a, **k: None)
    monkeypatch.setattr(ud_queue, "_get_redis_for_status", lambda: None)
    fake_db = MagicMock()
    monkeypatch.setattr(ud_queue, "db", fake_db)

    existing = MagicMock()
    existing.id = 70
    existing.day = date(2026, 8, 13)
    repo = MagicMock()
    repo.find_by_company_and_day.return_value = SimpleNamespace(id=70)
    monkeypatch.setattr(ud_queue, "DispatchRunRepository", lambda: repo)
    fake_dr = MagicMock()
    fake_dr.query.get.return_value = existing
    monkeypatch.setattr(ud_queue, "DispatchRun", fake_dr)

    reused = ud_queue.trigger_job(
        cid,
        {
            "for_date": "2026-08-13",
            "mode": "Auto",
            "overrides": {"a": 1},
            "dispatch_overrides": {"b": 2},
        },
    )
    assert reused["status"] == "queued"
    assert reused["dispatch_run_id"] == 70
    assert existing.status == DispatchStatus.PENDING

    existing.day = date(2026, 1, 1)
    created_run = MagicMock()
    created_run.id = 71
    fake_dr.return_value = created_run
    created = ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert created["dispatch_run_id"] == 71

    invalid = ud_queue.trigger_job(cid, {"for_date": "nope"})
    assert invalid["dispatch_run_id"] is None

    repo.find_by_company_and_day.return_value = None
    fake_dr.query.get.return_value = None
    today = ud_queue.trigger_job(cid, {"mode": "auto"})
    assert today["status"] == "queued"


def test_trigger_job_integrity_et_exceptions(monkeypatch):
    cid = 302
    monkeypatch.setattr(ud_queue, "trigger", lambda *a, **k: None)
    fake_db = MagicMock()
    orig = SimpleNamespace(pgcode="23505")
    fake_db.session.commit.side_effect = IntegrityError("stmt", {}, orig)
    monkeypatch.setattr(ud_queue, "db", fake_db)
    monkeypatch.setattr(
        "services.unified_dispatch.metrics.errors.track_integrity_error",
        lambda **k: None,
    )

    repo = MagicMock()
    monkeypatch.setattr(ud_queue, "DispatchRunRepository", lambda: repo)
    created_run = MagicMock()
    created_run.id = 80
    fake_dr = MagicMock(return_value=created_run)
    raced = MagicMock()
    raced.id = 81
    raced.day = date(2026, 8, 13)
    fake_dr.query.get.return_value = raced
    monkeypatch.setattr(ud_queue, "DispatchRun", fake_dr)

    def _find(_cid, _day):
        if repo.find_by_company_and_day.call_count > 1:
            return SimpleNamespace(id=81)
        return None

    repo.find_by_company_and_day.side_effect = _find
    out = ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert out["dispatch_run_id"] == 81

    mismatched = MagicMock()
    mismatched.id = 82
    mismatched.day = date(2026, 1, 1)
    fake_dr.query.get.return_value = mismatched
    repo.find_by_company_and_day.side_effect = None
    repo.find_by_company_and_day.return_value = SimpleNamespace(id=82)
    mismatch = ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert mismatch["dispatch_run_id"] == 80

    repo.find_by_company_and_day.return_value = None
    fake_dr.query.get.return_value = None
    missing = ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert missing["status"] == "queued"

    fake_db.session.commit.side_effect = RuntimeError("commit fail")
    generic = ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert generic["status"] == "queued"

    monkeypatch.setattr(
        ud_queue,
        "DispatchRunRepository",
        MagicMock(side_effect=RuntimeError("outer")),
    )
    outer = ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert outer["dispatch_run_id"] is None


def test_trigger_job_company_overrides(monkeypatch):
    cid = 303
    fake_db = MagicMock()
    monkeypatch.setattr(ud_queue, "db", fake_db)
    monkeypatch.setattr(
        ud_queue, "DispatchRunRepository", MagicMock(side_effect=RuntimeError("skip"))
    )

    captured = {}

    def _trig(_cid, **kwargs):
        captured.update(kwargs.get("params") or {})

    monkeypatch.setattr(ud_queue, "trigger", _trig)

    remapped = ud_queue.trigger_job(
        cid, {"for_date": "2026-08-13", "dispatch_overrides": {"z": 1}}
    )
    assert remapped["status"] == "queued"
    assert captured.get("overrides") == {"z": 1}

    company = MagicMock()
    company.id = cid
    company.get_autonomous_config.return_value = {
        "dispatch_overrides": {"from_company": True}
    }
    company_repo = MagicMock()
    company_repo.find_by_id.return_value = SimpleNamespace(id=cid)
    monkeypatch.setattr(ud_queue, "CompanyRepository", lambda: company_repo)
    fake_company = MagicMock()
    fake_company.query.get.return_value = company
    monkeypatch.setattr(ud_queue, "Company", fake_company)
    captured.clear()
    ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert captured.get("overrides") == {"from_company": True}

    company_repo.find_by_id.side_effect = RuntimeError("company down")
    captured.clear()
    ud_queue.trigger_job(cid, {"for_date": "2026-08-13"})
    assert "overrides" not in captured


def test_trigger_job_exception_externe(monkeypatch):
    monkeypatch.setattr(ud_queue, "trigger", lambda *a, **k: None)
    monkeypatch.setattr(
        ud_queue,
        "date",
        SimpleNamespace(
            fromisoformat=MagicMock(side_effect=RuntimeError("parse")),
        ),
    )
    out = ud_queue.trigger_job(304, {"for_date": "2026-08-13"})
    assert out["status"] == "queued"
    assert out["dispatch_run_id"] is None


def test_automation_blocked_et_trigger(monkeypatch):
    cid = 401
    monkeypatch.setattr(
        "services.unified_dispatch.utils.autonomous.get_manager_for_company",
        MagicMock(side_effect=RuntimeError("no mgr")),
    )
    assert (
        ud_queue._automation_blocked(cid, DispatchTriggerOrigin.BOOKING_CHANGE) is False
    )

    manager = MagicMock()
    manager.is_automation_allowed.return_value = True
    manager.mode.value = "fully_auto"
    monkeypatch.setattr(
        "services.unified_dispatch.utils.autonomous.get_manager_for_company",
        lambda _cid: manager,
    )
    assert (
        ud_queue._automation_blocked(cid, DispatchTriggerOrigin.BOOKING_CHANGE) is False
    )

    manager.is_automation_allowed.return_value = False
    manager.mode.value = "manual"
    monkeypatch.setattr(
        "services.unified_dispatch.metrics.prometheus.record_auto_trigger_blocked",
        lambda **k: None,
    )
    assert (
        ud_queue._automation_blocked(cid, DispatchTriggerOrigin.BOOKING_CHANGE) is True
    )

    monkeypatch.setattr(
        "services.unified_dispatch.metrics.prometheus.record_auto_trigger_blocked",
        MagicMock(side_effect=RuntimeError("metrics")),
    )
    assert (
        ud_queue._automation_blocked(cid, DispatchTriggerOrigin.BOOKING_CHANGE) is True
    )

    scheduled = []
    monkeypatch.setattr(
        ud_queue, "_schedule_run", lambda st, mode: scheduled.append(mode)
    )
    monkeypatch.setattr(
        ud_queue, "current_app", SimpleNamespace(_get_current_object=lambda: "app")
    )
    st = ud_queue._get_state(cid)
    st.backlog = [f"r{i}" for i in range(ud_queue.MAX_BACKLOG)]
    ud_queue.trigger(cid, reason="storm", params={"for_date": "2026-08-13"})
    assert st.backlog[-1].endswith("(saturated)")
    assert st.params["origin"] == DispatchTriggerOrigin.MANUAL.value
    assert st.app_ref == "app"

    monkeypatch.setattr(ud_queue, "current_app", SimpleNamespace())
    ud_queue.trigger(cid, reason="no-get-obj")
    assert st.app_ref is not None

    class _Boom:
        def _get_current_object(self):
            raise RuntimeError("no ctx")

    monkeypatch.setattr(ud_queue, "current_app", _Boom())
    ud_queue.trigger(cid, reason="boom-ctx")


def test_stop_all_et_schedule(monkeypatch):
    monkeypatch.setattr("threading.Timer", _FakeTimer)
    st = ud_queue._get_state(501)
    boom = MagicMock()
    boom.cancel.side_effect = RuntimeError("cancel")
    st.timer = boom
    other = ud_queue._get_state(502)
    other.timer = None
    ud_queue.stop_all()
    assert ud_queue._STOP_EVENT.is_set()
    assert st.timer is None

    ud_queue._STOP_EVENT.clear()
    ud_queue._schedule_run(st, "auto")
    assert isinstance(st.timer, _FakeTimer)
    assert st.timer.started is True
    first = st.timer
    ud_queue._schedule_run(st, "auto")
    assert first.cancelled is True


def test_try_run_branches(monkeypatch):
    monkeypatch.setattr("threading.Timer", _FakeTimer)
    st = ud_queue._get_state(601)
    ud_queue._STOP_EVENT.set()
    ud_queue._try_run(st, "auto")
    ud_queue._STOP_EVENT.clear()

    enqueued = []
    monkeypatch.setattr(
        ud_queue, "_enqueue_celery_task", lambda s, mode: enqueued.append(mode)
    )
    monkeypatch.setattr(ud_queue, "_get_redis_for_status", lambda: None)

    st.running = True
    st.last_start = datetime.now(UTC) - timedelta(seconds=ud_queue.LOCK_TTL_SEC + 5)
    ud_queue._try_run(st, "auto")
    assert enqueued == ["auto"]

    st.lock = MagicMock()
    st.lock.acquire.return_value = False
    st.running = False
    scheduled = []
    monkeypatch.setattr(
        ud_queue, "_schedule_run", lambda s, mode: scheduled.append("resched")
    )
    ud_queue._try_run(st, "auto")
    assert scheduled == ["resched"]

    st.lock.acquire.return_value = True
    st.running = True
    scheduled.clear()
    ud_queue._try_run(st, "auto")
    assert scheduled == ["resched"]
    st.lock.release.assert_called()

    st.running = False
    redis = MagicMock()
    monkeypatch.setattr(ud_queue, "_get_redis_for_status", lambda: redis)
    monkeypatch.setattr(
        "services.infrastructure.cache.invalidate_dispatch_status_cache",
        lambda *a, **k: None,
    )
    enqueued.clear()
    monkeypatch.setattr(ud_queue, "_schedule_run", lambda s, mode: None)
    ud_queue._try_run(st, "semi")
    assert enqueued == ["semi"]

    monkeypatch.setattr(
        "services.infrastructure.cache.invalidate_dispatch_status_cache",
        MagicMock(side_effect=RuntimeError("inv")),
    )
    st.running = False
    ud_queue._try_run(st, "auto")


def test_enqueue_celery_sans_app_dedup_et_erreur(monkeypatch):
    st = ud_queue.CompanyDispatchState(company_id=701)
    st.backlog.append("r")
    st.app_ref = None
    ud_queue._APP = None
    ud_queue._enqueue_celery_task(st, "auto")
    assert st.running is False
    assert st.backlog == []

    app = MagicMock()
    app.app_context.return_value.__enter__ = MagicMock(return_value=None)
    app.app_context.return_value.__exit__ = MagicMock(return_value=False)
    st.app_ref = app
    st.params = {"for_date": "2026-08-13"}
    redis = MagicMock()
    redis.setnx.return_value = False
    monkeypatch.setattr("ext.redis_client", redis)
    ud_queue._enqueue_celery_task(st, "auto")
    assert st.running is False

    redis.setnx.return_value = True
    task = MagicMock(id=str(uuid4()), state="PENDING")
    monkeypatch.setattr(
        "tasks.dispatch_tasks.run_dispatch_task.apply_async",
        lambda **k: task,
    )
    ud_queue._enqueue_celery_task(st, "auto")
    assert st.last_task_id == task.id
    redis.expire.assert_called()

    monkeypatch.setattr(
        "tasks.dispatch_tasks.run_dispatch_task.apply_async",
        MagicMock(side_effect=RuntimeError("broker")),
    )
    ud_queue._enqueue_celery_task(st, "auto")
    assert ud_queue._LAST_ERROR[st.company_id] == "broker"


def test_trigger_on_booking_change_delegue(monkeypatch):
    called = {}

    def _trig(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(ud_queue, "trigger", _trig)
    ud_queue.trigger_on_booking_change(801, params={"for_date": "2026-08-13"})
    assert called["company_id"] == 801
    assert called["origin"] == DispatchTriggerOrigin.BOOKING_CHANGE
