"""Couverture de ``services.notifications.system`` (alertes proactives)."""

from __future__ import annotations

import types
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from requests import RequestException
from sqlalchemy.exc import OperationalError

from services.notifications import system as system_mod
from services.notifications.system import (
    Alert,
    AlertingService,
    get_alerting_service,
)


def _alert(**kwargs) -> Alert:
    defaults = {
        "severity": "warning",
        "title": "Test",
        "message": "msg",
        "metric_name": "m",
        "threshold": 1.0,
        "current_value": 2.0,
        "timestamp": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return Alert(**defaults)


class _Col:
    def __ge__(self, _other):
        return self

    def isnot(self, _other):
        return self

    def in_(self, _other):
        return self


def _eta_query(logs):
    q = MagicMock()
    q.filter.return_value.limit.return_value.all.return_value = logs
    return q


def _delay_query(counts: list[int]):
    q = MagicMock()
    q.filter.return_value.count.side_effect = counts
    return q


def _fake_delay_event(counts: list[int] | None = None, query=None):
    return SimpleNamespace(
        detected_at=_Col(),
        severity=_Col(),
        query=query if query is not None else _delay_query(counts or [0]),
    )


@pytest.fixture
def service():
    system_mod._alerting_service_instance = None
    return AlertingService()


def test_alert_to_dict_metadata():
    a = _alert()
    data = a.to_dict()
    assert data["metadata"] == {}
    assert "timestamp" in data
    b = _alert(metadata={"k": 1})
    assert b.to_dict()["metadata"] == {"k": 1}


def test_check_all_alerts_agregue(service, monkeypatch):
    alerts = [_alert(title=f"a{i}") for i in range(5)]
    monkeypatch.setattr(service, "_check_websocket_disconnection_rate", lambda: alerts[0])
    monkeypatch.setattr(service, "_check_eta_accuracy", lambda: alerts[1])
    monkeypatch.setattr(service, "_check_dispatch_delay_rate", lambda: alerts[2])
    monkeypatch.setattr(service, "_check_osrm_health", lambda: alerts[3])
    monkeypatch.setattr(service, "_check_redis_health", lambda: alerts[4])
    assert service.check_all_alerts() == alerts

    monkeypatch.setattr(service, "_check_websocket_disconnection_rate", lambda: None)
    monkeypatch.setattr(service, "_check_eta_accuracy", lambda: None)
    monkeypatch.setattr(service, "_check_dispatch_delay_rate", lambda: None)
    monkeypatch.setattr(service, "_check_osrm_health", lambda: None)
    monkeypatch.setattr(service, "_check_redis_health", lambda: None)
    assert service.check_all_alerts() == []


def test_websocket_taux_et_erreurs(service, monkeypatch):
    monkeypatch.setattr(
        system_mod.ws_metrics,
        "get_stats",
        lambda: {"connections": {"total": 0, "disconnections_total": 1}},
    )
    assert service._check_websocket_disconnection_rate() is None

    monkeypatch.setattr(
        system_mod.ws_metrics,
        "get_stats",
        lambda: {"connections": {"total": 100, "disconnections_total": 11}},
    )
    warning = service._check_websocket_disconnection_rate()
    assert warning is not None
    assert warning.severity == "warning"

    monkeypatch.setattr(
        system_mod.ws_metrics,
        "get_stats",
        lambda: {"connections": {"total": 100, "disconnections_total": 25}},
    )
    critical = service._check_websocket_disconnection_rate()
    assert critical is not None
    assert critical.severity == "critical"

    monkeypatch.setattr(system_mod.ws_metrics, "get_stats", lambda: (_ for _ in ()).throw(KeyError("x")))
    assert service._check_websocket_disconnection_rate() is None

    monkeypatch.setattr(
        system_mod.ws_metrics, "get_stats", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert service._check_websocket_disconnection_rate() is None


def _install_eta_module(monkeypatch, *, logs=None, query_error=None, missing=False):
    import sys

    mod = types.ModuleType("models.eta_accuracy_log")
    if missing:
        monkeypatch.setitem(sys.modules, "models.eta_accuracy_log", mod)
        return

    class _QueryDesc:
        def __get__(self, _obj, _owner):
            if query_error is not None:
                raise query_error
            return _eta_query(logs or [])

    class EtaAccuracyLog:
        created_at = _Col()
        actual_duration_seconds = _Col()
        predicted_eta_seconds = _Col()
        query = _QueryDesc()

    mod.EtaAccuracyLog = EtaAccuracyLog  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "models.eta_accuracy_log", mod)


def test_eta_precision_et_erreurs(service, monkeypatch):
    _install_eta_module(monkeypatch, logs=[])
    assert service._check_eta_accuracy() is None

    _install_eta_module(
        monkeypatch,
        logs=[SimpleNamespace(actual_duration_seconds=0, predicted_eta_seconds=100)],
    )
    assert service._check_eta_accuracy() is None

    _install_eta_module(
        monkeypatch,
        logs=[SimpleNamespace(actual_duration_seconds=100, predicted_eta_seconds=130)],
    )
    warning = service._check_eta_accuracy()
    assert warning is not None
    assert warning.severity == "warning"

    _install_eta_module(
        monkeypatch,
        logs=[SimpleNamespace(actual_duration_seconds=100, predicted_eta_seconds=150)],
    )
    critical = service._check_eta_accuracy()
    assert critical is not None
    assert critical.severity == "critical"

    _install_eta_module(monkeypatch, missing=True)
    assert service._check_eta_accuracy() is None

    _install_eta_module(
        monkeypatch, query_error=OperationalError("stmt", {}, Exception("db"))
    )
    assert service._check_eta_accuracy() is None

    _install_eta_module(monkeypatch, query_error=TypeError("bad"))
    assert service._check_eta_accuracy() is None

    _install_eta_module(monkeypatch, query_error=RuntimeError("x"))
    assert service._check_eta_accuracy() is None


def test_dispatch_taux_absolu_et_erreurs(service, monkeypatch):
    monkeypatch.setattr(system_mod, "DelayEvent", _fake_delay_event([0]))
    assert service._check_dispatch_delay_rate() is None

    monkeypatch.setattr(system_mod, "DelayEvent", _fake_delay_event([10, 2]))
    warning = service._check_dispatch_delay_rate()
    assert warning is not None
    assert warning.severity == "warning"

    monkeypatch.setattr(system_mod, "DelayEvent", _fake_delay_event([10, 5]))
    critical = service._check_dispatch_delay_rate()
    assert critical is not None
    assert critical.severity == "critical"

    monkeypatch.setattr(
        system_mod.AssignmentRepository,
        "count_by_statuses",
        lambda self, statuses: 50,
    )
    monkeypatch.setattr(system_mod, "DelayEvent", _fake_delay_event([10, 1]))
    abs_warning = service._check_dispatch_delay_rate()
    assert abs_warning is not None
    assert abs_warning.severity == "warning"
    assert abs_warning.metadata["recent_assignments"] == 50

    monkeypatch.setattr(
        system_mod.AssignmentRepository,
        "count_by_statuses",
        lambda self, statuses: 20,
    )
    monkeypatch.setattr(system_mod, "DelayEvent", _fake_delay_event([10, 1]))
    abs_critical = service._check_dispatch_delay_rate()
    assert abs_critical is not None
    assert abs_critical.severity == "critical"

    monkeypatch.setattr(
        system_mod.AssignmentRepository,
        "count_by_statuses",
        lambda self, statuses: 0,
    )
    monkeypatch.setattr(system_mod, "DelayEvent", _fake_delay_event([10, 1]))
    assert service._check_dispatch_delay_rate() is None

    q = MagicMock()
    q.filter.side_effect = OperationalError("s", {}, Exception("e"))
    monkeypatch.setattr(system_mod, "DelayEvent", _fake_delay_event(query=q))
    assert service._check_dispatch_delay_rate() is None

    q.filter.side_effect = TypeError("t")
    assert service._check_dispatch_delay_rate() is None

    q.filter.side_effect = RuntimeError("r")
    assert service._check_dispatch_delay_rate() is None


def test_osrm_sante(service, monkeypatch):
    ok = MagicMock()
    ok.raise_for_status.return_value = None
    monkeypatch.setattr(system_mod.requests, "get", lambda *a, **k: ok)
    service._osrm_down_since = datetime.now(UTC) - timedelta(seconds=10)
    assert service._check_osrm_health() is None
    assert service._osrm_down_since is None
    assert service._last_osrm_check is not None

    def _fail(*_a, **_k):
        raise RequestException("down")

    monkeypatch.setattr(system_mod.requests, "get", _fail)
    service._osrm_down_since = None
    assert service._check_osrm_health() is None
    assert service._osrm_down_since is not None

    service._osrm_down_since = datetime.now(UTC) - timedelta(seconds=61)
    warning = service._check_osrm_health()
    assert warning is not None
    assert warning.severity == "warning"

    service._osrm_down_since = datetime.now(UTC) - timedelta(seconds=121)
    critical = service._check_osrm_health()
    assert critical is not None
    assert critical.severity == "critical"

    monkeypatch.setattr(
        system_mod.os, "getenv", lambda *_a, **_k: (_ for _ in ()).throw(RequestException("env"))
    )
    assert service._check_osrm_health() is None

    monkeypatch.setattr(
        system_mod.os, "getenv", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("env"))
    )
    assert service._check_osrm_health() is None


def test_redis_sante(service, monkeypatch):
    monkeypatch.setattr(system_mod, "redis_client", None)
    assert service._check_redis_health() is None

    client = MagicMock()
    client.ping.return_value = True
    monkeypatch.setattr(system_mod, "redis_client", client)
    service._redis_down_since = datetime.now(UTC)
    assert service._check_redis_health() is None
    assert service._redis_down_since is None
    assert service._last_redis_check is not None

    client.ping.side_effect = ConnectionError("down")
    service._redis_down_since = None
    assert service._check_redis_health() is None
    assert service._redis_down_since is not None

    service._redis_down_since = datetime.now(UTC) - timedelta(seconds=31)
    warning = service._check_redis_health()
    assert warning is not None
    assert warning.severity == "warning"

    service._redis_down_since = datetime.now(UTC) - timedelta(seconds=61)
    critical = service._check_redis_health()
    assert critical is not None
    assert critical.severity == "critical"

    class _BoolBoom:
        def __bool__(self):
            raise ConnectionError("bool")

    monkeypatch.setattr(system_mod, "redis_client", _BoolBoom())
    assert service._check_redis_health() is None

    class _BoolUnexpected:
        def __bool__(self):
            raise RuntimeError("bool")

    monkeypatch.setattr(system_mod, "redis_client", _BoolUnexpected())
    assert service._check_redis_health() is None


def test_send_alert_et_check_and_send(service, monkeypatch):
    alert = _alert()
    service.email_webhook_url = None
    assert service.send_alert(alert) is False

    service.email_webhook_url = "http://webhook.test"
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    monkeypatch.setattr(system_mod.requests, "post", lambda *a, **k: resp)
    assert service.send_alert(alert) is True

    monkeypatch.setattr(
        system_mod.requests,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(RequestException("net")),
    )
    assert service.send_alert(alert) is False

    monkeypatch.setattr(
        system_mod.requests,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("json")),
    )
    assert service.send_alert(alert) is False

    monkeypatch.setattr(
        system_mod.requests,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert service.send_alert(alert) is False

    a1, a2 = _alert(title="ok"), _alert(title="ko")
    monkeypatch.setattr(service, "check_all_alerts", lambda: [a1, a2])
    monkeypatch.setattr(service, "send_alert", lambda a: a is a1)
    sent = service.check_and_send_alerts()
    assert sent == [a1]


def test_get_alerting_service_singleton():
    system_mod._alerting_service_instance = None
    first = get_alerting_service()
    second = get_alerting_service()
    assert first is second
    system_mod._alerting_service_instance = None
