"""Tests serveur Prometheus standalone (consumers Kafka)."""

from __future__ import annotations

import services.monitoring.standalone_prometheus_server as mod


def test_start_standalone_prometheus_server_idempotent(monkeypatch) -> None:
    mod._started = False
    calls: list[tuple[int, str]] = []

    def _fake_start(port: int, addr: str = "") -> None:
        calls.append((port, addr))

    monkeypatch.setenv("STANDALONE_PROMETHEUS_ENABLED", "true")
    monkeypatch.setenv("STANDALONE_PROMETHEUS_PORT", "9199")

    import prometheus_client

    monkeypatch.setattr(
        prometheus_client, "start_http_server", _fake_start, raising=True
    )

    mod.start_standalone_prometheus_server()
    mod.start_standalone_prometheus_server()

    assert len(calls) == 1
    assert calls[0] == (9199, "0.0.0.0")
    assert mod._started is True


def test_start_standalone_prometheus_server_disabled(monkeypatch) -> None:
    mod._started = False
    mod._process_collector_registered = False
    monkeypatch.setenv("STANDALONE_PROMETHEUS_ENABLED", "false")

    import prometheus_client

    def _fail(*_args, **_kwargs) -> None:
        raise AssertionError("start_http_server ne doit pas être appelé")

    monkeypatch.setattr(prometheus_client, "start_http_server", _fail, raising=True)

    mod.start_standalone_prometheus_server()
    assert mod._started is False


def test_start_standalone_prometheus_server_registers_process_collector(
    monkeypatch,
) -> None:
    mod._started = False
    mod._process_collector_registered = False
    registered: list[str] = []

    def _fake_register(collector) -> None:
        registered.append(type(collector).__name__)

    def _fake_start(port: int, addr: str = "") -> None:
        return None

    monkeypatch.setenv("STANDALONE_PROMETHEUS_ENABLED", "true")
    import prometheus_client

    monkeypatch.setattr(
        prometheus_client, "start_http_server", _fake_start, raising=True
    )
    monkeypatch.setattr(
        prometheus_client.REGISTRY, "register", _fake_register, raising=True
    )
    class _FakeProcessCollector:
        pass

    monkeypatch.setattr(
        prometheus_client,
        "ProcessCollector",
        _FakeProcessCollector,
        raising=True,
    )

    mod.start_standalone_prometheus_server()
    assert mod._started is True
    assert mod._process_collector_registered is True
    assert len(registered) == 1
