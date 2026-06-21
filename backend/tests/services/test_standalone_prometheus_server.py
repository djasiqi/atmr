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
    monkeypatch.setenv("STANDALONE_PROMETHEUS_ENABLED", "false")

    import prometheus_client

    def _fail(*_args, **_kwargs) -> None:
        raise AssertionError("start_http_server ne doit pas être appelé")

    monkeypatch.setattr(prometheus_client, "start_http_server", _fail, raising=True)

    mod.start_standalone_prometheus_server()
    assert mod._started is False
