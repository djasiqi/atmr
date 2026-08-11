"""Tests export Prometheus multiprocess / process-local."""

from __future__ import annotations


def test_generate_prometheus_latest_without_multiproc_dir(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    from services.monitoring.prometheus_export import generate_prometheus_latest

    payload, content_type = generate_prometheus_latest()
    assert isinstance(payload, (bytes, bytearray))
    assert "text/plain" in content_type


def test_generate_prometheus_latest_with_multiproc_dir(monkeypatch, tmp_path):
    multiproc = tmp_path / "prom"
    multiproc.mkdir()
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(multiproc))
    from services.monitoring.prometheus_export import generate_prometheus_latest

    payload, content_type = generate_prometheus_latest()
    assert isinstance(payload, (bytes, bytearray))
    assert "text/plain" in content_type
    # Dir toujours présent (pas de crash MultiProcessCollector sur dir vide)
    assert multiproc.is_dir()
