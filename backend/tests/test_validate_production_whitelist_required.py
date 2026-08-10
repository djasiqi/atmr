"""Validation au démarrage : ADMIN_IP_WHITELIST_REQUIRED en production."""

from __future__ import annotations

import pytest


@pytest.fixture
def prod_env_minimal(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-key-min-32-chars-xx")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SOCKETIO_CORS_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("PDF_BASE_URL", "https://api.example.com")
    monkeypatch.setenv(
        "INTERNAL_SERVICE_TOKEN",
        "test-internal-service-token-with-at-least-32-characters",
    )


def test_production_whitelist_required_raises_when_empty(prod_env_minimal, monkeypatch):
    from app import validate_required_env_vars

    monkeypatch.setenv("ADMIN_IP_WHITELIST_REQUIRED", "true")
    monkeypatch.delenv("ADMIN_IP_WHITELIST", raising=False)
    with pytest.raises(RuntimeError, match="ADMIN_IP_WHITELIST"):
        validate_required_env_vars("production")


def test_production_whitelist_required_ok_with_entries(prod_env_minimal, monkeypatch):
    from app import validate_required_env_vars

    monkeypatch.setenv("ADMIN_IP_WHITELIST_REQUIRED", "true")
    monkeypatch.setenv("ADMIN_IP_WHITELIST", "10.0.0.0/24")
    validate_required_env_vars("production")


def test_production_whitelist_required_false_allows_empty(
    prod_env_minimal, monkeypatch
):
    from app import validate_required_env_vars

    monkeypatch.setenv("ADMIN_IP_WHITELIST_REQUIRED", "false")
    monkeypatch.delenv("ADMIN_IP_WHITELIST", raising=False)
    validate_required_env_vars("production")
