"""Tests du kill-switch startup iOS et exposition bootstrap/version-check."""

from __future__ import annotations


def test_ios_startup_fatal_recovery_disabled_flag(monkeypatch):
    from services.infrastructure.runtime_flags import (
        IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV,
        get_mobile_startup_runtime_flags,
        is_ios_startup_fatal_recovery_disabled,
    )

    monkeypatch.delenv(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV, raising=False)
    assert is_ios_startup_fatal_recovery_disabled() is False
    assert get_mobile_startup_runtime_flags() == {
        "ios_startup_fatal_recovery_disabled": False,
    }

    monkeypatch.setenv(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV, "true")
    assert is_ios_startup_fatal_recovery_disabled() is True
    assert get_mobile_startup_runtime_flags() == {
        "ios_startup_fatal_recovery_disabled": True,
    }


def test_get_runtime_flags_status_includes_mobile_startup(monkeypatch):
    from services.infrastructure.runtime_flags import (
        IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV,
        get_runtime_flags_status,
    )

    monkeypatch.setenv(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV, "true")
    status = get_runtime_flags_status()
    assert status["mobile_startup"]["ios_startup_fatal_recovery_disabled"] is True
    assert "ios_startup_fatal_recovery_disabled" in status["notes"]


def test_version_check_includes_startup_runtime(client, monkeypatch):
    from services.infrastructure.runtime_flags import IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV

    monkeypatch.setenv(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV, "true")
    response = client.post(
        "/api/v1/app/version-check",
        json={"platform": "ios", "current_version": "1.0.5"},
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["startup_runtime"]["ios_startup_fatal_recovery_disabled"] is True


def test_feature_flags_runtime_status_endpoint(client, monkeypatch):
    from services.infrastructure.runtime_flags import IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV

    monkeypatch.setenv(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV, "1")
    response = client.get("/api/feature-flags/runtime-status")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mobile_startup"]["ios_startup_fatal_recovery_disabled"] is True


def test_bootstrap_feature_flags_include_ios_startup_kill_switch(client, monkeypatch):
    from services.infrastructure.runtime_flags import IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV

    monkeypatch.setenv(IOS_STARTUP_FATAL_RECOVERY_DISABLED_ENV, "yes")
    response = client.get("/api/v1/auth/bootstrap")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["feature_flags"]["ios_startup_fatal_recovery_disabled"] is True
