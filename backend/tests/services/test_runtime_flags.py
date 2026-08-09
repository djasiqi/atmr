from services.infrastructure.runtime_flags import (
    env_truthy,
    is_ios_startup_fatal_recovery_disabled,
    is_skip_socketio,
    is_socket_gps_ingest_enabled,
)


def test_env_truthy(monkeypatch):
    monkeypatch.setenv("TEST_FLAG", "1")
    assert env_truthy("TEST_FLAG") is True
    monkeypatch.setenv("TEST_FLAG", "yes")
    assert env_truthy("TEST_FLAG") is True
    monkeypatch.setenv("TEST_FLAG", "false")
    assert env_truthy("TEST_FLAG") is False


def test_skip_socketio(monkeypatch):
    monkeypatch.setenv("SKIP_SOCKETIO", "true")
    assert is_skip_socketio() is True
    monkeypatch.setenv("SKIP_SOCKETIO", "1")
    assert is_skip_socketio() is True


def test_ios_startup_fatal_recovery_disabled(monkeypatch):
    monkeypatch.delenv("IOS_STARTUP_FATAL_RECOVERY_DISABLED", raising=False)
    assert is_ios_startup_fatal_recovery_disabled() is False
    monkeypatch.setenv("IOS_STARTUP_FATAL_RECOVERY_DISABLED", "true")
    assert is_ios_startup_fatal_recovery_disabled() is True


def test_socket_gps_ingest_enabled_default_true(monkeypatch):
    monkeypatch.delenv("SOCKET_GPS_INGEST_ENABLED", raising=False)
    assert is_socket_gps_ingest_enabled() is True


def test_socket_gps_ingest_enabled_kill_switch(monkeypatch):
    monkeypatch.setenv("SOCKET_GPS_INGEST_ENABLED", "false")
    assert is_socket_gps_ingest_enabled() is False
    monkeypatch.setenv("SOCKET_GPS_INGEST_ENABLED", "0")
    assert is_socket_gps_ingest_enabled() is False
