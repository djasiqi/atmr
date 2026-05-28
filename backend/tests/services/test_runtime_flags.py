from services.infrastructure.runtime_flags import env_truthy, is_skip_socketio, is_ios_startup_fatal_recovery_disabled


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
