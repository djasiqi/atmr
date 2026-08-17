"""Couverture de ``manage`` (CLI migrations / seed, politique eventlet)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from flask import Flask

import manage
from manage import (
    apply_eventlet_monkey_patch,
    bootstrap_eventlet,
    cli,
    env_disables_eventlet,
    get_app,
    is_migration_command,
    resolve_config_name,
    running_under_pytest,
)


@pytest.fixture
def dummy_app(monkeypatch):
    app = Flask("manage-test")
    monkeypatch.setattr(manage, "get_app", lambda: app)
    return app


def test_eventlet_helpers_et_bootstrap(monkeypatch, capsys):
    assert is_migration_command(["manage.py", "db", "upgrade"]) is True
    assert is_migration_command(["manage.py", "seed", "demo"]) is False
    assert env_disables_eventlet("1") is True
    assert env_disables_eventlet("0") is False
    assert running_under_pytest() is True
    assert resolve_config_name("") == "development"
    assert resolve_config_name("testing") == "testing"
    resolve_config_name()
    assert env_disables_eventlet() in {True, False}

    patched = MagicMock()
    monkeypatch.setattr(manage, "apply_eventlet_monkey_patch", patched)
    bootstrap_eventlet(apply_patch=False)

    disabled = bootstrap_eventlet(
        ["manage.py", "upgrade"],
        disable_env="0",
        apply_patch=False,
    )
    assert disabled is True
    warn = capsys.readouterr().out
    assert "Commande de migration détectée" in warn
    patched.assert_not_called()

    assert bootstrap_eventlet(["manage.py"], disable_env="1", apply_patch=True) is True
    ok = capsys.readouterr().out
    assert "eventlet désactivé" in ok
    patched.assert_not_called()

    assert bootstrap_eventlet(["manage.py"], disable_env="0", apply_patch=True) is False
    patched.assert_called_once()

    fake_eventlet = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "eventlet", fake_eventlet)
    apply_eventlet_monkey_patch()
    fake_eventlet.monkey_patch.assert_called_once()


def test_getattr_app_et_inconnu(monkeypatch):
    sentinel = Flask("lazy")
    monkeypatch.setattr(manage, "_app", None)
    monkeypatch.setattr(manage, "create_app", lambda _cfg: sentinel)
    assert manage.app is sentinel
    missing_attr = "nexiste_pas"
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(manage, missing_attr)


def test_cli_db_et_seed(dummy_app, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(manage, "_init", MagicMock())
    monkeypatch.setattr(manage, "_migrate", MagicMock())
    monkeypatch.setattr(manage, "_upgrade", MagicMock())
    monkeypatch.setattr(manage, "_stamp", MagicMock())
    seed = MagicMock(return_value={"users": 2})
    monkeypatch.setattr(manage, "reset_and_seed_demo_dataset", seed)

    assert runner.invoke(cli, ["--help"]).exit_code == 0

    init_res = runner.invoke(cli, ["db", "init"])
    assert init_res.exit_code == 0
    assert "initialisé" in init_res.output
    manage._init.assert_called_once()

    mig = runner.invoke(cli, ["db", "migrate", "-m", "add foo"])
    assert mig.exit_code == 0
    manage._migrate.assert_called_once_with(message="add foo")

    up = runner.invoke(cli, ["db", "upgrade"])
    assert up.exit_code == 0
    manage._upgrade.assert_called_once()

    stamped = runner.invoke(cli, ["db", "stamp"])
    assert stamped.exit_code == 0
    manage._stamp.assert_called_once_with(revision="head")
    assert "head" in stamped.output

    stamped2 = runner.invoke(cli, ["db", "stamp", "abc123"])
    assert stamped2.exit_code == 0
    manage._stamp.assert_called_with(revision="abc123")

    demo = runner.invoke(cli, ["seed", "demo", "--reset", "--profile", "sales"])
    assert demo.exit_code == 0
    seed.assert_called_once_with(profile_name="sales", reset=True)
    assert "profile=sales" in demo.output


def test_get_app_cache(monkeypatch):
    created = Flask("cached")
    factory = MagicMock(return_value=created)
    monkeypatch.setattr(manage, "_app", None)
    monkeypatch.setattr(manage, "create_app", factory)
    assert get_app() is created
    assert get_app() is created
    factory.assert_called_once()
