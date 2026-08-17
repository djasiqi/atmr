"""Couverture de ``validate_migration`` (imports dispatch_routes → dispatch)."""

from __future__ import annotations

from types import SimpleNamespace

from validate_migration import (
    describe_namespace,
    ensure_encryption_key,
    import_dispatch_ns,
    import_init_namespaces,
    main,
    run_validation,
)


def test_ensure_encryption_key_setdefault():
    env: dict[str, str] = {}
    key = ensure_encryption_key(env)
    assert env["APP_ENCRYPTION_KEY_B64"] == key
    assert ensure_encryption_key(env) == key
    ensure_encryption_key()


def test_run_validation_succes_et_echecs(capsys, monkeypatch):
    ns = SimpleNamespace(name="dispatch", description="Dispatch API")
    assert describe_namespace(ns) == ("dispatch", "Dispatch API")

    assert run_validation(
        load_dispatch_ns=lambda: ns,
        load_init_namespaces=lambda: lambda: None,
    ) == 0
    ok = capsys.readouterr().out
    assert "Import dispatch_ns réussi" in ok
    assert "Migration validée" in ok

    assert run_validation(
        load_dispatch_ns=lambda: (_ for _ in ()).throw(RuntimeError("boom-ns")),
        load_init_namespaces=lambda: None,
    ) == 1
    assert "Erreur import dispatch_ns" in capsys.readouterr().out

    assert run_validation(
        load_dispatch_ns=lambda: ns,
        load_init_namespaces=lambda: "pas-callable",
    ) == 1
    assert "Erreur import routes_api" in capsys.readouterr().out

    class _BadNs:
        @property
        def name(self):
            raise RuntimeError("no-name")

    assert run_validation(
        load_dispatch_ns=_BadNs,
        load_init_namespaces=lambda: lambda: None,
    ) == 1
    assert "Erreur vérification namespace" in capsys.readouterr().out

    monkeypatch.setattr("validate_migration.run_validation", lambda: 0)
    assert main() == 0


def test_imports_reels_dispatch():
    ns = import_dispatch_ns()
    assert getattr(ns, "name", None)
    assert callable(import_init_namespaces())
