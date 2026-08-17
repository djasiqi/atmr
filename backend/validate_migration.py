#!/usr/bin/env python3
"""Script de validation de la migration dispatch_routes -> dispatch."""

from __future__ import annotations

import base64
import os
import sys
from collections.abc import Callable
from typing import Any


def ensure_encryption_key(environ: dict[str, str] | None = None) -> str:
    """Génère une clé d'encryption valide pour les tests si absente."""
    env = os.environ if environ is None else environ
    return env.setdefault(
        "APP_ENCRYPTION_KEY_B64",
        base64.b64encode(os.urandom(32)).decode(),
    )


def import_dispatch_ns() -> Any:
    """Importe ``dispatch_ns`` depuis ``routes.dispatch``."""
    from routes.dispatch import dispatch_ns

    return dispatch_ns


def import_init_namespaces() -> Any:
    """Importe ``init_namespaces`` depuis ``routes_api``."""
    from routes_api import init_namespaces

    return init_namespaces


def describe_namespace(dispatch_ns: Any) -> tuple[str, str]:
    """Retourne (name, description) du namespace."""
    return dispatch_ns.name, dispatch_ns.description


def run_validation(
    *,
    load_dispatch_ns: Callable[[], Any] = import_dispatch_ns,
    load_init_namespaces: Callable[[], Any] = import_init_namespaces,
) -> int:
    """Valide les imports de migration. Retourne 0 si OK, 1 sinon."""
    print("🔍 Validation de la migration dispatch_routes -> dispatch...")
    print()

    try:
        dispatch_ns = load_dispatch_ns()
        print("✅ Import dispatch_ns réussi")
    except Exception as e:
        print(f"❌ Erreur import dispatch_ns: {e}")
        return 1

    try:
        init_namespaces = load_init_namespaces()
        assert callable(init_namespaces), "init_namespaces doit être callable"
        print("✅ Import routes_api réussi")
    except Exception as e:
        print(f"❌ Erreur import routes_api: {e}")
        return 1

    try:
        name, description = describe_namespace(dispatch_ns)
        print(f"✅ Namespace dispatch_ns créé: {name}")
        print(f"   Description: {description}")
    except Exception as e:
        print(f"❌ Erreur vérification namespace: {e}")
        return 1

    print()
    print("✅ Migration validée : tous les imports fonctionnent correctement !")
    print("   Le nouveau module routes.dispatch est opérationnel.")
    return 0


def main() -> int:
    ensure_encryption_key()
    return run_validation()


if __name__ == "__main__":
    sys.exit(main())
