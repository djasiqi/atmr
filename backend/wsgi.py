#!/usr/bin/env python3
"""WSGI entrypoint pour Gunicorn : expose `app`."""

from __future__ import annotations

import os
import sys

# ⚠️ CRITIQUE: Empêcher pytest de s'exécuter automatiquement en production
# Si pytest est installé et qu'un hook pytest est déclenché lors de l'import,
# on doit l'empêcher explicitement
if os.getenv("FLASK_ENV") == "production" or os.getenv("FLASK_CONFIG") == "production":
    # Désactiver pytest en production
    os.environ["PYTEST_DISABLED"] = "1"
    os.environ["DISABLE_PYTEST"] = "1"

    # Empêcher pytest de collecter les tests en important pytest uniquement si nécessaire
    # et en désactivant les hooks automatiques
    try:
        import pytest

        # Désactiver la collecte automatique de tests
        if hasattr(pytest, "config"):
            # Ne pas exécuter pytest si on est en production
            pass
    except ImportError:
        pass  # pytest n'est pas installé, c'est parfait

# ✅ Force UTF-8 encoding pour Python
if sys.version_info >= (3, 7):
    # Python 3.7+ : utiliser la variable d'environnement
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # Reconfigurer stdout/stderr en UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# ✅ CRITIQUE : monkey_patch eventlet AVANT tout import Flask/SocketIO
_async_mode = (os.getenv("SOCKETIO_ASYNC_MODE") or "eventlet").strip().lower()

if _async_mode == "eventlet":
    try:
        import eventlet

        eventlet.monkey_patch()
        print("✅ [WSGI] eventlet.monkey_patch() appliqué", flush=True)
    except ImportError:
        print("⚠️ [WSGI] eventlet non disponible", flush=True)
elif _async_mode == "gevent":
    try:
        from gevent import monkey  # pyright: ignore[reportMissingModuleSource]

        monkey.patch_all()
        print("✅ [WSGI] gevent.monkey.patch_all() appliqué", flush=True)
    except ImportError:
        print("⚠️ [WSGI] gevent non disponible", flush=True)

from app import create_app  # noqa: E402

# Gunicorn s'attend à trouver `app` dans ce module.
# On choisit la conf via FLASK_ENV/FLASK_CONFIG (fallback: production).
_cfg = os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG") or "production"
app = create_app(_cfg)
print(f"✅ [WSGI] Application Flask créée (config={_cfg})", flush=True)
