#!/usr/bin/env python3
"""WSGI entrypoint pour Gunicorn : expose `app`."""
from __future__ import annotations

import os
import sys

# ✅ Force UTF-8 encoding pour Python
if sys.version_info >= (3, 7):  # noqa: UP036
    # Python 3.7+ : utiliser la variable d'environnement
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Reconfigurer stdout/stderr en UTF-8
    if hasattr(sys.stdout, 'reconfigure'):  # pyright: ignore[reportAttributeAccessIssue]
        sys.stdout.reconfigure(encoding='utf-8')  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(sys.stderr, 'reconfigure'):  # pyright: ignore[reportAttributeAccessIssue]
        sys.stderr.reconfigure(encoding='utf-8')  # pyright: ignore[reportAttributeAccessIssue]

# ✅ CRITIQUE : monkey_patch eventlet AVANT tout import Flask/SocketIO
_async_mode = (os.getenv("SOCKETIO_ASYNC_MODE") or "eventlet").strip().lower()

if _async_mode == "eventlet":
    try:
        import eventlet
        eventlet.monkey_patch()
        print("✅ [WSGI] eventlet.monkey_patch() appliqué", flush=True)  # noqa: T201
    except ImportError:
        print("⚠️ [WSGI] eventlet non disponible", flush=True)  # noqa: T201
elif _async_mode == "gevent":
    try:
        from gevent import monkey  # pyright: ignore[reportMissingModuleSource]
        monkey.patch_all()  # pyright: ignore[reportMissingModuleSource]
        print("✅ [WSGI] gevent.monkey.patch_all() appliqué", flush=True)  # noqa: T201
    except ImportError:
        print("⚠️ [WSGI] gevent non disponible", flush=True)  # noqa: T201

from app import create_app  # noqa: E402

# Gunicorn s'attend à trouver `app` dans ce module.
# On choisit la conf via FLASK_ENV/FLASK_CONFIG (fallback: production).
_cfg = os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG") or "production"
app = create_app(_cfg)
print(f"✅ [WSGI] Application Flask créée (config={_cfg})", flush=True)  # noqa: T201
