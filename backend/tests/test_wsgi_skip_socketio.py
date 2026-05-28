"""Boot WSGI sans monkey-patch quand SKIP_SOCKETIO=1."""

from __future__ import annotations

import importlib
import os
import sys


def test_wsgi_skips_gevent_when_skip_socketio(monkeypatch):
    monkeypatch.setenv("SKIP_SOCKETIO", "true")
    monkeypatch.setenv("SOCKETIO_ASYNC_MODE", "gevent")
    for mod in ("wsgi", "app", "ext"):
        sys.modules.pop(mod, None)
    wsgi = importlib.import_module("wsgi")
    assert hasattr(wsgi, "app")
