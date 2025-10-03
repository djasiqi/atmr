#!/usr/bin/env python3
"""WSGI entrypoint pour Gunicorn : expose `app`."""
from __future__ import annotations
import os
from app import create_app

# Gunicorn s'attend à trouver `app` dans ce module.
# On choisit la conf via FLASK_ENV/FLASK_CONFIG (fallback: production).
_cfg = os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG") or "production"
app = create_app(_cfg)