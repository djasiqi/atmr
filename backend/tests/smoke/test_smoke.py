"""Tests smoke pour vérifier le démarrage de l'application
et les connexions critiques."""

import os
from unittest.mock import patch

import pytest
from sqlalchemy import text

from app import create_app
from ext import db, redis_client


def _create_testing_app():
    # create_app() initialise aussi Flask-RESTX / routes globales.
    # Dans une suite de tests, plusieurs appels peuvent provoquer
    # "setup finished" si l'objet API global est réutilisé.
    # Pour les smoke tests, on valide le démarrage + extensions sans routes.
    with patch.dict(os.environ, {"SKIP_ROUTES_INIT": "true"}):
        return create_app("testing")


def test_app_starts():
    """Test que l'app démarre correctement."""
    app = _create_testing_app()
    assert app is not None
    assert app.config["TESTING"] is True


def test_db_connection():
    """Test que la DB est accessible."""
    app = _create_testing_app()
    with app.app_context():
        result = db.session.execute(text("SELECT 1")).fetchone()
        assert result is not None
        assert result[0] == 1


def test_redis_connection():
    """Test que Redis est accessible."""
    if redis_client is None:
        pytest.skip("Redis n'est pas configuré ou non disponible")

    try:
        result = redis_client.ping()
        assert result is True
    except Exception as e:
        pytest.fail(f"Redis ping a échoué: {e}")
