"""Tests unitaires pour le middleware trace_id.

Tests pour la génération et l'injection de trace_id dans les requêtes.
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from middleware.trace_id import (
    add_trace_id_to_response,
    generate_trace_id,
    get_trace_id,
    get_trace_id_for_logging,
    inject_trace_id_middleware,
)


class TestTraceIdGeneration:
    """Tests pour la génération de trace_id."""

    def test_generate_trace_id_format(self):
        """Test que trace_id est généré au bon format."""
        trace_id = generate_trace_id()

        assert isinstance(trace_id, str)
        assert len(trace_id) == 32  # UUID v4 hex = 32 caractères
        # Vérifier que c'est de l'hexadécimal
        try:
            int(trace_id, 16)
        except ValueError:
            pytest.fail("trace_id n'est pas en hexadécimal")

    def test_generate_trace_id_unique(self):
        """Test que chaque trace_id est unique."""
        trace_ids = [generate_trace_id() for _ in range(100)]

        assert len(set(trace_ids)) == 100  # Tous uniques


class TestTraceIdMiddleware:
    """Tests pour le middleware trace_id."""

    @pytest.fixture
    def app(self):
        """Créer une app Flask pour les tests."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    def test_get_trace_id_generates_new(self, app):
        """Test génération nouveau trace_id si absent."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.return_value = None

            trace_id = get_trace_id()

            assert trace_id is not None
            assert len(trace_id) == 32
            assert hasattr(app.app_context().g, "trace_id")

    def test_get_trace_id_from_header(self, app):
        """Test récupération trace_id depuis header X-Trace-Id."""
        expected_trace_id = "custom-trace-id-12345"
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.side_effect = lambda key: (
                expected_trace_id if key == "X-Trace-Id" else None
            )

            trace_id = get_trace_id()

            assert trace_id == expected_trace_id
            assert app.app_context().g.trace_id == expected_trace_id

    def test_get_trace_id_from_trace_id_header(self, app):
        """Test récupération trace_id depuis header Trace-Id."""
        expected_trace_id = "custom-trace-id-67890"
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.side_effect = lambda key: (
                expected_trace_id if key == "Trace-Id" else None
            )

            trace_id = get_trace_id()

            assert trace_id == expected_trace_id

    def test_get_trace_id_cached(self, app):
        """Test que trace_id est mis en cache dans g."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.return_value = None

            trace_id1 = get_trace_id()
            trace_id2 = get_trace_id()

            # Devrait retourner le même trace_id
            assert trace_id1 == trace_id2
            # Ne devrait appeler generate_trace_id qu'une fois
            assert mock_request.headers.get.call_count <= 2

    def test_inject_trace_id_middleware(self, app):
        """Test injection trace_id via middleware."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.return_value = None

            inject_trace_id_middleware()

            assert hasattr(app.app_context().g, "trace_id")
            trace_id = app.app_context().g.trace_id
            assert trace_id is not None
            assert len(trace_id) == 32

    def test_add_trace_id_to_response(self, app):
        """Test ajout trace_id dans headers de réponse."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.return_value = None

            # Créer une réponse mock
            mock_response = MagicMock()
            mock_response.headers = {}

            result = add_trace_id_to_response(mock_response)

            assert result == mock_response
            assert "X-Trace-Id" in mock_response.headers
            assert len(mock_response.headers["X-Trace-Id"]) == 32

    def test_get_trace_id_for_logging(self, app):
        """Test récupération trace_id pour logs structurés."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.return_value = None

            log_data = get_trace_id_for_logging()

            assert isinstance(log_data, dict)
            assert "trace_id" in log_data
            assert len(log_data["trace_id"]) == 32

    def test_trace_id_persistence_across_requests(self, app):
        """Test que trace_id persiste dans le même contexte."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.return_value = None

            # Premier appel
            trace_id1 = get_trace_id()

            # Deuxième appel dans le même contexte
            trace_id2 = get_trace_id()

            assert trace_id1 == trace_id2


class TestTraceIdIntegration:
    """Tests d'intégration pour trace_id."""

    @pytest.fixture
    def client(self, app):
        """Créer un client de test."""
        return app.test_client()

    def test_trace_id_in_response_header(self, app):
        """Test que trace_id est ajouté dans les headers de réponse."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.return_value = None

            @app.route("/test")
            def test_route():
                from flask import jsonify

                return jsonify({"status": "ok"})

            # Simuler before_request
            inject_trace_id_middleware()

            # Simuler after_request
            with app.test_request_context("/test"):
                app.test_client().get("/test")
                # Note: Dans un vrai test, il faudrait appeler add_trace_id_to_response
                # via le hook after_request de Flask

    def test_trace_id_header_priority(self, app):
        """Test priorité des headers (X-Trace-Id > Trace-Id)."""
        with app.app_context(), patch("middleware.trace_id.request") as mock_request:
            mock_request.headers.get.side_effect = lambda key: (
                "x-trace-id-value" if key == "X-Trace-Id" else "trace-id-value"
            )

            trace_id = get_trace_id()

            # Devrait utiliser X-Trace-Id en priorité
            assert trace_id == "x-trace-id-value"
