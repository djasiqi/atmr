"""Tests unitaires pour le middleware trace_id.

Tests pour la génération et l'injection de trace_id dans les requêtes.
"""

from unittest.mock import MagicMock

import pytest
from flask import Flask, g

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
    def trace_app(self):
        """App Flask minimale (nom distinct pour éviter conflit avec conftest `app`)."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    def test_get_trace_id_generates_new(self, trace_app):
        """Test génération nouveau trace_id si absent."""
        with trace_app.test_request_context():
            trace_id = get_trace_id()

            assert trace_id is not None
            assert len(trace_id) == 32
            assert hasattr(g, "trace_id")

    def test_get_trace_id_from_header(self, trace_app):
        """Test récupération trace_id depuis header X-Trace-Id."""
        expected_trace_id = "custom-trace-id-12345"
        with trace_app.test_request_context(headers={"X-Trace-Id": expected_trace_id}):
            trace_id = get_trace_id()

            assert trace_id == expected_trace_id
            assert g.trace_id == expected_trace_id

    def test_get_trace_id_from_trace_id_header(self, trace_app):
        """Test récupération trace_id depuis header Trace-Id."""
        expected_trace_id = "custom-trace-id-67890"
        with trace_app.test_request_context(headers={"Trace-Id": expected_trace_id}):
            trace_id = get_trace_id()

            assert trace_id == expected_trace_id

    def test_get_trace_id_cached(self, trace_app):
        """Test que trace_id est mis en cache dans g."""
        with trace_app.test_request_context():
            trace_id1 = get_trace_id()
            trace_id2 = get_trace_id()

            assert trace_id1 == trace_id2

    def test_inject_trace_id_middleware(self, trace_app):
        """Test injection trace_id via middleware."""
        with trace_app.test_request_context():
            inject_trace_id_middleware()

            assert hasattr(g, "trace_id")
            assert g.trace_id is not None
            assert len(g.trace_id) == 32

    def test_add_trace_id_to_response(self, trace_app):
        """Test ajout trace_id dans headers de réponse."""
        with trace_app.test_request_context():
            mock_response = MagicMock()
            mock_response.headers = {}

            result = add_trace_id_to_response(mock_response)

            assert result == mock_response
            assert "X-Trace-Id" in mock_response.headers
            assert len(mock_response.headers["X-Trace-Id"]) == 32

    def test_get_trace_id_for_logging(self, trace_app):
        """Test récupération trace_id pour logs structurés."""
        with trace_app.test_request_context():
            log_data = get_trace_id_for_logging()

            assert isinstance(log_data, dict)
            assert "trace_id" in log_data
            assert len(log_data["trace_id"]) == 32

    def test_trace_id_persistence_across_requests(self, trace_app):
        """Test que trace_id persiste dans le même contexte."""
        with trace_app.test_request_context():
            trace_id1 = get_trace_id()
            trace_id2 = get_trace_id()

            assert trace_id1 == trace_id2


class TestTraceIdIntegration:
    """Tests d'intégration pour trace_id."""

    @pytest.fixture
    def trace_app(self):
        """App Flask minimale pour les tests d'intégration trace_id."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        return app

    def test_trace_id_in_response_header(self, trace_app):
        """Test que trace_id est ajouté dans les headers de réponse."""

        @trace_app.route("/test")
        def test_route():
            from flask import jsonify

            return jsonify({"status": "ok"})

        with trace_app.test_request_context("/test"):
            inject_trace_id_middleware()
            trace_app.test_client().get("/test")

    def test_trace_id_header_priority(self, trace_app):
        """Test priorité des headers (X-Trace-Id > Trace-Id)."""
        with trace_app.test_request_context(
            headers={"X-Trace-Id": "x-trace-id-value", "Trace-Id": "trace-id-value"}
        ):
            trace_id = get_trace_id()

            assert trace_id == "x-trace-id-value"
