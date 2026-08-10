"""Tests unitaires pour les modèles d'erreur API.

Tests pour les modèles Swagger standardisés d'erreurs.
"""

import pytest
from flask_restx import Api

from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)


def _schema_properties(model) -> tuple[dict, set[str]]:
    """Extrait propriétés et champs requis (OpenAPI 3 ou legacy Flask-RESTX)."""
    schema = model.__schema__
    if isinstance(schema, dict) and "properties" in schema:
        return schema["properties"], set(schema.get("required") or [])
    return schema, {k for k, v in schema.items() if getattr(v, "required", False)}


class TestAPIErrorModels:
    """Tests pour les modèles d'erreur API."""

    @pytest.fixture
    def api(self):
        """Créer une instance Flask-RESTX API pour les tests."""
        from flask import Flask

        app = Flask(__name__)
        return Api(app)

    def test_create_api_error_model(self, api):
        """Test création modèle APIError."""
        model = create_api_error_model(api)

        assert model is not None
        assert hasattr(model, "name")
        assert model.name == "APIError"

        props, required = _schema_properties(model)
        assert "error" in props
        assert "message" in props
        assert "trace_id" in props
        assert "details" in props

        assert "error" in required
        assert "message" in required
        assert "trace_id" not in required

    def test_create_validation_error_model(self, api):
        """Test création modèle ValidationError."""
        model = create_validation_error_model(api)

        assert model is not None
        assert model.name == "ValidationError"

        props, required = _schema_properties(model)
        assert "error" in props
        assert "message" in props
        assert "trace_id" in props
        assert "fields" in props

        assert "error" in required
        assert "message" in required

    def test_create_not_found_error_model(self, api):
        """Test création modèle NotFoundError."""
        model = create_not_found_error_model(api)

        assert model is not None
        assert model.name == "NotFoundError"

        props, required = _schema_properties(model)
        assert "error" in props
        assert "message" in props
        assert "trace_id" in props
        assert "resource" in props
        assert "resource_id" in props

        assert "error" in required
        assert "message" in required
        assert "resource" not in required
        assert "resource_id" not in required

    def test_create_permission_error_model(self, api):
        """Test création modèle PermissionError."""
        model = create_permission_error_model(api)

        assert model is not None
        assert model.name == "PermissionError"

        props, required = _schema_properties(model)
        assert "error" in props
        assert "message" in props
        assert "trace_id" in props
        assert "required_role" in props

        assert "error" in required
        assert "message" in required
        assert "required_role" not in required

    def test_all_models_have_trace_id(self, api):
        """Test que tous les modèles incluent trace_id."""
        models = [
            create_api_error_model(api),
            create_validation_error_model(api),
            create_not_found_error_model(api),
            create_permission_error_model(api),
        ]

        for model in models:
            props, required = _schema_properties(model)
            assert "trace_id" in props, f"Modèle {model.name} manque trace_id"
            assert "trace_id" not in required

    def test_error_models_consistency(self, api):
        """Test cohérence entre les modèles d'erreur."""
        api_error = create_api_error_model(api)
        validation_error = create_validation_error_model(api)

        for model in [api_error, validation_error]:
            _props, required = _schema_properties(model)
            assert "error" in required
            assert "message" in required
