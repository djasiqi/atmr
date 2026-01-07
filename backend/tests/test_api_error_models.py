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

        # Vérifier les champs
        fields = model.__schema__
        assert "error" in fields
        assert "message" in fields
        assert "trace_id" in fields
        assert "details" in fields

        # Vérifier que error et message sont requis
        assert fields["error"].required is True
        assert fields["message"].required is True
        assert fields["trace_id"].required is False

    def test_create_validation_error_model(self, api):
        """Test création modèle ValidationError."""
        model = create_validation_error_model(api)

        assert model is not None
        assert model.name == "ValidationError"

        fields = model.__schema__
        assert "error" in fields
        assert "message" in fields
        assert "trace_id" in fields
        assert "fields" in fields

        assert fields["error"].required is True
        assert fields["message"].required is True

    def test_create_not_found_error_model(self, api):
        """Test création modèle NotFoundError."""
        model = create_not_found_error_model(api)

        assert model is not None
        assert model.name == "NotFoundError"

        fields = model.__schema__
        assert "error" in fields
        assert "message" in fields
        assert "trace_id" in fields
        assert "resource" in fields
        assert "resource_id" in fields

        assert fields["error"].required is True
        assert fields["message"].required is True
        assert fields["resource"].required is False
        assert fields["resource_id"].required is False

    def test_create_permission_error_model(self, api):
        """Test création modèle PermissionError."""
        model = create_permission_error_model(api)

        assert model is not None
        assert model.name == "PermissionError"

        fields = model.__schema__
        assert "error" in fields
        assert "message" in fields
        assert "trace_id" in fields
        assert "required_role" in fields

        assert fields["error"].required is True
        assert fields["message"].required is True
        assert fields["required_role"].required is False

    def test_all_models_have_trace_id(self, api):
        """Test que tous les modèles incluent trace_id."""
        models = [
            create_api_error_model(api),
            create_validation_error_model(api),
            create_not_found_error_model(api),
            create_permission_error_model(api),
        ]

        for model in models:
            fields = model.__schema__
            assert "trace_id" in fields, f"Modèle {model.name} manque trace_id"
            assert fields["trace_id"].required is False

    def test_error_models_consistency(self, api):
        """Test cohérence entre les modèles d'erreur."""
        api_error = create_api_error_model(api)
        validation_error = create_validation_error_model(api)

        # Tous devraient avoir error et message requis
        for model in [api_error, validation_error]:
            fields = model.__schema__
            assert fields["error"].required is True
            assert fields["message"].required is True
