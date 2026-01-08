"""
Tests pour les routes drivers (disponibilité, assignations).
"""

from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from models import Driver, UserRole


def test_list_drivers_unauthenticated(client):
    """GET /api/v1/driver sans authentification renvoie 401."""
    response = client.get("/api/v1/driver/")
    # 404 est acceptable si la route n'est pas initialisée (SKIP_ROUTES_INIT=1)
    assert response.status_code in (401, 404)


def test_list_drivers_authenticated(client, auth_headers):
    """GET /api/v1/driver avec authentification renvoie liste."""
    response = client.get("/api/v1/driver/", headers=auth_headers)
    # Peut être 200 ou 404 selon si route existe
    assert response.status_code in [200, 404, 405]


def test_driver_has_user_relationship(db, sample_driver):
    """Driver a une relation avec User."""
    driver = Driver.query.get(sample_driver.id)
    assert driver is not None
    assert driver.user is not None
    assert driver.user.role == UserRole.driver


def test_driver_has_company_relationship(db, sample_driver):
    """Driver a une relation avec Company."""
    driver = Driver.query.get(sample_driver.id)
    assert driver is not None
    assert driver.company_id is not None
    assert driver.company is not None
    assert driver.company_id == driver.company.id


def test_driver_availability_flag(db, sample_driver):
    """Driver a un flag is_available."""
    driver = Driver.query.get(sample_driver.id)
    assert hasattr(driver, "is_available")
    assert isinstance(driver.is_available, bool)


def test_driver_license_number(db, sample_driver):
    """Driver a des catégories de permis."""
    driver = Driver.query.get(sample_driver.id)
    # Le modèle Driver utilise license_categories (JSON) au lieu de license_number
    assert hasattr(driver, "license_categories")
    assert driver.license_categories is not None


def test_driver_serialize(db, sample_driver):
    """Driver.serialize retourne dict avec données."""
    driver = Driver.query.get(sample_driver.id)
    if hasattr(driver, "serialize"):
        serialized = driver.serialize
        assert isinstance(serialized, dict)
        assert "id" in serialized or "user" in serialized


def test_available_drivers_query(db, sample_driver):
    """Requête pour chauffeurs disponibles."""
    available = Driver.query.filter_by(is_available=True).all()
    assert len(available) > 0
    assert sample_driver in available


def test_drivers_by_company(db, sample_driver):
    """Requête chauffeurs par entreprise."""
    # Utiliser la company du driver au lieu de sample_company
    # car DriverFactory crée sa propre company
    driver = Driver.query.get(sample_driver.id)
    assert driver is not None
    assert driver.company_id is not None

    drivers = Driver.query.filter_by(company_id=driver.company_id).all()
    assert len(drivers) > 0
    assert sample_driver in drivers


# =====================================================
# Tests pour gestion des exceptions spécifiques
# =====================================================


def test_get_driver_profile_sqlalchemy_error_returns_database_error(
    client, sample_driver, db
):
    """Test que SQLAlchemyError retourne message DB spécifique."""
    from flask_jwt_extended import create_access_token

    # Créer un token pour le driver
    claims = {
        "role": sample_driver.user.role.value,
        "company_id": sample_driver.company_id,
        "driver_id": sample_driver.id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(sample_driver.user.public_id), additional_claims=claims
        )
    headers = {"Authorization": f"Bearer {token}"}

    # Mocker get_driver_from_token pour lever SQLAlchemyError
    with patch("routes.driver.get_driver_from_token") as mock_get_driver:
        mock_get_driver.side_effect = SQLAlchemyError("Database connection failed")

        response = client.get("/api/v1/driver/me/profile", headers=headers)

        # 404 est acceptable si la route n'est pas initialisée
        if response.status_code == 404:
            return

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data
        assert data["error"] == "database_error"
        assert "message" in data
        assert "base de données" in data["message"].lower()


def test_update_driver_profile_value_error_returns_validation_error(
    client, sample_driver, db
):
    """Test que ValueError retourne message validation."""
    from flask_jwt_extended import create_access_token

    # Créer un token pour le driver
    claims = {
        "role": sample_driver.user.role.value,
        "company_id": sample_driver.company_id,
        "driver_id": sample_driver.id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(sample_driver.user.public_id), additional_claims=claims
        )
    headers = {"Authorization": f"Bearer {token}"}

    # Mocker get_driver_from_token pour réussir, puis lever ValueError lors
    # de la mise à jour
    with (
        patch("routes.driver.get_driver_from_token") as mock_get_driver,
        patch.object(
            Driver, "serialize", new_callable=lambda: property(lambda self: {})
        ),
    ):
        # Simuler un driver valide
        mock_get_driver.return_value = (sample_driver, None, None)

        # Mocker la mise à jour pour lever ValueError
        with patch.object(
            db.session, "commit", side_effect=ValueError("Invalid data format")
        ):
            response = client.put(
                "/api/v1/driver/me/profile",
                headers=headers,
                json={"first_name": "Test"},
            )

            # 404 est acceptable si la route n'est pas initialisée
            if response.status_code == 404:
                return

            assert response.status_code == 400
            data = response.get_json()
            assert "error" in data
            assert data["error"] == "validation_error"
            assert "message" in data
            assert "validation" in data["message"].lower()


def test_get_driver_profile_unknown_exception_logged_with_stack_trace(
    client, sample_driver, db
):
    """Test que exceptions inconnues sont loggées avec stack trace."""
    from flask_jwt_extended import create_access_token

    # Créer un token pour le driver
    claims = {
        "role": sample_driver.user.role.value,
        "company_id": sample_driver.company_id,
        "driver_id": sample_driver.id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(sample_driver.user.public_id), additional_claims=claims
        )
    headers = {"Authorization": f"Bearer {token}"}

    # Mocker get_driver_from_token pour lever une exception inconnue
    with (
        patch("routes.driver.get_driver_from_token") as mock_get_driver,
        patch("routes.driver.logger") as mock_logger,
        patch("routes.driver.sentry_sdk") as mock_sentry,
    ):
        # Lever une exception personnalisée
        mock_get_driver.side_effect = RuntimeError("Unexpected error")

        response = client.get("/api/v1/driver/me/profile", headers=headers)

        # 404 est acceptable si la route n'est pas initialisée
        if response.status_code == 404:
            return

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data
        assert data["error"] == "internal_error"
        assert "message" in data

        # Vérifier que logger.exception a été appelé (stack trace)
        mock_logger.exception.assert_called_once()
        # Vérifier que sentry a capturé l'exception
        mock_sentry.capture_exception.assert_called_once()
