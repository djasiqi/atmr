"""
Fixtures pytest pour les tests backend ATMR.
"""
import os

# Mock JSONB → JSON AVANT tout import (SQLite ne supporte pas JSONB)
from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql

postgresql.JSONB = JSON

import pytest
from flask import Flask

# Forcer environnement de test avant d'importer l'app
os.environ["FLASK_ENV"] = "testing"
os.environ["PDF_BASE_URL"] = "http://localhost:5000"  # Valeur factice pour tests
os.environ.setdefault("TEST_DATABASE_URL", "postgresql://atmr:atmr@localhost:5432/atmr_test")

from app import create_app
from ext import db as _db
from models import Company, User, UserRole


@pytest.fixture(scope="session")
def app() -> Flask:
    """Crée une instance Flask en mode test."""

    app = create_app()

    # ✅ FIX: Utiliser la DB PostgreSQL du docker-compose pour les tests
    # Évite les problèmes d'enums, contraintes nommées, et JSONB
    # Les tests utilisent des savepoints (transactions nested) donc pas de risque pour les données
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://atmr:atmr@postgres:5432/atmr"
    )

    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": database_url,
        "WTF_CSRF_ENABLED": False,
        "JWT_SECRET_KEY": "test-secret-key",
        "SECRET_KEY": "test-secret-key",
        "SQLALCHEMY_ECHO": False,  # Pas de logs SQL verbeux en tests
    })
    return app


@pytest.fixture
def db(app):
    """Crée une DB propre pour chaque test en utilisant des savepoints."""
    with app.app_context():
        # ✅ FIX: Utiliser un savepoint (nested transaction) pour rollback automatique
        # Chaque test démarre avec une DB vide et rollback à la fin

        # Commencer une transaction nested (SAVEPOINT)
        _db.session.begin_nested()

        yield _db

        # Rollback automatique du savepoint
        _db.session.rollback()
        _db.session.remove()


@pytest.fixture
def client(app, db):
    """Client de test Flask."""
    return app.test_client()


@pytest.fixture
def sample_company(db, sample_user):
    """Crée une entreprise de test."""
    company = Company(
        name="Test Transport SA",
        address="Rue de Test 1, 1000 Lausanne",
        contact_phone="0211234567",
        contact_email="contact@test-transport.ch",
        user_id=sample_user.id
    )
    db.session.add(company)
    db.session.commit()
    return company


@pytest.fixture
def sample_user(db):
    """Crée un utilisateur de test (rôle company)."""
    import uuid

    from ext import bcrypt

    # ✅ FIX: Générer un username unique par test pour éviter collisions
    unique_suffix = str(uuid.uuid4())[:8]

    # ✅ FIX: Vérifier si l'utilisateur existe déjà, sinon créer
    existing = User.query.filter_by(username="testuser").first()
    if existing:
        return existing

    user = User(
        username="testuser",  # Utiliser un username fixe mais vérifier d'abord
        email=f"test-{unique_suffix}@example.com",
        role=UserRole.company
    )
    # ✅ FIX: Générer le hash correctement avec bcrypt
    password_hash = bcrypt.generate_password_hash("password123")
    if isinstance(password_hash, bytes):
        user.password = password_hash.decode("utf-8")
    else:
        user.password = password_hash

    db.session.add(user)
    db.session.commit()
    db.session.refresh(user)  # ✅ Rafraîchir pour obtenir l'ID
    return user


@pytest.fixture
def auth_headers(client, sample_user):
    """Génère un token JWT valide pour l'utilisateur test."""
    response = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "password123"
    })
    data = response.get_json()
    if not data or "token" not in data:
        pytest.fail(f"Login failed: {response.get_json()}")

    token = data["token"]
    return {"Authorization": f"Bearer {token}"}


# ========== FIXTURES AVANCÉES AVEC FACTORIES ==========

@pytest.fixture
def factory_company(db):
    """Factory pour créer des companies de test."""
    from tests.factories import CompanyFactory
    return CompanyFactory


@pytest.fixture
def factory_driver(db):
    """Factory pour créer des drivers de test."""
    from tests.factories import DriverFactory
    return DriverFactory


@pytest.fixture
def factory_booking(db):
    """Factory pour créer des bookings de test."""
    from tests.factories import BookingFactory
    return BookingFactory


@pytest.fixture
def factory_assignment(db):
    """Factory pour créer des assignments de test."""
    from tests.factories import AssignmentFactory
    return AssignmentFactory


@pytest.fixture
def factory_client(db):
    """Factory pour créer des clients de test."""
    from tests.factories import ClientFactory
    return ClientFactory


@pytest.fixture
def factory_user(db):
    """Factory pour créer des users de test."""
    from tests.factories import UserFactory
    return UserFactory


# ========== FIXTURES POUR SCÉNARIOS DISPATCH ==========

@pytest.fixture
def dispatch_scenario(db):
    """
    Crée un scénario de dispatch complet (company, drivers, bookings, dispatch_run).
    Returns:
        dict avec company, drivers, bookings, dispatch_run
    """
    from tests.factories import create_dispatch_scenario
    return create_dispatch_scenario(num_bookings=5, num_drivers=3)


@pytest.fixture
def simple_booking(db, sample_company):
    """Crée un booking simple avec coordonnées valides."""
    from tests.factories import create_booking_with_coordinates
    return create_booking_with_coordinates(
        company=sample_company,
        pickup_lat=46.2044,
        pickup_lon=6.1432,
        dropoff_lat=46.2100,
        dropoff_lon=6.1500
    )


@pytest.fixture
def simple_driver(db, sample_company):
    """Crée un driver simple avec position valide."""
    from tests.factories import create_driver_with_position
    return create_driver_with_position(
        company=sample_company,
        latitude=46.2044,
        longitude=6.1432,
        is_available=True
    )


@pytest.fixture
def simple_assignment(db, simple_booking, simple_driver):
    """Crée un assignment simple avec booking et driver."""
    from tests.factories import create_assignment_with_booking_driver
    return create_assignment_with_booking_driver(
        booking=simple_booking,
        driver=simple_driver,
        company=simple_booking.company
    )


# ========== FIXTURES POUR MOCKS ==========

@pytest.fixture
def mock_osrm_client(monkeypatch):
    """Mock osrm_client fonctions pour éviter appels réseau."""
    def mock_get_distance_time(origin, dest, **kwargs):
        return (15.0, 1800.0)  # 15km, 1800 secondes (30 min)

    def mock_get_matrix(origins, destinations, **kwargs):
        n, m = len(origins), len(destinations)
        return {
            "durations": [[1800.0] * m for _ in range(n)],
            "distances": [[15000.0] * m for _ in range(n)]
        }

    def mock_eta_seconds(origin, dest, **kwargs):
        return 1800

    def mock_route_info(origin, dest, **kwargs):
        return {
            "duration_s": 1800.0,
            "distance_m": 15000.0,
            "geometry": "mock_geometry"
        }

    from services import osrm_client
    monkeypatch.setattr(osrm_client, "get_distance_time", mock_get_distance_time)
    monkeypatch.setattr(osrm_client, "get_matrix", mock_get_matrix)
    monkeypatch.setattr(osrm_client, "eta_seconds", mock_eta_seconds)
    monkeypatch.setattr(osrm_client, "route_info", mock_route_info)
    return True


@pytest.fixture
def mock_ml_predictor(monkeypatch):
    """Mock MLPredictor pour tests rapides."""
    class MockMLPredictor:
        def __init__(self, *args, **kwargs):
            self.is_trained = True

        def predict_delay(self, ____________________________________________________________________________________________________booking, driver, current_time=None):
            from services.unified_dispatch.ml_predictor import DelayPrediction
            return DelayPrediction(
                booking_id=booking.id,
                predicted_delay_minutes=5.0,
                confidence=0.85,
                risk_level="medium",
                contributing_factors={"distance_x_weather": 0.42}
            )

    from services.unified_dispatch import ml_predictor
    monkeypatch.setattr(ml_predictor, "DelayMLPredictor", MockMLPredictor)
    return MockMLPredictor()


@pytest.fixture
def mock_weather_service(monkeypatch):
    """Mock WeatherService pour éviter appels API."""
    class MockWeatherService:
        @staticmethod
        def get_weather(lat, lon):
            return {
                "temperature": 20.0,
                "weather_factor": 0.5,
                "is_default": False
            }

        @staticmethod
        def get_weather_factor(lat, lon):
            return 0.5

    from services import weather_service
    monkeypatch.setattr(weather_service, "WeatherService", MockWeatherService)
    return MockWeatherService


# ========== FIXTURES HELPERS ==========

@pytest.fixture
def cleanup_db(db):
    """Nettoie la DB après chaque test (supprime toutes les données)."""
    yield
    # Le rollback se fait déjà dans la fixture db(), mais on peut forcer ici
    db.session.rollback()
    db.session.remove()
