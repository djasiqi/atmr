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
# Désactiver la doc RESTX pour éviter les conflits d'endpoint /specs en tests
os.environ["API_DOCS"] = "off"
# Désactiver l'API legacy pendant les tests pour éviter conflits RestX
os.environ["API_LEGACY_ENABLED"] = "false"

from app import create_app
from ext import db as _db
from models import Company, User, UserRole


@pytest.fixture(scope="session")
def app() -> Flask:
    """Crée une instance Flask en mode test."""

    app = create_app()

    # ✅ FIX: Utiliser la DB PostgreSQL du workflow GitHub Actions pour les tests
    # Évite les problèmes d'enums, contraintes nommées, et JSONB
    # Les tests utilisent des savepoints (transactions nested) donc pas de risque pour les données
    # Workflow utilise test:test@localhost:5432/atmr_test
    database_url = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/atmr_test")

    app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": database_url,
            "WTF_CSRF_ENABLED": False,
            "JWT_SECRET_KEY": "test-secret-key",
            "SECRET_KEY": "test-secret-key",
            "SQLALCHEMY_ECHO": False,  # Pas de logs SQL verbeux en tests
        }
    )
    return app


@pytest.fixture
def app_context(app):
    """Crée un contexte d'application Flask pour les tests."""
    with app.app_context():
        yield app


@pytest.fixture
def db_session(db):
    """Alias pour db pour compatibilité avec les tests existants."""
    return db


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
    import uuid

    # ✅ Vérifier si une company existe déjà pour cet utilisateur
    existing_company = Company.query.filter_by(user_id=sample_user.id).first()
    if existing_company:
        return existing_company

    # Utiliser un email unique pour éviter les conflits potentiels
    unique_suffix = str(uuid.uuid4())[:8]
    company = Company()
    company.name = "Test Transport SA"
    company.address = "Rue de Test 1, 1000 Lausanne"
    company.contact_phone = "0211234567"
    company.contact_email = f"contact_{unique_suffix}@test-transport.ch"
    company.user_id = sample_user.id
    db.session.add(company)
    db.session.flush()  # Use flush instead of commit to work with savepoints
    return company


@pytest.fixture
def sample_user(db):
    """Crée un utilisateur de test (rôle company) sans supprimer d'entités liées."""
    import uuid

    unique_suffix = str(uuid.uuid4())[:8]
    user = User()
    user.username = f"testuser_{unique_suffix}"
    user.email = f"test-{unique_suffix}@example.com"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)

    db.session.add(user)
    db.session.flush()  # Use flush instead of commit to work with savepoints
    db.session.refresh(user)
    return user


@pytest.fixture
def sample_admin_user(db):
    """Crée un utilisateur admin de test."""
    import uuid

    unique_suffix = str(uuid.uuid4())[:8]
    user = User()
    user.username = f"admin_{unique_suffix}"
    user.email = f"admin-{unique_suffix}@example.com"
    user.role = UserRole.admin
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)

    db.session.add(user)
    db.session.flush()
    db.session.refresh(user)
    return user


@pytest.fixture
def auth_headers(client, sample_user):
    """Génère un token JWT valide pour l'utilisateur test sans appeler /login."""
    from flask_jwt_extended import create_access_token

    cache_key = f"token_{sample_user.id}"
    if not hasattr(auth_headers, "_token_cache"):
        auth_headers._token_cache = {}  # type: ignore[attr-defined]
    if cache_key in auth_headers._token_cache:  # type: ignore[attr-defined]
        token = auth_headers._token_cache[cache_key]  # type: ignore[attr-defined]
        return {"Authorization": f"Bearer {token}"}

    claims = {
        "role": sample_user.role.value,
        "company_id": getattr(sample_user, "company_id", None),
        "driver_id": getattr(sample_user, "driver_id", None),
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(identity=str(sample_user.public_id), additional_claims=claims)
    auth_headers._token_cache[cache_key] = token  # type: ignore[attr-defined]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(client, sample_admin_user):
    """Génère un token JWT valide pour un utilisateur admin."""
    from flask_jwt_extended import create_access_token

    cache_key = f"admin_token_{sample_admin_user.id}"
    if not hasattr(admin_headers, "_token_cache"):
        admin_headers._token_cache = {}  # type: ignore[attr-defined]
    if cache_key in admin_headers._token_cache:  # type: ignore[attr-defined]
        token = admin_headers._token_cache[cache_key]  # type: ignore[attr-defined]
        return {"Authorization": f"Bearer {token}"}

    claims = {
        "role": sample_admin_user.role.value,
        "company_id": getattr(sample_admin_user, "company_id", None),
        "driver_id": getattr(sample_admin_user, "driver_id", None),
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(identity=str(sample_admin_user.public_id), additional_claims=claims)
    admin_headers._token_cache[cache_key] = token  # type: ignore[attr-defined]
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
        company=sample_company, pickup_lat=46.2044, pickup_lon=6.1432, dropoff_lat=46.2100, dropoff_lon=6.1500
    )


@pytest.fixture
def simple_driver(db, sample_company):
    """Crée un driver simple avec position valide."""
    from tests.factories import create_driver_with_position

    return create_driver_with_position(company=sample_company, latitude=46.2044, longitude=6.1432, is_available=True)


@pytest.fixture
def sample_driver(factory_driver):
    """Alias pour factory_driver pour compatibilité avec les tests existants."""
    return factory_driver()


@pytest.fixture
def simple_assignment(db, simple_booking, simple_driver):
    """Crée un assignment simple avec booking et driver."""
    from tests.factories import create_assignment_with_booking_driver

    return create_assignment_with_booking_driver(
        booking=simple_booking, driver=simple_driver, company=simple_booking.company
    )


@pytest.fixture
def sample_client(db, sample_company):
    """Crée un client de test avec utilisateur associé."""
    import uuid

    from ext import bcrypt
    from models.client import Client
    from models.enums import UserRole
    from models.user import User

    # Utiliser un email unique pour éviter les conflits
    unique_suffix = str(uuid.uuid4())[:8]
    user = User()
    user.username = f"clientuser_{unique_suffix}"
    user.email = f"client-{unique_suffix}@example.com"
    user.role = UserRole.client
    user.first_name = "Jean"
    user.last_name = "Dupont"
    user.phone = "0791234567"
    user.address = "Rue Client 1, 1000 Lausanne"
    user.public_id = str(uuid.uuid4())
    password_hash = bcrypt.generate_password_hash("password123")
    user.password = password_hash.decode("utf-8") if isinstance(password_hash, bytes) else password_hash  # type: ignore[unnecessary-isinstance]
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = sample_company.id
    client.billing_address = "Rue Client 1, 1000 Lausanne"
    client.contact_email = user.email
    client.contact_phone = "0791234567"
    db.session.add(client)
    db.session.flush()  # Use flush instead of commit to work with savepoints
    return client


# ========== FIXTURES POUR MOCKS ==========


@pytest.fixture
def mock_osrm_client(monkeypatch):
    """Mock osrm_client fonctions pour éviter appels réseau."""

    def mock_get_distance_time(origin, dest, **kwargs):
        return (15.0, 1800.0)  # 15km, 1800 secondes (30 min)

    def mock_get_matrix(origins, destinations, **kwargs):
        n, m = len(origins), len(destinations)
        return {"durations": [[1800.0] * m for _ in range(n)], "distances": [[15000.0] * m for _ in range(n)]}

    def mock_eta_seconds(origin, dest, **kwargs):
        return 1800

    def mock_route_info(origin, dest, **kwargs):
        return {"duration_s": 1800.0, "distance_m": 15000.0, "geometry": "mock_geometry"}

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
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self.is_trained = True

        def predict_delay(self, booking, driver, current_time=None):
            from services.unified_dispatch.ml_predictor import DelayPrediction

            return DelayPrediction(
                booking_id=booking.id,
                predicted_delay_minutes=5.0,
                confidence=0.85,
                risk_level="medium",
                contributing_factors={"distance_x_weather": 0.42},
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
            return {"temperature": 20.0, "weather_factor": 0.5, "is_default": False}

        @staticmethod
        def get_weather_factor(lat, lon):
            return 0.5

    from services import weather_service

    monkeypatch.setattr(weather_service, "WeatherService", MockWeatherService)
    return MockWeatherService


# ========== FIXTURES SAFETY GUARDS ==========


@pytest.fixture
def safety_guards():
    """Crée une instance de SafetyGuards pour les tests."""
    try:
        from services.safety_guards import SafetyGuards

        return SafetyGuards()
    except ImportError:
        pytest.skip("SafetyGuards non disponible")


# ========== FIXTURES HELPERS ==========


@pytest.fixture
def cleanup_db(db):
    """Nettoie la DB après chaque test (supprime toutes les données)."""
    yield
    # Le rollback se fait déjà dans la fixture db(), mais on peut forcer ici
    db.session.rollback()
    db.session.remove()


# ========== FIXTURES D3 - CHAOS ENGINEERING ==========


@pytest.fixture
def reset_chaos():
    """Reset automatique du chaos injector après chaque test.

    ✅ D3: Garantit que le chaos est toujours désactivé après un test,
    même si le test échoue.
    """
    try:
        from chaos.injectors import get_chaos_injector

        injector = get_chaos_injector()

        yield injector

    except ImportError:
        # Module chaos non disponible, continuer normalement
        yield None
    finally:
        # Reset automatique après le test
        try:
            from chaos.injectors import get_chaos_injector

            injector = get_chaos_injector()
            injector.enabled = False
            injector.osrm_down = False
            injector.db_read_only = False
            injector.latency_ms = 0
            injector.error_rate = 0.0
            injector.timeout_rate = 0.0
        except ImportError:
            pass


@pytest.fixture
def chaos_injector():
    """Fixture pour obtenir l'injecteur de chaos avec reset automatique.

    ✅ D3: Retourne l'injecteur de chaos et garantit le reset après le test.

    Usage:
        def test_something(chaos_injector):
            chaos_injector.enable()
            chaos_injector.set_osrm_down(True)
            # ... test ...
    """
    try:
        from chaos.injectors import get_chaos_injector

        injector = get_chaos_injector()

        # S'assurer que le chaos est désactivé au départ
        injector.disable()
        injector.set_osrm_down(False)
        injector.set_db_read_only(False)

        yield injector

    except ImportError:
        # Module chaos non disponible, continuer normalement
        pytest.skip("Chaos injector module not available")
    finally:
        # Reset automatique après le test
        try:
            from chaos.injectors import get_chaos_injector, reset_chaos

            reset_chaos()
        except ImportError:
            pass


@pytest.fixture
def mock_osrm_down():
    """Fixture pour activer/désactiver automatiquement OSRM down.

    ✅ D3: Active OSRM down au début du test et le désactive à la fin.

    Usage:
        def test_with_osrm_down(mock_osrm_down):
            # OSRM down est automatiquement activé
            # ... test ...
            # OSRM down est automatiquement désactivé après le test
    """
    # Initialiser les variables pour éviter les erreurs de linter
    initial_enabled = False
    initial_osrm_down = False

    try:
        from chaos.injectors import get_chaos_injector

        injector = get_chaos_injector()

        # Sauvegarder l'état initial
        initial_enabled = injector.enabled
        initial_osrm_down = injector.osrm_down

        # Activer OSRM down
        injector.enable()
        injector.set_osrm_down(True)

        yield injector

    except ImportError:
        # Module chaos non disponible, continuer normalement
        pytest.skip("Chaos injector module not available")
    finally:
        # Restaurer l'état initial
        try:
            from chaos.injectors import get_chaos_injector

            injector = get_chaos_injector()
            injector.set_osrm_down(initial_osrm_down)
            if not initial_enabled:
                injector.disable()
        except ImportError:
            pass


@pytest.fixture
def mock_db_read_only():
    """Fixture pour activer/désactiver automatiquement DB read-only.

    ✅ D3: Active DB read-only au début du test et le désactive à la fin.

    Usage:
        def test_with_db_readonly(mock_db_read_only):
            # DB read-only est automatiquement activé
            # ... test ...
            # DB read-only est automatiquement désactivé après le test
    """
    # Initialiser les variables pour éviter les erreurs de linter
    initial_enabled = False
    initial_db_read_only = False

    try:
        from chaos.injectors import get_chaos_injector

        injector = get_chaos_injector()

        # Sauvegarder l'état initial
        initial_enabled = injector.enabled
        initial_db_read_only = injector.db_read_only

        # Activer DB read-only
        injector.enable()
        injector.set_db_read_only(True)

        yield injector

    except ImportError:
        # Module chaos non disponible, continuer normalement
        pytest.skip("Chaos injector module not available")
    finally:
        # Restaurer l'état initial
        try:
            from chaos.injectors import get_chaos_injector

            injector = get_chaos_injector()
            injector.set_db_read_only(initial_db_read_only)
            if not initial_enabled:
                injector.disable()
        except ImportError:
            pass
