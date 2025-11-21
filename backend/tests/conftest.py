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

    # ✅ FIX: Passer explicitement "testing" pour désactiver force_https dans Talisman
    app = create_app(config_name="testing")

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
            # ✅ FIX: Configurer pour éviter les redirections 302 dans les tests E2E
            "SERVER_NAME": "localhost:5000",
            "PREFERRED_URL_SCHEME": "http",
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
        _db.session.expire_all()  # ✅ AJOUT: Expirer tous les objets pour forcer le rechargement après rollback
        _db.session.remove()


@pytest.fixture
def client(app, db):
    """Client de test Flask qui ne suit pas les redirections automatiquement."""
    # ✅ FIX: Ne pas suivre les redirections pour éviter les 302 dans les tests E2E
    # Les tests doivent pouvoir vérifier les codes HTTP directement (200, 400, etc.)
    # Flask moderne ne supporte plus follow_redirects dans test_client(), on crée un wrapper
    base = app.test_client()

    class NoRedirectClient:
        """Wrapper client qui définit follow_redirects=False par défaut."""

        def __init__(self, client):  # pyright: ignore[reportMissingSuperCall]
            # Cette classe n'hérite pas d'une classe parente qui nécessite super().__init__()
            self._client = client

        def _with_defaults(self, kwargs):
            # Ensure follow_redirects default is False for compatibility with older tests
            if "follow_redirects" not in kwargs:
                kwargs["follow_redirects"] = False
            return kwargs

        def get(self, *args, **kwargs):
            kwargs = self._with_defaults(kwargs)
            return self._client.get(*args, **kwargs)

        def post(self, *args, **kwargs):
            kwargs = self._with_defaults(kwargs)
            return self._client.post(*args, **kwargs)

        def put(self, *args, **kwargs):
            kwargs = self._with_defaults(kwargs)
            return self._client.put(*args, **kwargs)

        def patch(self, *args, **kwargs):
            kwargs = self._with_defaults(kwargs)
            return self._client.patch(*args, **kwargs)

        def delete(self, *args, **kwargs):
            kwargs = self._with_defaults(kwargs)
            return self._client.delete(*args, **kwargs)

        def open(self, *args, **kwargs):
            # low-level entrypoint used in tests sometimes
            kwargs = self._with_defaults(kwargs)
            return self._client.open(*args, **kwargs)

        def __getattr__(self, name):
            # delegate everything else to original client
            return getattr(self._client, name)

    return NoRedirectClient(base)


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


@pytest.fixture
def authenticated_client(client, sample_user):
    """Client Flask authentifié avec token JWT."""
    from flask_jwt_extended import create_access_token

    claims = {
        "role": sample_user.role.value,
        "company_id": getattr(sample_user, "company_id", None),
        "driver_id": getattr(sample_user, "driver_id", None),
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(identity=str(sample_user.public_id), additional_claims=claims)

    # Créer une classe wrapper qui ajoute automatiquement les headers
    class AuthenticatedClient(object):
        def __init__(self, client, token):
            super().__init__()
            self._client = client
            self._token = token
            self._headers = {"Authorization": f"Bearer {token}"}

        def _add_headers(self, kwargs):
            """Ajoute les headers d'authentification si non présents."""
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            kwargs["headers"].update(self._headers)
            return kwargs

        def get(self, *args, **kwargs):
            kwargs = self._add_headers(kwargs)
            return self._client.get(*args, **kwargs)

        def post(self, *args, **kwargs):
            kwargs = self._add_headers(kwargs)
            return self._client.post(*args, **kwargs)

        def put(self, *args, **kwargs):
            kwargs = self._add_headers(kwargs)
            return self._client.put(*args, **kwargs)

        def patch(self, *args, **kwargs):
            kwargs = self._add_headers(kwargs)
            return self._client.patch(*args, **kwargs)

        def delete(self, *args, **kwargs):
            kwargs = self._add_headers(kwargs)
            return self._client.delete(*args, **kwargs)

        def __getattr__(self, name):
            """Déléguer les autres attributs au client original."""
            return getattr(self._client, name)

    return AuthenticatedClient(client, token)


@pytest.fixture
def sample_booking(db, sample_company, sample_client):
    """Crée un booking de test pour les tests ML monitoring et autres."""
    from datetime import datetime, timedelta, timezone

    from models.booking import Booking
    from models.enums import BookingStatus

    booking = Booking()
    booking.customer_name = "Test Customer"
    booking.pickup_location = "Rue de Test 1, 1000 Lausanne"
    booking.dropoff_location = "Rue de Test 2, 1000 Lausanne"
    booking.pickup_lat = 46.2044
    booking.pickup_lon = 6.1432
    booking.dropoff_lat = 46.2100
    booking.dropoff_lon = 6.1500
    booking.booking_type = "standard"
    booking.scheduled_time = datetime.now(timezone.utc) + timedelta(hours=2)
    booking.amount = 50.0
    booking.status = BookingStatus.PENDING
    booking.user_id = sample_client.user_id
    booking.client_id = sample_client.id
    booking.company_id = sample_company.id
    booking.duration_seconds = 1800
    booking.distance_meters = 5000

    db.session.add(booking)
    db.session.flush()  # Use flush instead of commit to work with savepoints
    db.session.refresh(booking)
    return booking


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


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Mock automatique des services externes (OSRM, Redis) pour tous les tests.

    Cette fixture s'applique automatiquement à tous les tests pour éviter
    les appels réseau et améliorer la performance et la fiabilité des tests.
    """
    from unittest.mock import MagicMock

    # Mock OSRM - utiliser les mêmes fonctions que mock_osrm_client
    def mock_build_distance_matrix_osrm(coords, **kwargs):
        """Retourne une matrice de durées simulée (secondes) basée sur haversine."""
        from services.osrm_client import _fallback_eta_seconds

        n = len(coords)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                else:
                    duration = _fallback_eta_seconds(coords[i], coords[j])
                    row.append(float(duration))
            matrix.append(row)
        return matrix

    def mock_route_info(origin, dest, **kwargs):
        """Retourne des données de route simulées basées sur haversine."""
        from services.osrm_client import _fallback_eta_seconds, _haversine_km

        km = _haversine_km(origin, dest)
        duration_s = _fallback_eta_seconds(origin, dest)

        return {
            "duration": float(duration_s),
            "distance": int(km * 1000),  # mètres
            "geometry": {"type": "LineString", "coordinates": [[origin[1], origin[0]], [dest[1], dest[0]]]},
            "legs": [{"distance": int(km * 1000), "duration": float(duration_s)}],
            "fallback": False,  # Simuler un appel OSRM réussi
        }

    def mock_get_distance_time(origin, dest, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.osrm_client import _fallback_eta_seconds, _haversine_km

        km = _haversine_km(origin, dest)
        duration_s = _fallback_eta_seconds(origin, dest)
        return (km * 1000, duration_s)  # mètres, secondes

    def mock_get_matrix(origins, destinations, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.osrm_client import _fallback_eta_seconds, _haversine_km

        n, m = len(origins), len(destinations)
        durations = []
        distances = []
        for i in range(n):
            dur_row = []
            dist_row = []
            for j in range(m):
                km = _haversine_km(origins[i], destinations[j])
                duration_s = _fallback_eta_seconds(origins[i], destinations[j])
                dur_row.append(float(duration_s))
                dist_row.append(km * 1000)  # mètres
            durations.append(dur_row)
            distances.append(dist_row)
        return {"durations": durations, "distances": distances}

    def mock_eta_seconds(origin, dest, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.osrm_client import _fallback_eta_seconds

        return _fallback_eta_seconds(origin, dest)

    # Patcher OSRM
    from services import osrm_client

    monkeypatch.setattr(osrm_client, "build_distance_matrix_osrm", mock_build_distance_matrix_osrm)
    monkeypatch.setattr(osrm_client, "route_info", mock_route_info)
    monkeypatch.setattr(osrm_client, "get_distance_time", mock_get_distance_time)
    monkeypatch.setattr(osrm_client, "get_matrix", mock_get_matrix)
    monkeypatch.setattr(osrm_client, "eta_seconds", mock_eta_seconds)

    # Mock Redis - créer un mock Redis centralisé
    mock_redis = MagicMock()
    # Configurer les méthodes Redis courantes
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.lpush.return_value = 1
    mock_redis.lrange.return_value = []
    mock_redis.ltrim.return_value = True
    mock_redis.expire.return_value = True
    mock_redis.keys.return_value = []
    mock_redis.ping.return_value = True

    # Patcher Redis dans les modules qui l'utilisent
    # Note: On patch seulement si le module existe pour éviter les erreurs
    try:
        import redis

        # Mock redis.from_url pour retourner notre mock
        monkeypatch.setattr(redis, "from_url", lambda *args, **kwargs: mock_redis)
    except ImportError:
        pass

    # Patcher les clients Redis spécifiques si disponibles
    try:
        from services import redis_client

        monkeypatch.setattr(redis_client, "RedisClient", MagicMock(return_value=mock_redis))
    except ImportError:
        pass

    # Retourner un dictionnaire avec les mocks pour permettre l'accès si nécessaire
    # Note: Utilisation de return au lieu de yield car il n'y a pas de teardown nécessaire
    # Les mocks sont déjà appliqués via monkeypatch, donc ils sont actifs pour tous les tests
    return {
        "osrm": {
            "build_distance_matrix_osrm": mock_build_distance_matrix_osrm,
            "route_info": mock_route_info,
            "get_distance_time": mock_get_distance_time,
            "get_matrix": mock_get_matrix,
            "eta_seconds": mock_eta_seconds,
        },
        "redis": mock_redis,
    }


@pytest.fixture
def mock_osrm_client(monkeypatch):
    """Mock osrm_client fonctions pour éviter appels réseau.

    ✅ FIX: Mock les fonctions réelles utilisées (build_distance_matrix_osrm, route_info)
    au lieu de fonctions qui n'existent pas.
    """

    def mock_build_distance_matrix_osrm(coords, **kwargs):
        """Retourne une matrice de durées simulée (secondes) basée sur haversine."""
        from services.osrm_client import _fallback_eta_seconds

        n = len(coords)
        # Matrice symétrique avec durées simulées basées sur haversine
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                else:
                    # Simuler une durée basée sur la distance haversine
                    duration = _fallback_eta_seconds(coords[i], coords[j])
                    row.append(float(duration))
            matrix.append(row)
        return matrix

    def mock_route_info(origin, dest, **kwargs):
        """Retourne des données de route simulées basées sur haversine."""
        from services.osrm_client import _fallback_eta_seconds, _haversine_km

        km = _haversine_km(origin, dest)
        duration_s = _fallback_eta_seconds(origin, dest)

        return {
            "duration": float(duration_s),
            "distance": int(km * 1000),  # mètres
            "geometry": {"type": "LineString", "coordinates": [[origin[1], origin[0]], [dest[1], dest[0]]]},
            "legs": [{"distance": int(km * 1000), "duration": float(duration_s)}],
        }

    def mock_get_distance_time(origin, dest, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.osrm_client import _fallback_eta_seconds, _haversine_km

        km = _haversine_km(origin, dest)
        duration_s = _fallback_eta_seconds(origin, dest)
        return (km * 1000, duration_s)  # mètres, secondes

    def mock_get_matrix(origins, destinations, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.osrm_client import _fallback_eta_seconds, _haversine_km

        n, m = len(origins), len(destinations)
        durations = []
        distances = []
        for i in range(n):
            dur_row = []
            dist_row = []
            for j in range(m):
                km = _haversine_km(origins[i], destinations[j])
                duration_s = _fallback_eta_seconds(origins[i], destinations[j])
                dur_row.append(float(duration_s))
                dist_row.append(km * 1000)  # mètres
            durations.append(dur_row)
            distances.append(dist_row)
        return {"durations": durations, "distances": distances}

    def mock_eta_seconds(origin, dest, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.osrm_client import _fallback_eta_seconds

        return _fallback_eta_seconds(origin, dest)

    from services import osrm_client

    # ✅ FIX: Mock les fonctions réelles utilisées
    monkeypatch.setattr(osrm_client, "build_distance_matrix_osrm", mock_build_distance_matrix_osrm)
    monkeypatch.setattr(osrm_client, "route_info", mock_route_info)
    # Garder les anciens mocks pour compatibilité
    monkeypatch.setattr(osrm_client, "get_distance_time", mock_get_distance_time)
    monkeypatch.setattr(osrm_client, "get_matrix", mock_get_matrix)
    monkeypatch.setattr(osrm_client, "eta_seconds", mock_eta_seconds)
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


# ========== FIXTURES PII MASKING ==========


@pytest.fixture
def pii_config():
    """Configuration pour les tests PII masking.

    ✅ FIX: Fixture optionnelle pour configurer les variables d'environnement
    nécessaires aux tests PII (clés de chiffrement, etc.)
    Note: Les fonctions PII dans shared.logging_utils sont statiques,
    donc cette fixture est optionnelle mais utile pour certains tests.
    """
    import os

    # Configurer les clés de chiffrement si nécessaire
    os.environ.setdefault("APP_ENCRYPTION_KEY_B64", "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY")
    return True
