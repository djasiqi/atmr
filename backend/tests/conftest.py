"""
Fixtures pytest pour les tests backend ATMR.

📚 BONNES PRATIQUES D'ISOLATION DES TESTS
=========================================

1. **Isolation via Savepoints** :
   - Chaque test utilise un savepoint (nested transaction) via la fixture `db`
   - Le rollback automatique en fin de test garantit l'isolation entre les tests
   - Les objets commités dans les fixtures sont visibles dans le savepoint du test

2. **Fixtures persistées** :
   - Les fixtures qui créent des objets DB DOIVENT appeler `db.session.commit()`
   - Utiliser le helper `persisted_fixture()` pour créer des fixtures génériques
   - Recharger les objets depuis la DB après commit pour garantir la persistance

3. **Rollback défensif de engine.run()** :
   - `engine.run()` fait un rollback défensif qui peut expirer les objets non commités
   - TOUJOURS commit les objets avant d'appeler `engine.run()`
   - Utiliser `ensure_committed()` si nécessaire pour forcer un commit explicite

4. **Gestion des savepoints multiples** :
   - Utiliser `nested_savepoint()` pour créer des savepoints imbriqués si nécessaire
   - Chaque savepoint peut être rollback indépendamment
   - Le rollback du savepoint parent rollback tous les savepoints enfants

5. **Rechargement après rollback** :
   - Après un rollback, utiliser `db.session.expire_all()` puis recharger depuis la DB
   - Ne pas réutiliser les objets expirés sans les recharger
   - Utiliser `query.filter_by().first()` plutôt que `query.get()`
     pour forcer un nouveau query

📝 EXEMPLES D'UTILISATION :
---------------------------

```python
# Fixture générique persistée
@pytest.fixture
def my_entity(db):
    return persisted_fixture(db, MyEntityFactory(), MyEntity)

# Utilisation avec ensure_committed
def test_something(db, my_entity):
    with ensure_committed(db):
        result = engine.run(company_id=my_entity.id)

# Savepoint multiple
def test_nested_transaction(db):
    with nested_savepoint(db):
        # Créer des objets
        obj = MyEntityFactory()
        db.session.add(obj)
        db.session.commit()
        # Rollback automatique à la fin du context manager
```
"""

# Plugins de fixtures routes admin (doit être dans un conftest top-level du package tests).
pytest_plugins = ["tests.routes.admin_route_fixtures"]

import importlib.util
import os
from unittest import mock

# Note: Les tests backend utilisent PostgreSQL par défaut (voir DATABASE_URL plus bas),
# on conserve donc le type JSONB natif pour préserver les index GIN en migrations.
import pytest  # noqa: E402
from flask import Flask  # noqa: E402

# Forcer environnement de test avant d'importer l'app
os.environ["FLASK_ENV"] = "testing"
os.environ["PDF_BASE_URL"] = "http://localhost:5000"  # Valeur factice pour tests
os.environ.setdefault(
    "TEST_DATABASE_URL", "postgresql://atmr:atmr@localhost:5432/atmr_test"
)
# ✅ Tests d'intégration : Postgres par défaut si DATABASE_URL absent (respecte CI / env explicite)
# Default 5433 = port docker-compose.test.yml (postgres_test) pour éviter confusion 5432 vs 5433
os.environ.setdefault(
    "DATABASE_URL",
    os.getenv("DATABASE_URL_TEST") or "postgresql://test:test@localhost:5433/atmr_test",
)
# Désactiver la doc RESTX pour éviter les conflits d'endpoint /specs en tests
os.environ["API_DOCS"] = "off"
# Désactiver l'API legacy pendant les tests pour éviter conflits RestX
os.environ["API_LEGACY_ENABLED"] = "false"
# Socket.IO test_client ne supporte pas message_queue broker.
# Override strictement pour les tests backend.
os.environ["SOCKETIO_TEST_DISABLE_MESSAGE_QUEUE"] = "1"
os.environ.setdefault("DEMO_DEFAULT_PASSWORD", "test-demo-password-for-pytest")
os.environ.setdefault("ALLOW_NON_DEMO_SEED", "true")

from app import create_app  # noqa: E402
from ext import db as _db  # noqa: E402
from models import Company, User, UserRole  # noqa: E402


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def pytest_ignore_collect(collection_path: object, config: object) -> bool:
    """Skip optional heavy test suites when optional deps are not installed.

    Docker dev image used in local dev does not necessarily include RL/ML deps
    (torch/gymnasium/optuna). Without this guard, collection fails and blocks
    running coverage on the rest of the suite.

    Note: ``collection_path`` (pathlib) remplace ``path`` (py.path) depuis pytest 8.1.
    """
    _ = config
    p = str(collection_path).replace("\\", "/")

    run_e2e = os.getenv("RUN_E2E_TESTS", "0") in {"1", "true", "True"}
    has_torch = _module_available("torch")
    has_gymnasium = _module_available("gymnasium")
    has_optuna = _module_available("optuna")

    is_e2e = "/tests/e2e/" in p
    is_ml = "/tests/ml/" in p
    is_rl = "/tests/rl/" in p
    is_md5_migration = p.endswith("/tests/security/test_md5_to_sha256_migration.py")
    is_load_testing = "/tests/load_testing/" in p

    return (
        (is_e2e and not run_e2e)
        or (is_ml and not has_torch)
        or (is_rl and (not has_torch or not has_gymnasium or not has_optuna))
        or (is_md5_migration and not has_torch)
        or is_load_testing  # locust/urllib3 RecursionError avec Python 3.14
    )


def _build_session_flask_app() -> Flask:
    """Construit l'application Flask partagée pour la session de tests."""

    # ✅ FIX: Passer explicitement "testing" pour désactiver force_https dans Talisman
    app = create_app(config_name="testing")

    # ✅ Supprimer les warnings OpenTelemetry en mode test
    import logging

    # Ignorer les warnings OpenTelemetry dans les tests
    logging.getLogger("shared.otel_setup").setLevel(logging.ERROR)
    # Ignorer aussi les warnings de l'app Flask pour OpenTelemetry
    app.logger.setLevel(logging.ERROR)

    # ✅ FIX: Utiliser la DB PostgreSQL du workflow GitHub Actions
    # pour les tests
    # Évite les problèmes d'enums, contraintes nommées, et JSONB
    # Les tests utilisent des savepoints (transactions nested)
    # donc pas de risque pour les données
    # Default 5433 = port docker-compose.test.yml (postgres_test)
    # DATABASE_URL_TEST prioritaire si défini
    database_url = (
        os.getenv("DATABASE_URL_TEST")
        or os.getenv("DATABASE_URL")
        or "postgresql://test:test@localhost:5433/atmr_test"
    )

    app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": database_url,
            "WTF_CSRF_ENABLED": False,
            "CSRF_ENABLED": False,  # ✅ S1: Désactiver CSRF en tests
            "JWT_SECRET_KEY": "test-secret-key",
            "SECRET_KEY": "test-secret-key",
            # Les cookies de test doivent être host-only afin que le client Flask
            # les renvoie à localhost (un domaine de production les ignorerait).
            "COOKIE_DOMAIN": None,
            "SQLALCHEMY_ECHO": False,  # Pas de logs SQL verbeux en tests
            # Socket.IO test_client ne supporte pas message_queue.
            # Override strictement en tests, sans impact prod.
            "SOCKETIO_TEST_DISABLE_MESSAGE_QUEUE": True,
            # ✅ FIX: Configurer pour éviter les redirections 302 dans les tests E2E
            "SERVER_NAME": "localhost:5000",
            "PREFERRED_URL_SCHEME": "http",
            # Pas de tolérance JWT en tests : les tokens expirés doivent être rejetés.
            "JWT_DECODE_LEEWAY": 0,
        }
    )
    return app


@pytest.fixture(scope="session")
def _session_flask_app() -> Flask:
    """App Flask session-scoped (nom privé : évite ScopeMismatch avec fixtures locales `app`)."""
    return _build_session_flask_app()


@pytest.fixture(scope="session")
def app(_session_flask_app: Flask) -> Flask:
    """Crée une instance Flask en mode test."""
    return _session_flask_app


def _postgresql_schema_ready(app: Flask) -> bool:
    """Vérifie que le schéma PostgreSQL (table user) existe. Skip si SQLite ou erreur connexion."""
    uri = (app.config.get("SQLALCHEMY_DATABASE_URI") or "").lower()
    if "sqlite" in uri:
        return False
    try:
        from sqlalchemy import text

        with app.app_context():
            try:
                _db.session.execute(text('SELECT 1 FROM "user" LIMIT 1'))
                return True
            finally:
                _db.session.rollback()
    except Exception as e:
        err = str(e).lower()
        if (
            "does not exist" in err
            or "relation" in err
            or "connection" in err
            or "connect" in err
        ):
            return False
        raise


@pytest.fixture(scope="session")
def postgresql_schema_ready(_session_flask_app: Flask) -> bool:
    """Session-scoped: schéma PostgreSQL prêt pour les tests d'intégration."""
    return _postgresql_schema_ready(_session_flask_app)


@pytest.fixture(autouse=True)
def skip_integration_if_schema_missing(request, postgresql_schema_ready):
    """Skip les tests @pytest.mark.integration si le schéma PostgreSQL n'est pas prêt."""
    if "integration" in request.keywords and not postgresql_schema_ready:
        pytest.skip(
            "PostgreSQL schema not ready (table 'user' missing). "
            + "Run: docker compose -f docker-compose.test.yml up -d postgres_test && flask db upgrade"
        )


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
def mocker():
    """Compatibilité minimale avec pytest-mock dans l'image de test.

    Les tests l'utilisent avec ``mocker.patch.object(...)`` dans un contexte ;
    le module standard fournit exactement cette API sans dépendance additionnelle.
    """
    return mock


@pytest.fixture
def db(app):
    """Crée une DB propre pour chaque test en utilisant des savepoints."""
    import logging

    logger = logging.getLogger(__name__)
    logger.debug("🔧 [Fixture db] Setup: création savepoint")

    with app.app_context():
        # ✅ FIX: Utiliser un savepoint (nested transaction) pour rollback automatique
        # Chaque test démarre avec une DB vide et rollback à la fin

        # Commencer une transaction nested (SAVEPOINT)
        _db.session.begin_nested()
        logger.debug("🔧 [Fixture db] Savepoint créé")

        yield _db

        # Rollback automatique du savepoint
        logger.debug("🔧 [Fixture db] Teardown: rollback savepoint")
        try:
            _db.session.rollback()
            # ✅ AJOUT: Expirer tous les objets pour forcer
            # le rechargement après rollback
            _db.session.expire_all()
            _db.session.remove()
            logger.debug("🔧 [Fixture db] Teardown: rollback terminé")
        except Exception as e:
            logger.error("❌ [Fixture db] Erreur lors du teardown: %s", e)
            raise


@pytest.fixture
def client(app, db):
    """Client de test Flask qui ne suit pas les redirections automatiquement."""
    # ✅ FIX: Ne pas suivre les redirections pour éviter
    # les 302 dans les tests E2E
    # Les tests doivent pouvoir vérifier les codes HTTP directement
    # (200, 400, etc.)
    # Flask moderne ne supporte plus follow_redirects dans test_client(),
    # on crée un wrapper
    base = app.test_client()

    class NoRedirectClient:
        """Wrapper client qui définit follow_redirects=False par défaut."""

        def __init__(self, client):  # pyright: ignore[reportMissingSuperCall]
            # Cette classe n'hérite pas d'une classe parente
            # qui nécessite super().__init__()
            self._client = client

        def _with_defaults(self, kwargs):
            # Ensure follow_redirects default is False
            # for compatibility with older tests
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
def test_company(sample_company):
    """Alias d'entreprise pour les tests historiques hors dossier intégration."""
    return sample_company


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
def auth_headers(client, sample_user, sample_company):
    """Génère un token JWT valide pour l'entreprise de test."""
    from flask_jwt_extended import create_access_token

    cache_key = f"token_{sample_user.id}"
    if not hasattr(auth_headers, "_token_cache"):
        auth_headers._token_cache = {}  # type: ignore[attr-defined]
    if cache_key in auth_headers._token_cache:  # type: ignore[attr-defined]
        token = auth_headers._token_cache[cache_key]  # type: ignore[attr-defined]
        return {"Authorization": f"Bearer {token}"}

    claims = {
        "role": sample_user.role.value,
        "company_id": sample_company.id,
        "driver_id": getattr(sample_user, "driver_id", None),
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(sample_user.public_id), additional_claims=claims
        )
    auth_headers._token_cache[cache_key] = token
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
        token = create_access_token(
            identity=str(sample_admin_user.public_id), additional_claims=claims
        )
    admin_headers._token_cache[cache_key] = token
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def authenticated_client(client, sample_user):
    """Client Flask authentifié avec token JWT."""
    from datetime import timedelta

    from flask_jwt_extended import create_access_token

    claims = {
        "role": sample_user.role.value,
        "company_id": getattr(sample_user, "company_id", None),
        "driver_id": getattr(sample_user, "driver_id", None),
        "aud": "atmr-api",
    }
    with client.application.app_context():
        # ✅ FIX: Utiliser un token avec expiration longue (24h)
        # pour éviter les problèmes en tests
        # Utiliser public_id comme identity (comme dans bookings.py:588)
        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims=claims,
            expires_delta=timedelta(hours=24),  # Token valide 24h pour les tests
        )

    # Créer une classe wrapper qui ajoute automatiquement les headers
    class AuthenticatedClient:
        def __init__(self, client, token):  # pyright: ignore[reportMissingSuperCall]
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
    from datetime import UTC, datetime, timedelta

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
    booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
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
def company(db):
    """Créer une entreprise pour les tests.

    ⚠️ COUPLAGE IMPORTANT :
    - Cette fixture DOIT être commitée avant utilisation
      car `engine.run()` fait un rollback défensif
    - Les fixtures `drivers` et `bookings` dépendent de cette fixture
      (ordre d'exécution garanti par pytest)
    - L'objet est rechargé depuis la DB pour garantir qu'il est bien persisté

    🔄 ISOLATION :
    - Chaque test utilise un savepoint (nested transaction) via la fixture `db`
    - Le rollback automatique en fin de test garantit l'isolation entre les tests
    - Les objets commités dans cette fixture sont visibles dans le savepoint du test

    📝 UTILISATION :
    - Utiliser cette fixture comme dépendance pour `drivers` et `bookings`
    - Ne pas modifier l'objet retourné sans recharger depuis la DB après `engine.run()`
    """
    from models import Company
    from tests.factories import CompanyFactory

    company = CompanyFactory()
    db.session.add(company)
    db.session.flush()  # Force l'assignation de l'ID
    # ✅ FIX: Commit pour garantir persistance avant engine.run()
    # engine.run() fait un rollback défensif qui peut expirer la Company
    # si elle n'est pas commitée
    db.session.commit()
    # ✅ FIX: Expirer et recharger pour s'assurer que l'objet est bien en DB
    db.session.expire(company)
    company_id = company.id
    # Utiliser db.session.get() pour SQLAlchemy 2.0+ (comme dans test_dispatch_e2e.py)
    company = db.session.get(Company, company_id)
    assert company is not None, "Company must be persisted before use"
    return company


@pytest.fixture
def drivers(db, company=None):
    """Créer plusieurs chauffeurs pour les tests.

    ✅ DÉCOUPLAGE P2.4 :
    - Le paramètre `company` est optionnel pour réduire les couplages
    - Si `company` n'est pas fournie, une company est créée automatiquement
    - Permet d'utiliser cette fixture indépendamment ou avec une company existante

    🔄 ISOLATION :
    - Les drivers sont commités dans le savepoint du test
    - Le rollback automatique en fin de test garantit l'isolation

    📝 UTILISATION :
    - `def test_example(drivers):` - Company créée automatiquement
    - `def test_example(company, drivers):` - Company passée explicitement
    """
    from models import Company
    from tests.conftest import persisted_fixture
    from tests.factories import CompanyFactory, DriverFactory

    # ✅ P2.4: Créer company si non fournie (découplage)
    # ✅ FIX: Vérifier explicitement que company est None (pas juste falsy)
    if company is None:
        company = CompanyFactory()
        company = persisted_fixture(db, company, Company)
    else:
        # ✅ FIX: S'assurer que la company passée est bien commitée
        # et rechargée pour éviter les problèmes d'isolation
        db.session.flush()
        db.session.commit()
        # Recharger pour garantir que l'objet est bien en DB
        company_id = company.id
        db.session.expire(company)
        company = db.session.get(Company, company_id)
        assert company is not None, "Company must be persisted"

    # ✅ FIX: Vérifier que company.id est bien défini avant de créer les drivers
    assert company.id is not None, "Company ID must be set before creating drivers"

    drivers_list = [
        DriverFactory(company=company, is_active=True, is_available=True),
        DriverFactory(company=company, is_active=True, is_available=True),
        DriverFactory(company=company, is_active=True, is_available=True),
    ]
    db.session.flush()  # Force l'assignation des IDs
    # ✅ FIX: Commit pour garantir persistance
    db.session.commit()
    return drivers_list


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
        dropoff_lon=6.1500,
    )


@pytest.fixture
def simple_driver(db, sample_company):
    """Crée un driver simple avec position valide."""
    from tests.factories import create_driver_with_position

    return create_driver_with_position(
        company=sample_company, latitude=46.2044, longitude=6.1432, is_available=True
    )


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

    from models.client import Client
    from models.enums import ClientType, ManagementMode, UserRole
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
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = sample_company.id
    client.client_type = ClientType.TRANSPORT
    client.management_mode = ManagementMode.MANAGED
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
    import logging
    from unittest.mock import MagicMock

    logger = logging.getLogger(__name__)
    logger.debug("🔧 [Fixture mock_external_services] Setup: mock services externes")

    # Mock OSRM - utiliser les mêmes fonctions que mock_osrm_client
    def mock_build_distance_matrix_osrm(coords, **kwargs):
        """Retourne une matrice de durées simulée (secondes) basée sur haversine."""
        from services.geolocation.osrm import _fallback_eta_seconds

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

    def mock_route_info(origin, dest=None, destination=None, **kwargs):
        """Retourne des données de route simulées basées sur haversine."""
        from services.geolocation.osrm import _fallback_eta_seconds, _haversine_km

        # ✅ FIX: Accepter 'destination' (nom réel) ou 'dest' (nom du mock)
        dest_coord = destination if destination is not None else dest
        if dest_coord is None:
            msg = (
                "mock_route_info() missing 1 required positional argument: "
                "'dest' or 'destination'"
            )
            raise TypeError(msg)

        km = _haversine_km(origin, dest_coord)
        duration_s = _fallback_eta_seconds(origin, dest_coord)

        return {
            "duration": float(duration_s),
            "distance": int(km * 1000),  # mètres
            "geometry": {
                "type": "LineString",
                "coordinates": [[origin[1], origin[0]], [dest_coord[1], dest_coord[0]]],
            },
            "legs": [{"distance": int(km * 1000), "duration": float(duration_s)}],
            "fallback": False,  # Simuler un appel OSRM réussi
        }

    def mock_get_distance_time(origin, dest, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.geolocation.osrm import _fallback_eta_seconds, _haversine_km

        km = _haversine_km(origin, dest)
        duration_s = _fallback_eta_seconds(origin, dest)
        return (km * 1000, duration_s)  # mètres, secondes

    def mock_get_matrix(origins, destinations, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.geolocation.osrm import _fallback_eta_seconds, _haversine_km

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
        from services.geolocation.osrm import _fallback_eta_seconds

        return _fallback_eta_seconds(origin, dest)

    # Patcher OSRM (services.geolocation.osrm, ex-osrm_client)
    from services.geolocation import osrm as osrm_client

    monkeypatch.setattr(
        osrm_client, "build_distance_matrix_osrm", mock_build_distance_matrix_osrm
    )
    monkeypatch.setattr(osrm_client, "route_info", mock_route_info)
    monkeypatch.setattr(osrm_client, "get_distance_time", mock_get_distance_time)
    monkeypatch.setattr(osrm_client, "get_matrix", mock_get_matrix)
    monkeypatch.setattr(osrm_client, "eta_seconds", mock_eta_seconds)

    # Mock Redis - créer un mock Redis centralisé
    # Redis mock stateful (refresh tokens Lot 1 / fail-closed)
    _redis_kv: dict[str, object] = {}
    _redis_zsets: dict[str, dict[str, float]] = {}

    mock_redis = MagicMock()

    def _redis_get(key, *args, **kwargs):
        return _redis_kv.get(key)

    def _redis_setex(key, _ttl, value):
        _redis_kv[key] = value
        return True

    def _redis_set(key, value, *args, **kwargs):
        _redis_kv[key] = value
        return True

    def _redis_delete(*keys):
        n = 0
        for key in keys:
            if key in _redis_kv:
                del _redis_kv[key]
                n += 1
            if key in _redis_zsets:
                del _redis_zsets[key]
                n += 1
        return n

    def _redis_exists(key, *args, **kwargs):
        return 1 if key in _redis_kv else 0

    def _redis_zadd(key, mapping):
        z = _redis_zsets.setdefault(key, {})
        z.update({str(k): float(v) for k, v in mapping.items()})
        return len(mapping)

    def _redis_zcard(key):
        return len(_redis_zsets.get(key, {}))

    def _redis_zrange(key, start, end):
        items = sorted(_redis_zsets.get(key, {}).items(), key=lambda kv: kv[1])
        if end == -1:
            end = len(items) - 1
        return [k for k, _ in items[start : end + 1]]

    def _redis_zscore(key, member):
        return _redis_zsets.get(key, {}).get(str(member))

    def _redis_zrem(key, *members):
        z = _redis_zsets.get(key, {})
        n = 0
        for m in members:
            if str(m) in z:
                del z[str(m)]
                n += 1
        return n

    mock_redis.get.side_effect = _redis_get
    mock_redis.set.side_effect = _redis_set
    mock_redis.setex.side_effect = _redis_setex
    mock_redis.delete.side_effect = _redis_delete
    mock_redis.exists.side_effect = _redis_exists
    mock_redis.lpush.return_value = 1
    mock_redis.lrange.return_value = []
    mock_redis.ltrim.return_value = True
    mock_redis.expire.return_value = True
    mock_redis.keys.return_value = []
    mock_redis.ping.return_value = True
    mock_redis.zcard.side_effect = _redis_zcard
    mock_redis.zadd.side_effect = _redis_zadd
    mock_redis.zrange.side_effect = _redis_zrange
    mock_redis.zscore.side_effect = _redis_zscore
    mock_redis.zrem.side_effect = _redis_zrem

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

        monkeypatch.setattr(
            redis_client, "RedisClient", MagicMock(return_value=mock_redis)
        )
    except ImportError:
        pass

    # Retourner un dictionnaire avec les mocks pour permettre
    # l'accès si nécessaire
    # Note: Utilisation de return au lieu de yield
    # car il n'y a pas de teardown nécessaire
    # Les mocks sont déjà appliqués via monkeypatch,
    # donc ils sont actifs pour tous les tests
    logger.debug("🔧 [Fixture mock_external_services] Setup terminé")
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

    ✅ FIX: Mock les fonctions réelles utilisées
    (build_distance_matrix_osrm, route_info)
    au lieu de fonctions qui n'existent pas.
    """

    def mock_build_distance_matrix_osrm(coords, **kwargs):
        """Retourne une matrice de durées simulée (secondes) basée sur haversine."""
        from services.geolocation.osrm import _fallback_eta_seconds

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

    def mock_route_info(origin, dest=None, destination=None, **kwargs):
        """Retourne des données de route simulées basées sur haversine."""
        from services.geolocation.osrm import _fallback_eta_seconds, _haversine_km

        # ✅ FIX: Accepter 'destination' (nom réel) ou 'dest' (nom du mock)
        dest_coord = destination if destination is not None else dest
        if dest_coord is None:
            msg = (
                "mock_route_info() missing 1 required positional argument: "
                "'dest' or 'destination'"
            )
            raise TypeError(msg)

        km = _haversine_km(origin, dest_coord)
        duration_s = _fallback_eta_seconds(origin, dest_coord)

        return {
            "duration": float(duration_s),
            "distance": int(km * 1000),  # mètres
            "geometry": {
                "type": "LineString",
                "coordinates": [[origin[1], origin[0]], [dest_coord[1], dest_coord[0]]],
            },
            "legs": [{"distance": int(km * 1000), "duration": float(duration_s)}],
        }

    def mock_get_distance_time(origin, dest, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.geolocation.osrm import _fallback_eta_seconds, _haversine_km

        km = _haversine_km(origin, dest)
        duration_s = _fallback_eta_seconds(origin, dest)
        return (km * 1000, duration_s)  # mètres, secondes

    def mock_get_matrix(origins, destinations, **kwargs):
        """Mock pour compatibilité avec anciens tests."""
        from services.geolocation.osrm import _fallback_eta_seconds, _haversine_km

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
        from services.geolocation.osrm import _fallback_eta_seconds

        return _fallback_eta_seconds(origin, dest)

    from services.geolocation import osrm as osrm_client

    # ✅ FIX: Mock les fonctions réelles utilisées
    monkeypatch.setattr(
        osrm_client, "build_distance_matrix_osrm", mock_build_distance_matrix_osrm
    )
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
            from services.unified_dispatch.ml_predictor import (  # pyright: ignore[reportMissingImports]
                DelayPrediction,
            )

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
        from services.security.safety import SafetyGuards

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
    os.environ.setdefault(
        "APP_ENCRYPTION_KEY_B64", "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY"
    )
    return True


# ✅ FIX 6.2: Helpers pour gérer les transactions dans les tests
# Réduit les couplages dangereux entre fixtures et engine.run()
from contextlib import contextmanager, suppress  # noqa: E402
from typing import Any, Iterator, Type, TypeVar  # noqa: E402

T = TypeVar("T")


def persisted_fixture(
    db_session: Any,
    factory_instance: Any,
    model_class: Type[T],
    *,
    reload: bool = True,
    assert_exists: bool = True,
) -> T:
    """Helper générique pour créer des fixtures persistées.

    Crée un objet via une factory, le commit dans la DB, et le recharge pour garantir
    la persistance. Utile pour créer des fixtures qui doivent être visibles après
    le rollback défensif de `engine.run()`.

    📝 UTILISATION :
    ```python
    @pytest.fixture
    def my_entity(db):
        return persisted_fixture(db, MyEntityFactory(), MyEntity)

    @pytest.fixture
    def my_entity_with_params(db, company):
        factory = MyEntityFactory(company=company)
        return persisted_fixture(db, factory, MyEntity)
    ```

    Args:
        db_session: Session SQLAlchemy (généralement la fixture `db`)
        factory_instance: Instance de factory (ex: `CompanyFactory()`)
        model_class: Classe du modèle SQLAlchemy (ex: `Company`)
        reload: Si True, expire et recharge l'objet depuis la DB
        assert_exists: Si True, vérifie que l'objet existe après reload

    Returns:
        Instance du modèle persisté et rechargé depuis la DB
    """
    # Ajouter l'objet à la session
    # ✅ FIX: db_session est l'instance Flask-SQLAlchemy, utiliser .session
    db_session.session.add(factory_instance)
    db_session.session.flush()  # Force l'assignation de l'ID

    # Commit pour garantir la persistance
    db_session.session.commit()

    if reload:
        # Expirer et recharger pour s'assurer que l'objet est bien en DB
        db_session.session.expire(factory_instance)
        reloaded = db_session.session.query(model_class).get(factory_instance.id)

        if assert_exists:
            assert reloaded is not None, (
                f"{model_class.__name__} must be persisted before use "
                f"(id={factory_instance.id})"
            )

        return reloaded if reloaded is not None else factory_instance

    return factory_instance


@contextmanager
def ensure_committed(db_session: Any) -> Iterator[None]:
    """Context manager pour garantir que les objets sont commités avant utilisation.

    ⚠️ PROBLÈME RÉSOLU :
    - `engine.run()` fait un rollback défensif qui peut expirer
      les objets non commités
    - Ce helper garantit que tous les objets en attente
      sont commités avant utilisation

    📝 UTILISATION :
    ```python
    def test_dispatch(db, company, drivers, bookings):
        # Les fixtures garantissent déjà le commit,
        # mais on peut forcer un commit explicite
        with ensure_committed(db):
            # Tous les objets sont garantis commités ici
            result = engine.run(company_id=company.id, ...)
    ```

    🔄 ISOLATION :
    - Utilise le savepoint du test (nested transaction)
    - Le rollback automatique en fin de test garantit l'isolation
    - Les objets commités restent visibles dans le savepoint

    Args:
        db_session: Session SQLAlchemy (généralement la fixture `db`)

    Yields:
        None (context manager)
    """
    # Flush pour s'assurer que tous les objets en attente sont visibles
    db_session.flush()
    # Commit pour garantir la persistance (dans le savepoint du test)
    db_session.commit()
    try:
        yield
    finally:
        # Optionnel: on pourrait faire un flush ici si nécessaire
        # Mais le rollback automatique en fin de test gère le nettoyage
        pass


@contextmanager
def nested_savepoint(db_session: Any) -> Iterator[None]:
    """Context manager pour créer un savepoint imbriqué (nested transaction).

    Permet de créer des savepoints multiples pour isoler des parties de code
    dans un test. Le rollback du savepoint parent rollback automatiquement
    tous les savepoints enfants.

    📝 UTILISATION :
    ```python
    def test_nested_transaction(db):
        # Créer des objets dans le savepoint principal
        obj1 = MyEntityFactory()
        db.session.add(obj1)
        db.session.commit()

        # Créer un savepoint imbriqué
        with nested_savepoint(db):
            obj2 = MyEntityFactory()
            db.session.add(obj2)
            db.session.commit()
            # obj2 sera rollback à la fin du context manager
            # (ligne trop longue corrigée)

        # obj1 existe toujours, obj2 a été rollback
        assert obj1.id is not None
    ```

    ⚠️ ATTENTION :
    - Les savepoints imbriqués sont rollback automatiquement
      si le savepoint parent est rollback
    - Ne pas utiliser pour isoler des tests (utiliser la fixture `db` à la place)
    - Utile pour tester des scénarios de rollback partiel dans un même test

    Args:
        db_session: Session SQLAlchemy (généralement la fixture `db`)

    Yields:
        None (context manager)
    """
    # Créer un savepoint imbriqué
    savepoint = db_session.begin_nested()
    try:
        yield
    except Exception:
        # Rollback le savepoint en cas d'exception
        # ✅ FIX: Vérifier si le savepoint est encore actif avant de rollback
        # (engine.run() peut fermer la transaction principale,
        # ce qui ferme aussi les savepoints)
        if savepoint.is_active:
            with suppress(Exception):
                savepoint.rollback()
        raise
    finally:
        # Rollback automatique du savepoint à la fin
        # ✅ FIX: Vérifier si le savepoint est encore actif avant de rollback
        # (engine.run() peut fermer la transaction principale,
        # ce qui ferme aussi les savepoints)
        if savepoint.is_active:
            with suppress(Exception):
                savepoint.rollback()


# ============================================================
# Fixtures pour mock du temps (tests déterministes)
# ============================================================


@pytest.fixture
def frozen_time():
    """Fixture pour mocker le temps avec freezegun.

    Utilise une date fixe par défaut (2025-01-15 10:00:00) pour rendre
    les tests déterministes. Les tests peuvent utiliser cette fixture
    pour contrôler le temps.

    Exemple d'utilisation:
        def test_something(frozen_time):
            # Le temps est maintenant figé à 2025-01-15 10:00:00
            from datetime import timedelta
            from shared.time_utils import now_local
            assert now_local().year == 2025

            # Avancer le temps
            frozen_time.tick(timedelta(hours=1))
            assert now_local().hour == 11

    Note: Cette fixture utilise freezegun qui mock datetime.now(),
    datetime.utcnow(), et time.time() automatiquement.
    """
    from datetime import UTC, datetime

    try:
        from freezegun import freeze_time  # type: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("freezegun not installed")
        return  # type: ignore[unreachable]

    # Date fixe par défaut pour tests déterministes.
    # now_local() convertit depuis datetime.now(UTC) vers Europe/Zurich puis
    # retire tzinfo.
    # Pour obtenir 10:00 "local" (Zurich) en hiver, on fige donc 09:00 en UTC.
    FIXED_DATE = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)

    with freeze_time(FIXED_DATE) as frozen:
        yield frozen


@pytest.fixture
def mock_now_local(monkeypatch):
    """Fixture pour mocker now_local() directement.

    Alternative à frozen_time si on veut seulement mocker now_local()
    sans affecter datetime.now().

    Exemple d'utilisation:
        def test_something(mock_now_local):
            from shared.time_utils import now_local
            # now_local() retourne maintenant une date fixe
            assert now_local().year == 2025
    """
    from datetime import datetime

    FIXED_DATE = datetime(2025, 1, 15, 10, 0, 0)

    def mock_now():
        return FIXED_DATE

    monkeypatch.setattr("shared.time_utils.now_local", mock_now)
    return FIXED_DATE


@pytest.fixture
def requires_postgresql(db):
    """Skip si le dialecte n'est pas PostgreSQL (tests finalize / expire réalistes)."""
    try:
        bind = db.session.get_bind()
        dialect = bind.dialect.name if bind else "unknown"
    except Exception:
        dialect = "unknown"
    if dialect != "postgresql":
        pytest.skip(
            f"PostgreSQL requis (dialecte actuel: {dialect}). "
            + "Démarrer postgres_test ou définir DATABASE_URL vers une instance Postgres."
        )
    try:
        from sqlalchemy import text

        db.session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL injoignable pour ce test: {exc}")
