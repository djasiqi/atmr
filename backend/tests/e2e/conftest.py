"""Fixtures pytest spécifiques aux tests E2E.

Ce module fournit des fixtures spécialisées pour les tests end-to-end,
encapsulant la création d'entités de test et de clients Flask authentifiés
pour différents rôles utilisateur.

IMPORTANT: Les migrations Alembic sont appliquées automatiquement au démarrage
de la session de test via la fixture `e2e_db_migrations`.

Note: Pour que les migrations fonctionnent correctement, DISABLE_EVENTLET=1
doit être défini dans l'environnement. Voir docs/migrations.md pour plus de détails.
"""

import logging
import os
from typing import Any  # noqa: I001

import pytest
from flask import Flask
from flask.testing import FlaskClient
from sqlalchemy import text

from models import Client, Company, Driver, User
from tests.e2e.helpers.e2e_helpers import (
    create_authenticated_client,
    create_test_client,
    create_test_company,
    create_test_driver,
)

logger = logging.getLogger(__name__)


# =====================================================
# Fixture: Application automatique des migrations
# =====================================================


@pytest.fixture(scope="session", autouse=True)
def e2e_db_migrations(request):
    """Applique automatiquement les migrations Alembic au démarrage des tests E2E.

    Cette fixture session-scoped s'exécute une seule fois avant tous les tests E2E.
    Elle garantit que la base de données a le schéma à jour.

    Comportement:
    - Applique `flask db upgrade heads` via Alembic programmatique
    - Échoue explicitement si les migrations échouent (pas de skip silencieux)
    - Peut être désactivée avec SKIP_E2E_MIGRATIONS=1

    Raises:
        RuntimeError: Si les migrations échouent
    """
    # Option pour désactiver (utile si migrations déjà appliquées manuellement)
    if os.getenv("SKIP_E2E_MIGRATIONS") == "1":
        logger.info("⏭️  [E2E] SKIP_E2E_MIGRATIONS=1, migrations ignorées")
        yield
        return

    logger.info("🔄 [E2E] Application automatique des migrations Alembic...")

    try:
        # Importer l'app Flask et les extensions
        from app import create_app
        from ext import db as _db

        # Créer une instance d'app pour les migrations
        app = create_app(config_name="testing")

        # Configurer l'URL de la base de données
        database_url = (
            os.getenv("DATABASE_URL_TEST")
            or os.getenv("DATABASE_URL")
            or "postgresql://test:test@localhost:5432/atmr_test"
        )
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url

        with app.app_context():
            # Vérifier la connexion à la DB
            try:
                _db.session.execute(text("SELECT 1"))
                logger.info("✅ [E2E] Connexion DB établie: %s", database_url.split("@")[-1])
            except Exception as conn_err:
                msg = (
                    f"❌ [E2E] Impossible de se connecter à la base de données: {conn_err}\n"
                    + f"URL: {database_url}\n"
                    + "Vérifiez que PostgreSQL est démarré et accessible."
                )
                raise RuntimeError(msg) from conn_err

            # Appliquer les migrations via Alembic
            try:
                from flask_migrate import upgrade as flask_migrate_upgrade

                # Exécuter les migrations (équivalent de `flask db upgrade heads`)
                flask_migrate_upgrade(revision="heads")
                logger.info("✅ [E2E] Migrations appliquées avec succès")

            except Exception as migrate_err:
                msg = (
                    f"❌ [E2E] Échec de l'application des migrations: {migrate_err}\n"
                    + "Vérifiez:\n"
                    + "  1. DISABLE_EVENTLET=1 est défini (obligatoire pour les migrations)\n"
                    + "  2. Les fichiers de migration dans backend/migrations/versions/\n"
                    + "  3. La chaîne de migrations (DISABLE_EVENTLET=1 flask db heads)\n"
                    + "  4. Les permissions sur la base de données\n"
                    + "Voir docs/migrations.md pour plus de détails."
                )
                raise RuntimeError(msg) from migrate_err

            # Vérifier que les tables critiques existent
            _verify_critical_tables(_db)

    except ImportError as import_err:
        msg = (
            f"❌ [E2E] Import error lors de l'initialisation: {import_err}\n"
            + "Vérifiez que tous les packages sont installés."
        )
        raise RuntimeError(msg) from import_err

    yield

    logger.info("🧹 [E2E] Fin de session, nettoyage terminé")


def _verify_critical_tables(db) -> None:
    """Vérifie que les tables critiques pour les tests E2E existent.

    Raises:
        RuntimeError: Si une table critique est manquante
    """
    critical_tables = [
        "user",
        "company",
        "booking",
        # Tables institution (ÉTAPE 4-5)
        "institutions",
        "institution_patients",
        "transport_requests",
        "request_offers",
    ]

    missing_tables = []
    for table in critical_tables:
        try:
            db.session.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
        except Exception:
            missing_tables.append(table)
            db.session.rollback()

    if missing_tables:
        msg = (
            f"❌ [E2E] Tables manquantes après migrations: {missing_tables}\n"
            + "Les migrations n'ont peut-être pas été appliquées correctement.\n"
            + "Essayez: docker compose run --rm api flask db upgrade heads"
        )
        raise RuntimeError(msg)

    logger.info("✅ [E2E] Tables critiques vérifiées: %s", critical_tables)


# =====================================================
# Fixtures de base pour E2E
# =====================================================


@pytest.fixture
def e2e_company(db: Any) -> Company:
    """Company de test pour E2E.

    Returns:
        Company créée et persistée en DB

    Exemple:
        ```python
        def test_something(e2e_company):
            assert e2e_company.id is not None
        ```
    """
    return create_test_company(db)


@pytest.fixture
def e2e_client_user(db_session: Any, e2e_company: Company) -> tuple[Client, User]:
    """Client et User de test pour E2E.

    Args:
        db_session: Session DB (fixture pytest)
        e2e_company: Company de test (fixture)

    Returns:
        Tuple (Client, User) créé et persisté en DB

    Exemple:
        ```python
        def test_something(e2e_client_user):
            client, user = e2e_client_user
            assert client.user_id == user.id
        ```
    """
    client = create_test_client(db_session, company=e2e_company)
    # Le User est créé automatiquement par ClientFactory
    user = client.user
    return (client, user)


@pytest.fixture
def e2e_driver(db_session: Any, e2e_company: Company) -> Driver:
    """Chauffeur de test pour E2E.

    Args:
        db_session: Session DB (fixture pytest)
        e2e_company: Company de test (fixture)

    Returns:
        Driver créé et persisté en DB

    Exemple:
        ```python
        def test_something(e2e_driver):
            assert e2e_driver.company_id == e2e_company.id
        ```
    """
    return create_test_driver(db_session, company=e2e_company)


# =====================================================
# Fixtures de clients authentifiés pour E2E
# =====================================================


@pytest.fixture
def e2e_authenticated_company_client(
    app: Flask,
    e2e_company: Company,
) -> FlaskClient:
    """Client Flask authentifié en tant que company.

    Args:
        app: Instance Flask (fixture pytest)
        e2e_company: Company de test (fixture)

    Returns:
        FlaskClient avec authentification company configurée

    Exemple:
        ```python
        def test_company_endpoint(e2e_authenticated_company_client):
            response = e2e_authenticated_company_client.get("/api/v1/company/me")
            assert response.status_code == 200
        ```
    """
    # La company a un user associé (créé par CompanyFactory)
    user = e2e_company.user
    return create_authenticated_client(app, user)


@pytest.fixture
def e2e_authenticated_client_user(
    app: Flask,
    e2e_client_user: tuple[Client, User],
) -> FlaskClient:
    """Client Flask authentifié en tant que client (utilisateur final).

    Args:
        app: Instance Flask (fixture pytest)
        e2e_client_user: Tuple (Client, User) de test (fixture)

    Returns:
        FlaskClient avec authentification client configurée

    Exemple:
        ```python
        def test_client_endpoint(e2e_authenticated_client_user, e2e_client_user):
            client, _ = e2e_client_user
            response = e2e_authenticated_client_user.get(
                f"/api/v1/clients/{client.public_id}"
            )
            assert response.status_code == 200
        ```
    """
    _client, user = e2e_client_user
    return create_authenticated_client(app, user)


# =====================================================
# Fixtures utilitaires pour E2E
# =====================================================


@pytest.fixture
def e2e_client(app: Flask) -> FlaskClient:
    """Client Flask non authentifié pour tests E2E.

    Args:
        app: Instance Flask (fixture pytest)

    Returns:
        FlaskClient standard (non authentifié)

    Exemple:
        ```python
        def test_login(e2e_client):
            response = e2e_client.post(
                "/api/v1/auth/login",
                json={"email": "user@example.com", "password": "password123"},
            )
            assert response.status_code == 200
        ```
    """
    return app.test_client()
