"""Fixtures pytest spécifiques aux tests E2E.

Ce module fournit des fixtures spécialisées pour les tests end-to-end,
encapsulant la création d'entités de test et de clients Flask authentifiés
pour différents rôles utilisateur.
"""

from typing import Any  # noqa: I001

import pytest
from flask import Flask
from flask.testing import FlaskClient

from models import Client, Company, Driver, User
from tests.e2e.helpers.e2e_helpers import (
    create_authenticated_client,
    create_test_client,
    create_test_company,
    create_test_driver,
)


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
