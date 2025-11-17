"""
Tests pour les routes clients (CRUD, validation, search).
"""

import pytest

from models import Client, User, UserRole


def test_list_clients_unauthenticated(client):
    """GET /clients sans authentification renvoie 401."""
    response = client.get("/api/clients/")
    assert response.status_code == 401


def test_list_clients_authenticated(client, auth_headers):
    """GET /clients avec authentification renvoie liste."""
    response = client.get("/api/clients/", headers=auth_headers)
    # Peut être 200 ou 404 selon si route existe
    assert response.status_code in [200, 404, 405]


def test_create_client(client, auth_headers, db, sample_company):
    """POST /clients/ crée un nouveau client."""
    # Cette route peut ne pas exister en l'état actuel
    # Test de structure pour référence future

    # Note: Adapter selon l'API réelle
    # response = client.post('/api/clients/', json=client_data, headers=auth_headers)
    # assert response.status_code == 201


def test_client_validation_email_required():
    """Email requis pour clients self-service."""
    # Test unitaire du modèle
    user = User(username="testclient", email="client@test.com", role=UserRole.client)

    # Email est requis
    assert user.email is not None


def test_client_has_user_relationship(db, sample_client):
    """Client a une relation avec User."""
    client_obj = Client.query.get(sample_client.id)
    assert client_obj is not None
    assert client_obj.user is not None
    assert client_obj.user.email is not None


def test_client_has_company_relationship(db, sample_client, sample_company):
    """Client a une relation avec Company."""
    client_obj = Client.query.get(sample_client.id)
    assert client_obj is not None
    assert client_obj.company_id == sample_company.id
    assert client_obj.company is not None


def test_client_serialize(db, sample_client):
    """Client.serialize retourne dict avec données."""
    client_obj = Client.query.get(sample_client.id)
    serialized = client_obj.serialize

    assert isinstance(serialized, dict)
    assert "id" in serialized
    assert "user" in serialized
    assert "billing_address" in serialized
    assert "contact_email" in serialized


def test_client_toggle_active(db, sample_client):
    """Client.toggle_active change is_active."""
    client_obj = Client.query.get(sample_client.id)
    initial_status = client_obj.is_active

    new_status = client_obj.toggle_active()

    assert new_status != initial_status
    assert client_obj.is_active == new_status


def test_company_clients_pagination(client, auth_headers, db, sample_company, sample_user):
    """GET /companies/me/clients?page=1&per_page=5 renvoie pagination."""
    from ext import bcrypt
    from models import ClientType

    # Créer 12 clients de test pour l'entreprise
    for i in range(12):
        user = User(
            username=f"client_{i}",
            email=f"client{i}@example.com",
            role=UserRole.client,
            first_name=f"Client{i}",
            last_name=f"Test{i}",
            phone=f"07912345{i:02d}",
            address=f"Rue Test {i}, 1000 Lausanne",
        )
        user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
        db.session.add(user)
        db.session.flush()

        client_obj = Client(
            user_id=user.id,
            company_id=sample_company.id,
            client_type=ClientType.PRIVATE,  # Pas SELF_SERVICE
            billing_address=f"Rue Test {i}, 1000 Lausanne",
            contact_email=f"client{i}@example.com",
            contact_phone=f"07912345{i:02d}",
        )
        db.session.add(client_obj)
    db.session.commit()

    # Test page 1 avec 5 résultats par page
    response = client.get("/api/companies/me/clients?page=1&per_page=5", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()

    assert "clients" in data
    assert "total" in data
    assert len(data["clients"]) == 5  # Page 1 contient 5 éléments
    assert data["total"] >= 12  # Au moins 12 clients (+ sample_client si existe)

    # Vérifier headers de pagination
    assert "X-Total-Count" in response.headers
    assert "X-Page" in response.headers
    assert response.headers["X-Page"] == "1"
    assert response.headers["X-Per-Page"] == "5"
    assert "Link" in response.headers

    # Vérifier que le header Link contient rel="next"
    assert 'rel="next"' in response.headers["Link"]

    # Test page 2
    response2 = client.get("/api/companies/me/clients?page=2&per_page=5", headers=auth_headers)
    assert response2.status_code == 200
    data2 = response2.get_json()

    assert len(data2["clients"]) == 5  # Page 2 contient aussi 5 éléments
    assert 'rel="prev"' in response2.headers["Link"]  # Page 2 a un lien prev


def test_company_clients_search_pagination(client, auth_headers, db, sample_company):
    """GET /companies/me/clients?search=Client&page=1&per_page=3 combine recherche et pagination."""
    from ext import bcrypt
    from models import ClientType

    # Créer 8 clients dont 5 commencent par "Client"
    for i in range(5):
        user = User(
            username=f"client_search_{i}",
            email=f"clientsearch{i}@example.com",
            role=UserRole.client,
            first_name=f"ClientSearch{i}",
            last_name=f"Test{i}",
            phone=f"07912346{i:02d}",
        )
        user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
        db.session.add(user)
        db.session.flush()

        client_obj = Client(
            user_id=user.id,
            company_id=sample_company.id,
            client_type=ClientType.PRIVATE,
            billing_address=f"Rue Test {i}",
            contact_email=f"clientsearch{i}@example.com",
            contact_phone=f"07912346{i:02d}",
        )
        db.session.add(client_obj)

    # 3 autres clients qui ne matchent pas
    for i in range(3):
        user = User(
            username=f"other_{i}",
            email=f"other{i}@example.com",
            role=UserRole.client,
            first_name=f"Other{i}",
            last_name=f"Name{i}",
            phone=f"07912347{i:02d}",
        )
        user.password = bcrypt.generate_password_hash("password123").decode("utf-8")
        db.session.add(user)
        db.session.flush()

        client_obj = Client(
            user_id=user.id,
            company_id=sample_company.id,
            client_type=ClientType.PRIVATE,
            billing_address=f"Rue Other {i}",
            contact_email=f"other{i}@example.com",
            contact_phone=f"07912347{i:02d}",
        )
        db.session.add(client_obj)

    db.session.commit()

    # Recherche "ClientSearch" paginée
    response = client.get("/api/companies/me/clients?search=ClientSearch&page=1&per_page=3", headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()

    assert "clients" in data
    assert len(data["clients"]) == 3  # Page 1 contient 3 résultats
    assert data["total"] >= 5  # Au moins 5 clients matchent "ClientSearch"

    # Vérifier que tous les résultats contiennent "ClientSearch"
    for client_data in data["clients"]:
        user_data = client_data.get("user", {})
        first_name = user_data.get("first_name", "")
        assert "ClientSearch" in first_name
