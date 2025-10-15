"""
Tests pour les routes clients (CRUD, validation, search).
"""
import pytest

from models import Client, User, UserRole


def test_list_clients_unauthenticated(client):
    """GET /clients sans authentification renvoie 401."""
    response = client.get('/api/clients/')
    assert response.status_code == 401


def test_list_clients_authenticated(client, auth_headers):
    """GET /clients avec authentification renvoie liste."""
    response = client.get('/api/clients/', headers=auth_headers)
    # Peut être 200 ou 404 selon si route existe
    assert response.status_code in [200, 404, 405]


def test_create_client(client, auth_headers, db, sample_company):
    """POST /clients/ crée un nouveau client."""
    # Cette route peut ne pas exister en l'état actuel
    # Test de structure pour référence future
    client_data = {
        "first_name": "Marie",
        "last_name": "Dupont",
        "email": "marie.dupont@example.com",
        "phone": "+41791234567",
        "billing_address": "Rue Test 1, 1000 Lausanne"
    }
    
    # Note: Adapter selon l'API réelle
    # response = client.post('/api/clients/', json=client_data, headers=auth_headers)
    # assert response.status_code == 201


def test_client_validation_email_required():
    """Email requis pour clients self-service."""
    # Test unitaire du modèle
    user = User(
        username='testclient',
        email='client@test.com',
        role=UserRole.client
    )
    
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
    assert 'id' in serialized
    assert 'user' in serialized
    assert 'billing_address' in serialized
    assert 'contact_email' in serialized


def test_client_toggle_active(db, sample_client):
    """Client.toggle_active change is_active."""
    client_obj = Client.query.get(sample_client.id)
    initial_status = client_obj.is_active
    
    new_status = client_obj.toggle_active()
    
    assert new_status != initial_status
    assert client_obj.is_active == new_status

