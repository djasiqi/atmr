"""
Tests pour les routes d'authentification.
"""
import pytest

from models import User, UserRole


def test_login_success(client, sample_user):
    """Login avec credentials valides renvoie un token."""
    response = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "password123"
    })

    assert response.status_code == 200
    data = response.get_json()
    assert "token" in data
    assert "user" in data
    assert data["user"]["email"] == "test@example.com"


def test_login_invalid_password(client, sample_user):
    """Login avec mauvais mot de passe renvoie 401."""
    response = client.post("/api/auth/login", json={
        "email": "test@example.com",
        "password": "wrongpassword"
    })

    assert response.status_code == 401


def test_login_nonexistent_user(client):
    """Login avec email inexistant renvoie 401."""
    response = client.post("/api/auth/login", json={
        "email": "nonexistent@example.com",
        "password": "password123"
    })

    assert response.status_code == 401


def test_protected_route_without_token(client):
    """Accès à une route protégée sans token renvoie 401."""
    response = client.get("/api/bookings/")
    assert response.status_code == 401


def test_protected_route_with_token(client, auth_headers):
    """Accès à une route protégée avec token valide fonctionne."""
    response = client.get("/api/bookings/", headers=auth_headers)
    # Devrait renvoyer 200 (ou 403 si pas les permissions)
    assert response.status_code in [200, 403]

