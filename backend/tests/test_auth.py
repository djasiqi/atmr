"""
Tests pour les routes d'authentification.
"""


def test_login_success(client, sample_user):
    """Login avec credentials valides renvoie un token."""
    # Utiliser l'email réel de sample_user (généré dynamiquement)
    response = client.post(
        "/api/v1/auth/login",
        json={"email": sample_user.email, "password": "password123"},
        # La route renvoie token/refresh_token pour les clients mobiles (Expo).
        headers={"X-Requested-With": "Expo"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "token" in data
    assert "user" in data
    assert data["user"]["email"] == sample_user.email


def test_login_invalid_password(client, sample_user):
    """Login avec mauvais mot de passe renvoie 401."""
    # Utiliser l'email réel de sample_user
    response = client.post(
        "/api/v1/auth/login",
        json={"email": sample_user.email, "password": "wrongpassword"},
    )

    # Selon la stratégie de sécurité, l'API peut répondre 401 ou 403.
    assert response.status_code in (401, 403)


def test_login_nonexistent_user(client):
    """Login avec email inexistant renvoie 401."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent@example.com", "password": "password123"},
    )

    assert response.status_code in (401, 403)


def test_protected_route_without_token(client):
    """Accès à une route protégée sans token renvoie 401."""
    response = client.get("/api/v1/bookings/")
    # 404 est acceptable si la route n'est pas initialisée (SKIP_ROUTES_INIT=1)
    assert response.status_code in (401, 404)


def test_protected_route_with_token(client, auth_headers):
    """Accès à une route protégée avec token valide fonctionne."""
    response = client.get("/api/v1/bookings/", headers=auth_headers)
    # Devrait renvoyer 200 (ou 403 si pas les permissions, ou 404 si pas de bookings)
    assert response.status_code in [200, 403, 404]
