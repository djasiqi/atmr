"""
Tests pour la rotation et la révocation des refresh tokens (Phase 2).

Ces tests vérifient que :
1. Le refresh token est roté lors d'un refresh
2. L'ancien token est révoqué après rotation
3. Tous les tokens sont révoqués au logout
4. Le nombre de tokens actifs est limité
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

from services.refresh_token_service import RefreshTokenService


class FakeRedis:
    """Mock Redis simple qui simule les opérations nécessaires pour RefreshTokenService."""

    def __init__(self):
        self._data: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}
        self._ttl: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        self._data[key] = value
        self._ttl[key] = ttl
        return True

    def exists(self, key: str) -> bool:
        return key in self._data

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
            if key in self._sets:
                del self._sets[key]
        return count

    def sadd(self, key: str, *values: str) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        count = 0
        for value in values:
            if value not in self._sets[key]:
                self._sets[key].add(value)
                count += 1
        return count

    def smembers(self, key: str) -> set[str]:
        return self._sets.get(key, set())

    def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    def srem(self, key: str, *values: str) -> int:
        if key not in self._sets:
            return 0
        count = 0
        for value in values:
            if value in self._sets[key]:
                self._sets[key].remove(value)
                count += 1
        return count

    def srandmember(self, key: str, count: int = 1) -> list[str] | str | None:
        if key not in self._sets or not self._sets[key]:
            return None if count == 1 else []
        members = list(self._sets[key])
        if count == 1:
            return members[0] if members else None
        return members[:count]

    def expire(self, key: str, ttl: int) -> bool:
        if key in self._data or key in self._sets:
            self._ttl[key] = ttl
            return True
        return False


@pytest.fixture(autouse=True)
def mock_redis(monkeypatch, app):
    """Fixture qui remplace le mock Redis global par un mock plus complet."""
    fake_redis = FakeRedis()

    # Patcher redis.from_url pour retourner notre fake Redis
    try:
        import redis

        def from_url_mock(*args, **kwargs):
            return fake_redis

        monkeypatch.setattr(redis, "from_url", from_url_mock)

        # Forcer toutes les instances de RefreshTokenService à utiliser le nouveau mock
        # en remplaçant leur client Redis directement
        def refresh_token_service_init_patch(original_init):
            def patched_init(self):
                original_init(self)
                self.redis_client = fake_redis

            return patched_init

        # Patcher __init__ de RefreshTokenService
        from services.refresh_token_service import RefreshTokenService

        monkeypatch.setattr(
            RefreshTokenService,
            "__init__",
            refresh_token_service_init_patch(RefreshTokenService.__init__),
        )
    except ImportError:
        pass

    yield fake_redis

    # Nettoyer après le test
    fake_redis._data.clear()
    fake_redis._sets.clear()
    fake_redis._ttl.clear()


@pytest.fixture
def token_service(app: Flask, mock_redis) -> RefreshTokenService:
    """Fixture pour créer une instance de RefreshTokenService."""
    with app.app_context():
        return RefreshTokenService()


def test_token_rotation_on_refresh(
    client: FlaskClient,
    auth_headers: dict[str, str],
    sample_user,
) -> None:
    """Test que le refresh token est roté lors d'un refresh."""
    # 1. Login
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user.email,
            "password": "password123",
        },
    )
    assert login_response.status_code == 200

    # Récupérer le refresh token depuis le cookie ou la réponse JSON
    old_refresh_token = None
    if login_response.headers.getlist("Set-Cookie"):
        for cookie in login_response.headers.getlist("Set-Cookie"):
            if "refresh_token=" in cookie:
                # Extraire la valeur du cookie
                cookie_parts = cookie.split(";")[0]
                old_refresh_token = cookie_parts.split("=", 1)[1]

    # Si pas dans le cookie, chercher dans la réponse JSON (mobile)
    if not old_refresh_token and login_response.is_json:
        data = login_response.get_json()
        if data and "refresh_token" in data:
            old_refresh_token = data["refresh_token"]

    assert old_refresh_token is not None, "Refresh token non trouvé après login"

    # 2. Refresh - utiliser set_cookie pour envoyer le cookie
    client.set_cookie("refresh_token", old_refresh_token)
    refresh_response = client.post(
        "/api/v1/auth/refresh-token",
        headers={"Content-Type": "application/json"},
    )

    if refresh_response.status_code != 200:
        error_data = refresh_response.get_json()
        error_msg = f"Erreur {refresh_response.status_code}: {error_data}"
        pytest.fail(error_msg)

    assert refresh_response.status_code == 200

    # Récupérer le nouveau refresh token
    new_refresh_token = None
    if refresh_response.headers.getlist("Set-Cookie"):
        for cookie in refresh_response.headers.getlist("Set-Cookie"):
            if "refresh_token=" in cookie:
                cookie_parts = cookie.split(";")[0]
                new_refresh_token = cookie_parts.split("=", 1)[1]

    if not new_refresh_token and refresh_response.is_json:
        data = refresh_response.get_json()
        if data and "refresh_token" in data:
            new_refresh_token = data["refresh_token"]

    assert new_refresh_token is not None, "Nouveau refresh token non trouvé"

    # 3. Vérifier que le nouveau token est différent
    assert new_refresh_token != old_refresh_token, (
        "Le nouveau token doit être différent"
    )

    # 4. Vérifier que l'ancien token est révoqué (via RefreshTokenService)
    with client.application.app_context():
        from models import User

        token_service = RefreshTokenService()
        # Récupérer l'ID utilisateur depuis la DB
        user = User.query.filter_by(email=sample_user.email).first()
        assert user is not None, "L'utilisateur devrait exister"

        assert not token_service.is_token_valid(old_refresh_token, user.id), (
            "L'ancien token devrait être révoqué"
        )

        # 5. Vérifier que le nouveau token est valide
        assert token_service.is_token_valid(new_refresh_token, user.id), (
            "Le nouveau token devrait être valide"
        )


def test_revoke_all_tokens_on_logout(
    client: FlaskClient,
    auth_headers: dict[str, str],
    sample_user,
) -> None:
    """Test que tous les tokens sont révoqués au logout."""
    # 1. Login plusieurs fois (créer plusieurs tokens)
    tokens = []
    for _ in range(3):
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": sample_user.email,
                "password": "password123",
            },
        )
        assert login_response.status_code == 200

        # Récupérer le refresh token
        refresh_token = None
        if login_response.headers.getlist("Set-Cookie"):
            for cookie in login_response.headers.getlist("Set-Cookie"):
                if "refresh_token=" in cookie:
                    cookie_parts = cookie.split(";")[0]
                    refresh_token = cookie_parts.split("=", 1)[1]

        if not refresh_token and login_response.is_json:
            data = login_response.get_json()
            if data and "refresh_token" in data:
                refresh_token = data["refresh_token"]

        if refresh_token:
            tokens.append(refresh_token)

    assert len(tokens) > 0, "Aucun token n'a été créé"

    # 2. Logout avec le dernier token (celui dans auth_headers)
    logout_response = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert logout_response.status_code == 200

    # 3. Vérifier que tous les tokens sont révoqués
    with client.application.app_context():
        from models import User

        token_service = RefreshTokenService()
        user = User.query.filter_by(email=sample_user.email).first()
        assert user is not None, "L'utilisateur devrait exister"

        for token in tokens:
            assert not token_service.is_token_valid(token, user.id), (
                f"Le token devrait être révoqué après logout: {token[:8]}..."
            )


def test_token_limit_per_user(
    client: FlaskClient,
    sample_user,
    app: Flask,
) -> None:
    """Test que le nombre de tokens actifs est limité."""
    # Configurer une limite basse pour le test
    os.environ["MAX_ACTIVE_REFRESH_TOKENS"] = "2"

    # 1. Login plusieurs fois
    tokens = []
    for _ in range(4):  # Créer 4 tokens alors que la limite est 2
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "email": sample_user.email,
                "password": "password123",
            },
        )
        assert login_response.status_code == 200

        # Récupérer le refresh token
        refresh_token = None
        if login_response.headers.getlist("Set-Cookie"):
            for cookie in login_response.headers.getlist("Set-Cookie"):
                if "refresh_token=" in cookie:
                    cookie_parts = cookie.split(";")[0]
                    refresh_token = cookie_parts.split("=", 1)[1]

        if not refresh_token and login_response.is_json:
            data = login_response.get_json()
            if data and "refresh_token" in data:
                refresh_token = data["refresh_token"]

        if refresh_token:
            tokens.append(refresh_token)

    # 2. Vérifier que seul le nombre maximum de tokens est actif
    with app.app_context():
        from models import User

        token_service = RefreshTokenService()
        user = User.query.filter_by(email=sample_user.email).first()
        assert user is not None, "L'utilisateur devrait exister"

        # Vérifier que le nombre de tokens actifs ne dépasse pas la limite
        active_count = sum(
            1 for token in tokens if token_service.is_token_valid(token, user.id)
        )
        assert active_count <= 2, "Le nombre de tokens actifs devrait être limité à 2"

        # Vérifier aussi directement via get_active_token_count
        count = token_service.get_active_token_count(user.id)
        assert count <= 2, (
            f"Le nombre de tokens actifs ({count}) devrait être limité à 2"
        )


def test_token_store_and_retrieve(
    token_service: RefreshTokenService, app: Flask
) -> None:
    """Test le stockage et la récupération de tokens."""
    with app.app_context():
        user_id = 1
        token = "test_refresh_token_123"

        # Stocker un token
        token_service.store_token(user_id, token)

        # Vérifier qu'il est valide
        assert token_service.is_token_valid(token, user_id), (
            "Le token devrait être valide"
        )

        # Vérifier le nombre de tokens actifs
        count = token_service.get_active_token_count(user_id)
        assert isinstance(count, int), "get_active_token_count devrait retourner un int"
        assert count >= 1, "Au moins un token devrait être actif"

    # Révoquer le token
    token_service.revoke_token(token)

    # Vérifier qu'il est maintenant révoqué
    assert not token_service.is_token_valid(token, user_id), (
        "Le token devrait être révoqué"
    )


def test_revoke_all_user_tokens(token_service: RefreshTokenService, app: Flask) -> None:
    """Test la révocation de tous les tokens d'un utilisateur."""
    with app.app_context():
        user_id = 1

        # Créer plusieurs tokens
        tokens = [f"token_{i}" for i in range(5)]
        for token in tokens:
            token_service.store_token(user_id, token)

        # Vérifier qu'ils sont tous actifs
        for token in tokens:
            assert token_service.is_token_valid(token, user_id), (
                f"Le token {token} devrait être actif"
            )

        # Révoquer tous les tokens
        token_service.revoke_all_user_tokens(user_id)

        # Vérifier qu'ils sont tous révoqués
        for token in tokens:
            assert not token_service.is_token_valid(token, user_id), (
                f"Le token {token} devrait être révoqué"
            )
