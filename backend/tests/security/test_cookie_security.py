"""Tests de sécurité pour la validation des cookies en production.

Ces tests vérifient que les paramètres de sécurité des cookies sont
correctement validés en production.
"""

import pytest
from flask import Flask  # pyright: ignore[reportMissingImports]

from config import (
    DevelopmentConfig,
    ProductionConfig,
    validate_production_security,
)


def test_cookie_secure_enforced_in_production():
    """Test que COOKIE_SECURE est True en production."""
    app = Flask(__name__)
    app.config.from_object(ProductionConfig)
    app.config["FLASK_ENV"] = "production"

    # Tenter de définir COOKIE_SECURE=False devrait échouer
    app.config["COOKIE_SECURE"] = False

    with pytest.raises(RuntimeError, match="COOKIE_SECURE"):
        validate_production_security(app)


def test_cookie_httponly_enforced_in_production():
    """Test que COOKIE_HTTP_ONLY est True en production."""
    app = Flask(__name__)
    app.config.from_object(ProductionConfig)
    app.config["FLASK_ENV"] = "production"
    app.config["COOKIE_HTTP_ONLY"] = False

    with pytest.raises(RuntimeError, match="COOKIE_HTTP_ONLY"):
        validate_production_security(app)


def test_cookie_samesite_valid_in_production():
    """Test que COOKIE_SAME_SITE est Strict ou Lax en production."""
    app = Flask(__name__)
    app.config.from_object(ProductionConfig)
    app.config["FLASK_ENV"] = "production"
    app.config["COOKIE_SAME_SITE"] = "None"

    with pytest.raises(RuntimeError, match="COOKIE_SAME_SITE"):
        validate_production_security(app)


def test_cookie_samesite_none_requires_secure():
    """Test que COOKIE_SAME_SITE=None est rejeté même avec Secure=True."""
    app = Flask(__name__)
    app.config.from_object(ProductionConfig)
    app.config["FLASK_ENV"] = "production"
    app.config["COOKIE_SECURE"] = True
    app.config["COOKIE_SAME_SITE"] = "None"

    with pytest.raises(RuntimeError, match="COOKIE_SAME_SITE"):
        validate_production_security(app)


def test_production_config_has_secure_cookies():
    """Test que la configuration de production a les bons paramètres par défaut."""
    config = ProductionConfig()

    assert config.COOKIE_SECURE is True
    assert config.COOKIE_HTTP_ONLY is True
    assert config.COOKIE_SAME_SITE in ["Strict", "Lax"]


def test_development_config_allows_insecure():
    """Test que le développement permet Secure=False."""
    config = DevelopmentConfig()

    # En dev, Secure=False est acceptable pour localhost HTTP
    assert config.COOKIE_SECURE is False
    assert config.COOKIE_SAME_SITE == "Lax"


def test_validation_passes_with_correct_production_config():
    """Test que la validation passe avec une configuration production correcte."""
    app = Flask(__name__)
    app.config.from_object(ProductionConfig)
    app.config["FLASK_ENV"] = "production"
    app.config["COOKIE_SECURE"] = True
    app.config["COOKIE_HTTP_ONLY"] = True
    app.config["COOKIE_SAME_SITE"] = "Strict"

    # Ne doit pas lever d'exception
    validate_production_security(app)


def test_validation_passes_with_lax_samesite():
    """Test que SameSite=Lax est accepté en production."""
    app = Flask(__name__)
    app.config.from_object(ProductionConfig)
    app.config["FLASK_ENV"] = "production"
    app.config["COOKIE_SECURE"] = True
    app.config["COOKIE_HTTP_ONLY"] = True
    app.config["COOKIE_SAME_SITE"] = "Lax"

    # Ne doit pas lever d'exception
    validate_production_security(app)


def test_validation_ignores_non_production():
    """Test que la validation est ignorée en développement."""
    app = Flask(__name__)
    app.config.from_object(DevelopmentConfig)
    app.config["FLASK_ENV"] = "development"
    app.config["COOKIE_SECURE"] = False  # Acceptable en dev
    app.config["COOKIE_HTTP_ONLY"] = False  # Même si dangereux, pas validé en dev

    # Ne doit pas lever d'exception car pas en production
    validate_production_security(app)


def test_validation_ignores_testing():
    """Test que la validation est ignorée en mode testing."""
    app = Flask(__name__)
    app.config.from_object(DevelopmentConfig)
    app.config["FLASK_ENV"] = "testing"
    app.config["COOKIE_SECURE"] = False

    # Ne doit pas lever d'exception
    validate_production_security(app)


# ========================
# Phase 2.2: Tests d'intégration pour vérifier les cookies
# ========================


def test_login_sets_httponly_cookies(
    client,  # noqa: ANN001
    sample_user,  # noqa: ANN001
) -> None:
    """Test que les cookies de login ont le flag HttpOnly."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user.email,
            "password": "password123",  # Utiliser le mot de passe par défaut
        },
    )

    assert response.status_code == 200

    set_cookie_headers = response.headers.getlist("Set-Cookie")
    combined_cookies = ", ".join(set_cookie_headers)

    assert "access_token=" in combined_cookies or "refresh_token=" in combined_cookies
    assert "HttpOnly" in combined_cookies

    # Vérifier que chaque cookie sensible a HttpOnly
    for cookie_header in set_cookie_headers:
        if "access_token=" in cookie_header or "refresh_token=" in cookie_header:
            assert "HttpOnly" in cookie_header, (
                f"Cookie devrait avoir HttpOnly: {cookie_header}"
            )


def test_cookies_have_secure_flag_in_production(
    app: Flask,  # noqa: ANN001
    client,  # noqa: ANN001
    sample_user,  # noqa: ANN001
) -> None:
    """Test que les cookies ont Secure=True en production."""
    app.config["FLASK_ENV"] = "production"
    app.config["COOKIE_SECURE"] = True

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user.email,
            "password": "password123",
        },
    )

    assert response.status_code == 200

    set_cookie_headers = response.headers.getlist("Set-Cookie")
    combined_cookies = ", ".join(set_cookie_headers)

    assert "Secure" in combined_cookies

    # Vérifier que chaque cookie sensible a Secure
    for cookie_header in set_cookie_headers:
        if "access_token=" in cookie_header or "refresh_token=" in cookie_header:
            assert "Secure" in cookie_header, (
                f"Cookie devrait avoir Secure en production: {cookie_header}"
            )


def test_cookies_have_samesite_flag(
    client,  # noqa: ANN001
    sample_user,  # noqa: ANN001
) -> None:
    """Test que les cookies ont le flag SameSite."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user.email,
            "password": "password123",
        },
    )

    assert response.status_code == 200

    set_cookie_headers = response.headers.getlist("Set-Cookie")
    combined_cookies = ", ".join(set_cookie_headers)

    assert "SameSite" in combined_cookies
    assert "SameSite=Strict" in combined_cookies or "SameSite=Lax" in combined_cookies

    # Vérifier que chaque cookie sensible a SameSite
    for cookie_header in set_cookie_headers:
        if "access_token=" in cookie_header or "refresh_token=" in cookie_header:
            assert "SameSite=" in cookie_header, (
                f"Cookie devrait avoir SameSite: {cookie_header}"
            )


def test_cookies_have_max_age(
    client,  # noqa: ANN001
    sample_user,  # noqa: ANN001
) -> None:
    """Test que les cookies ont une expiration définie."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user.email,
            "password": "password123",
        },
    )

    assert response.status_code == 200

    set_cookie_headers = response.headers.getlist("Set-Cookie")
    combined_cookies = ", ".join(set_cookie_headers)

    # Vérifier que Max-Age ou expires est présent
    assert "Max-Age" in combined_cookies or "expires" in combined_cookies.lower()

    # Vérifier que chaque cookie sensible a une expiration
    for cookie_header in set_cookie_headers:
        if "access_token=" in cookie_header or "refresh_token=" in cookie_header:
            assert "Max-Age" in cookie_header or "expires" in cookie_header.lower(), (
                f"Cookie devrait avoir une expiration: {cookie_header}"
            )


def test_cookies_not_accessible_via_javascript(
    client,  # noqa: ANN001
    sample_user,  # noqa: ANN001
) -> None:
    """Test que les cookies ne peuvent pas être lus depuis JavaScript (HttpOnly)."""
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user.email,
            "password": "password123",
        },
    )

    assert response.status_code == 200

    # Vérifier que HttpOnly est présent pour les cookies sensibles
    set_cookie_headers = response.headers.getlist("Set-Cookie")
    for cookie_header in set_cookie_headers:
        if "access_token" in cookie_header or "refresh_token" in cookie_header:
            assert "HttpOnly" in cookie_header, (
                "Les cookies de tokens devraient avoir HttpOnly "
                "pour empêcher l'accès JavaScript"
            )
