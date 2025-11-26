"""Tests unitaires pour le décorateur IP whitelist.

Valide le fonctionnement de la restriction d'accès par IP :
- Vérification IPs autorisées
- Blocage IPs non autorisées
- Support réseaux CIDR
- Détection IP via headers proxy
- Configuration via variables d'environnement
"""

from unittest.mock import patch

import pytest
from flask import Flask

from security.ip_whitelist import ip_whitelist_required


@pytest.fixture
def fresh_app():
    """Crée une nouvelle instance Flask pour chaque test.

    Nécessaire car Flask ne permet pas d'ajouter des routes après
    qu'une requête ait été traitée.
    """
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    return app


class TestIPWhitelistAllowed:
    """Tests pour IPs autorisées."""

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_allowed_ip(self, mock_getenv, fresh_app):
        """Test IP autorisée (accès accordé)."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist",
            environ_base={"REMOTE_ADDR": "192.168.1.100"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_cidr_network(self, mock_getenv, fresh_app):
        """Test réseau CIDR autorisé."""
        # Mock whitelist avec réseau CIDR
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-cidr", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.0/24"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-cidr",
            environ_base={"REMOTE_ADDR": "192.168.1.50"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_localhost_dev(self, mock_getenv, fresh_app):
        """Test localhost autorisé en développement."""
        # Mock environnement développement
        mock_getenv.side_effect = lambda key, default=None: (
            "development" if key == "FLASK_ENV" else None
        )

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-localhost", methods=["GET"])
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-localhost",
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_from_env(self, mock_getenv, fresh_app):
        """Test configuration via variable d'environnement."""
        # Mock whitelist depuis env
        mock_getenv.side_effect = lambda key, default=None: (
            "10.0.0.0/24" if key == "ADMIN_IP_WHITELIST" else "production"
        )

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-env", methods=["GET"])
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-env",
            environ_base={"REMOTE_ADDR": "10.0.0.5"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_x_forwarded_for(self, mock_getenv, fresh_app):
        """Test détection IP via X-Forwarded-For."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-xff", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-xff",
            environ_base={"REMOTE_ADDR": "10.0.0.1"},
            headers={"X-Forwarded-For": "192.168.1.100"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_x_real_ip(self, mock_getenv, fresh_app):
        """Test détection IP via X-Real-IP."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-xri", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.200"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-xri",
            environ_base={"REMOTE_ADDR": "10.0.0.1"},
            headers={"X-Real-IP": "192.168.1.200"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_x_forwarded_for_multiple(self, mock_getenv, fresh_app):
        """Test X-Forwarded-For avec plusieurs IPs (prend la première)."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-xff-multi", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-xff-multi",
            environ_base={"REMOTE_ADDR": "10.0.0.1"},
            headers={"X-Forwarded-For": "192.168.1.100, 10.0.0.2, 172.16.0.1"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}


class TestIPWhitelistBlocked:
    """Tests pour IPs bloquées."""

    @patch("security.ip_whitelist.send_ip_whitelist_alert")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_blocked_ip(self, mock_getenv, mock_send_alert, fresh_app):
        """Test IP non autorisée (accès refusé) avec alerte."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-blocked", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-blocked",
            environ_base={"REMOTE_ADDR": "192.168.1.200"},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        assert response.status_code == 403

        # Vérifier que l'alerte a été envoyée
        mock_send_alert.assert_called_once_with(
            client_ip="192.168.1.200",
            endpoint="/test-whitelist-blocked",
            method="GET",
            user_agent="Mozilla/5.0",
            user_id=None,
        )

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_invalid_ip(self, mock_getenv, fresh_app):
        """Test IP invalide (accès refusé)."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-invalid", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-invalid",
            environ_base={"REMOTE_ADDR": "invalid-ip"},
        )
        assert response.status_code == 403

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_no_client_ip(self, mock_getenv, fresh_app):
        """Test impossible de déterminer IP (accès refusé)."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-no-ip", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-no-ip",
            environ_base={},
        )
        assert response.status_code == 403


class TestIPWhitelistConfiguration:
    """Tests pour la configuration de la whitelist."""

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_no_config(self, mock_getenv, fresh_app):
        """Test pas de whitelist configurée (fail-open)."""
        # Mock pas de whitelist
        mock_getenv.side_effect = lambda key, default=None: (
            "development" if key == "FLASK_ENV" else None
        )

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-no-config", methods=["GET"])
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-no-config",
            environ_base={"REMOTE_ADDR": "192.168.1.100"},
        )
        # En développement, devrait autoriser (fail-open)
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_no_config_production(self, mock_getenv, fresh_app):
        """Test pas de whitelist en production (avertissement mais autorise)."""
        # Mock production sans whitelist
        mock_getenv.side_effect = lambda key, default=None: (
            "production" if key == "FLASK_ENV" else None
        )

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-no-config-prod", methods=["GET"])
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-no-config-prod",
            environ_base={"REMOTE_ADDR": "192.168.1.100"},
        )
        # En production sans whitelist, devrait quand même autoriser (fail-open)
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_multiple_ips(self, mock_getenv, fresh_app):
        """Test whitelist avec plusieurs IPs."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-multi", methods=["GET"])
        @ip_whitelist_required(allowed_ips=["192.168.1.100", "10.0.0.0/24"])
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-multi",
            environ_base={"REMOTE_ADDR": "10.0.0.5"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}

    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_ipv6(self, mock_getenv, fresh_app):
        """Test whitelist avec IPv6."""
        # Mock whitelist
        mock_getenv.return_value = None

        # Créer une route de test avec le décorateur
        @fresh_app.route("/test-whitelist-ipv6", methods=["GET"])
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        # Utiliser test_client pour faire une vraie requête HTTP
        client = fresh_app.test_client()
        response = client.get(
            "/test-whitelist-ipv6",
            environ_base={"REMOTE_ADDR": "::1"},
        )
        assert response.status_code == 200
        assert response.get_json() == {"status": "ok"}
