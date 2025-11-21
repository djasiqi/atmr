"""Tests unitaires pour le décorateur IP whitelist.

Valide le fonctionnement de la restriction d'accès par IP :
- Vérification IPs autorisées
- Blocage IPs non autorisées
- Support réseaux CIDR
- Détection IP via headers proxy
- Configuration via variables d'environnement
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from flask import Flask, abort

from security.ip_whitelist import ip_whitelist_required
from security.ip_whitelist_alerts import send_ip_whitelist_alert


@pytest.fixture
def mock_request():
    """Mock Flask request object."""
    request = MagicMock()
    request.environ = {}
    request.headers = {}
    request.method = "GET"
    request.path = "/api/admin/stats"
    return request


@pytest.fixture
def app():
    """Créer une app Flask pour les tests."""
    return Flask(__name__)


class TestIPWhitelistAllowed:
    """Tests pour IPs autorisées."""

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_allowed_ip(self, mock_getenv, mock_request):
        """Test IP autorisée (accès accordé)."""
        # Mock request avec IP autorisée
        mock_request.environ = {"REMOTE_ADDR": "192.168.1.100"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = "192.168.1.100"

        # Décorateur
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_cidr_network(self, mock_getenv, mock_request):
        """Test réseau CIDR autorisé."""
        # Mock request avec IP dans le réseau
        mock_request.environ = {"REMOTE_ADDR": "192.168.1.50"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist avec réseau CIDR
        mock_getenv.return_value = None

        # Décorateur avec réseau CIDR
        @ip_whitelist_required(allowed_ips=["192.168.1.0/24"])
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_localhost_dev(self, mock_getenv, mock_request):
        """Test localhost autorisé en développement."""
        # Mock request avec localhost
        mock_request.environ = {"REMOTE_ADDR": "127.0.0.1"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock environnement développement
        mock_getenv.side_effect = lambda key, default=None: ("development" if key == "FLASK_ENV" else None)

        # Décorateur (localhost autorisé par défaut)
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_from_env(self, mock_getenv, mock_request):
        """Test configuration via variable d'environnement."""
        # Mock request
        mock_request.environ = {"REMOTE_ADDR": "10.0.0.5"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist depuis env
        mock_getenv.side_effect = lambda key, default=None: (
            "10.0.0.0/24" if key == "ADMIN_IP_WHITELIST" else "production"
        )

        # Décorateur sans IPs explicites (utilise env)
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_x_forwarded_for(self, mock_getenv, mock_request):
        """Test détection IP via X-Forwarded-For."""
        # Mock request avec X-Forwarded-For
        mock_request.environ = {"REMOTE_ADDR": "10.0.0.1"}
        mock_request.headers = {"X-Forwarded-For": "192.168.1.100"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_x_real_ip(self, mock_getenv, mock_request):
        """Test détection IP via X-Real-IP."""
        # Mock request avec X-Real-IP
        mock_request.environ = {"REMOTE_ADDR": "10.0.0.1"}
        mock_request.headers = {"X-Real-IP": "192.168.1.200"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur
        @ip_whitelist_required(allowed_ips=["192.168.1.200"])
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_x_forwarded_for_multiple(self, mock_getenv, mock_request):
        """Test X-Forwarded-For avec plusieurs IPs (prend la première)."""
        # Mock request avec plusieurs IPs dans X-Forwarded-For
        mock_request.environ = {"REMOTE_ADDR": "10.0.0.1"}
        mock_request.headers = {"X-Forwarded-For": "192.168.1.100, 10.0.0.2, 172.16.0.1"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur (première IP autorisée)
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}


class TestIPWhitelistBlocked:
    """Tests pour IPs bloquées."""

    @patch("security.ip_whitelist.send_ip_whitelist_alert")
    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_blocked_ip(self, mock_getenv, mock_request, mock_send_alert):
        """Test IP non autorisée (accès refusé) avec alerte."""
        # Mock request avec IP non autorisée
        mock_request.environ = {"REMOTE_ADDR": "192.168.1.200"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"
        mock_request.headers = {"User-Agent": "Mozilla/5.0"}

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        from werkzeug.exceptions import Forbidden

        with pytest.raises(Forbidden, match="Accès non autorisé"):
            test_endpoint()

        # Vérifier que l'alerte a été envoyée
        mock_send_alert.assert_called_once_with(
            client_ip="192.168.1.200",
            endpoint="/api/admin/stats",
            method="GET",
            user_agent="Mozilla/5.0",
            user_id=None,
        )

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_invalid_ip(self, mock_getenv, mock_request):
        """Test IP invalide (accès refusé)."""
        # Mock request avec IP invalide
        mock_request.environ = {"REMOTE_ADDR": "invalid-ip"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        from werkzeug.exceptions import Forbidden

        with pytest.raises(Forbidden, match="Accès non autorisé"):
            test_endpoint()

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_no_client_ip(self, mock_getenv, mock_request):
        """Test impossible de déterminer IP (accès refusé)."""
        # Mock request sans IP
        mock_request.environ = {}
        mock_request.headers = {}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur
        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        from werkzeug.exceptions import Forbidden

        with pytest.raises(Forbidden, match="Accès non autorisé"):
            test_endpoint()


class TestIPWhitelistConfiguration:
    """Tests pour la configuration de la whitelist."""

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_no_config(self, mock_getenv, mock_request):
        """Test pas de whitelist configurée (fail-open)."""
        # Mock request
        mock_request.environ = {"REMOTE_ADDR": "192.168.1.100"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock pas de whitelist
        mock_getenv.side_effect = lambda key, default=None: ("development" if key == "FLASK_ENV" else None)

        # Décorateur sans configuration
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        # En développement, devrait autoriser (fail-open)
        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_no_config_production(self, mock_getenv, mock_request):
        """Test pas de whitelist en production (avertissement mais autorise)."""
        # Mock request
        mock_request.environ = {"REMOTE_ADDR": "192.168.1.100"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock production sans whitelist
        mock_getenv.side_effect = lambda key, default=None: ("production" if key == "FLASK_ENV" else None)

        # Décorateur sans configuration
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        # En production sans whitelist, devrait quand même autoriser (fail-open)
        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_multiple_ips(self, mock_getenv, mock_request):
        """Test whitelist avec plusieurs IPs."""
        # Mock request
        mock_request.environ = {"REMOTE_ADDR": "10.0.0.5"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur avec plusieurs IPs
        @ip_whitelist_required(allowed_ips=["192.168.1.100", "10.0.0.0/24"])
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_ip_whitelist_ipv6(self, mock_getenv, mock_request):
        """Test whitelist avec IPv6."""
        # Mock request avec IPv6
        mock_request.environ = {"REMOTE_ADDR": "::1"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"

        # Mock whitelist
        mock_getenv.return_value = None

        # Décorateur avec localhost IPv6 (autorisé par défaut)
        @ip_whitelist_required()
        def test_endpoint():
            return {"status": "ok"}

        result = test_endpoint()
        assert result == {"status": "ok"}
