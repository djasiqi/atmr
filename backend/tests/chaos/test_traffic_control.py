"""Tests unitaires pour chaos/traffic_control.py

Valide les fonctions de validation et la sécurité des appels subprocess.
"""

from unittest.mock import MagicMock, patch

import pytest

# Import conditionnel pour éviter les erreurs si le module n'est pas disponible
try:
    from chaos.traffic_control import (
        MAX_JITTER_MS,
        MAX_LATENCY_MS,
        SUBPROCESS_TIMEOUT,
        TrafficControlManager,
    )
except ImportError:
    pytest.skip("chaos.traffic_control module not available", allow_module_level=True)


class TestTrafficControlValidations:
    """Tests pour les fonctions de validation."""

    def test_validate_interface_valid(self):
        """Test validation d'interfaces valides."""
        assert TrafficControlManager._validate_interface("eth0") is True
        assert TrafficControlManager._validate_interface("ens33") is True
        assert TrafficControlManager._validate_interface("wlan0") is True
        assert TrafficControlManager._validate_interface("lo") is True

    def test_validate_interface_invalid(self):
        """Test validation d'interfaces invalides."""
        assert TrafficControlManager._validate_interface("") is False
        assert TrafficControlManager._validate_interface("eth0;rm -rf /") is False
        assert TrafficControlManager._validate_interface("eth 0") is False  # Espace
        assert TrafficControlManager._validate_interface("eth0" * 5) is False  # Trop long
        assert TrafficControlManager._validate_interface(None) is False
        assert TrafficControlManager._validate_interface(123) is False

    def test_validate_latency_ms_valid(self):
        """Test validation de latence valide."""
        assert TrafficControlManager._validate_latency_ms(1) is True
        assert TrafficControlManager._validate_latency_ms(100) is True
        assert TrafficControlManager._validate_latency_ms(MAX_LATENCY_MS) is True

    def test_validate_latency_ms_invalid(self):
        """Test validation de latence invalide."""
        assert TrafficControlManager._validate_latency_ms(0) is False
        assert TrafficControlManager._validate_latency_ms(-1) is False
        assert TrafficControlManager._validate_latency_ms(MAX_LATENCY_MS + 1) is False
        assert TrafficControlManager._validate_latency_ms(None) is False
        assert TrafficControlManager._validate_latency_ms("100") is False

    def test_validate_jitter_ms_valid(self):
        """Test validation de jitter valide."""
        assert TrafficControlManager._validate_jitter_ms(0) is True
        assert TrafficControlManager._validate_jitter_ms(100) is True
        assert TrafficControlManager._validate_jitter_ms(MAX_JITTER_MS) is True

    def test_validate_jitter_ms_invalid(self):
        """Test validation de jitter invalide."""
        assert TrafficControlManager._validate_jitter_ms(-1) is False
        assert TrafficControlManager._validate_jitter_ms(MAX_JITTER_MS + 1) is False
        assert TrafficControlManager._validate_jitter_ms(None) is False
        assert TrafficControlManager._validate_jitter_ms("100") is False

    def test_validate_percent_valid(self):
        """Test validation de pourcentage valide."""
        assert TrafficControlManager._validate_percent(0.0) is True
        assert TrafficControlManager._validate_percent(50.0) is True
        assert TrafficControlManager._validate_percent(100.0) is True
        assert TrafficControlManager._validate_percent(0) is True  # Int
        assert TrafficControlManager._validate_percent(100) is True  # Int

    def test_validate_percent_invalid(self):
        """Test validation de pourcentage invalide."""
        assert TrafficControlManager._validate_percent(-0.1) is False
        assert TrafficControlManager._validate_percent(100.1) is False
        assert TrafficControlManager._validate_percent(None) is False
        assert TrafficControlManager._validate_percent("50") is False


class TestTrafficControlManager:
    """Tests pour TrafficControlManager."""

    def test_init_valid_interface(self):
        """Test initialisation avec interface valide."""
        manager = TrafficControlManager(interface="eth0")
        assert manager.interface == "eth0"
        assert manager.active is False

    def test_init_invalid_interface(self):
        """Test initialisation avec interface invalide."""
        with pytest.raises(ValueError, match="Invalid interface name"):
            TrafficControlManager(interface="eth0;rm -rf /")

    @patch("chaos.traffic_control.os.geteuid")
    @patch("chaos.traffic_control.subprocess.run")
    def test_add_latency_success(self, mock_subprocess, mock_geteuid):
        """Test ajout de latence avec succès."""
        mock_geteuid.return_value = 0  # Root
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        manager = TrafficControlManager(interface="eth0")
        result = manager.add_latency(ms=100, jitter_ms=10)

        assert result is True
        assert manager.active is True
        mock_subprocess.assert_called_once()
        # Vérifier que timeout est présent
        call_args = mock_subprocess.call_args
        assert call_args[1]["timeout"] == SUBPROCESS_TIMEOUT
        assert "shell" not in call_args[1] or call_args[1].get("shell") is False

    @patch("chaos.traffic_control.os.geteuid")
    def test_add_latency_no_root(self, mock_geteuid):
        """Test ajout de latence sans privilèges root."""
        mock_geteuid.return_value = 1000  # Non-root

        manager = TrafficControlManager(interface="eth0")
        result = manager.add_latency(ms=100)

        assert result is False

    @patch("chaos.traffic_control.os.geteuid")
    def test_add_latency_invalid_params(self, mock_geteuid):
        """Test ajout de latence avec paramètres invalides."""
        mock_geteuid.return_value = 0  # Root

        manager = TrafficControlManager(interface="eth0")

        # Latence invalide
        assert manager.add_latency(ms=0) is False
        assert manager.add_latency(ms=-1) is False
        assert manager.add_latency(ms=MAX_LATENCY_MS + 1) is False

        # Jitter invalide
        assert manager.add_latency(ms=100, jitter_ms=-1) is False
        assert manager.add_latency(ms=100, jitter_ms=MAX_JITTER_MS + 1) is False

    @patch("chaos.traffic_control.os.geteuid")
    @patch("chaos.traffic_control.subprocess.run")
    def test_add_latency_timeout(self, mock_subprocess, mock_geteuid):
        """Test gestion du timeout lors de l'ajout de latence."""
        from subprocess import TimeoutExpired

        mock_geteuid.return_value = 0  # Root
        mock_subprocess.side_effect = TimeoutExpired(cmd=["tc"], timeout=10)

        manager = TrafficControlManager(interface="eth0")
        result = manager.add_latency(ms=100)

        assert result is False

    @patch("chaos.traffic_control.os.geteuid")
    @patch("chaos.traffic_control.subprocess.run")
    def test_add_packet_loss_success(self, mock_subprocess, mock_geteuid):
        """Test ajout de perte de paquets avec succès."""
        mock_geteuid.return_value = 0  # Root
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        manager = TrafficControlManager(interface="eth0")
        result = manager.add_packet_loss(percent=10.0)

        assert result is True
        mock_subprocess.assert_called_once()
        # Vérifier que timeout est présent
        call_args = mock_subprocess.call_args
        assert call_args[1]["timeout"] == SUBPROCESS_TIMEOUT

    @patch("chaos.traffic_control.os.geteuid")
    def test_add_packet_loss_invalid_percent(self, mock_geteuid):
        """Test ajout de perte de paquets avec pourcentage invalide."""
        mock_geteuid.return_value = 0  # Root

        manager = TrafficControlManager(interface="eth0")

        # Pourcentage invalide
        assert manager.add_packet_loss(percent=-0.1) is False
        assert manager.add_packet_loss(percent=100.1) is False

    @patch("chaos.traffic_control.os.geteuid")
    @patch("chaos.traffic_control.subprocess.run")
    def test_clear_success(self, mock_subprocess, mock_geteuid):
        """Test suppression des règles TC avec succès."""
        mock_geteuid.return_value = 0  # Root
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        manager = TrafficControlManager(interface="eth0")
        manager.active = True
        result = manager.clear()

        assert result is True
        assert manager.active is False
        mock_subprocess.assert_called_once()
        # Vérifier que timeout est présent
        call_args = mock_subprocess.call_args
        assert call_args[1]["timeout"] == SUBPROCESS_TIMEOUT

    @patch("chaos.traffic_control.subprocess.run")
    def test_is_active_success(self, mock_subprocess):
        """Test vérification de règles TC actives."""
        mock_result = MagicMock()
        mock_result.stdout = "netem delay 100ms"
        mock_subprocess.return_value = mock_result

        manager = TrafficControlManager(interface="eth0")
        result = manager.is_active()

        assert result is True
        mock_subprocess.assert_called_once()
        # Vérifier que timeout est présent
        call_args = mock_subprocess.call_args
        assert call_args[1]["timeout"] == SUBPROCESS_TIMEOUT

    @patch("chaos.traffic_control.subprocess.run")
    def test_is_active_timeout(self, mock_subprocess):
        """Test gestion du timeout lors de la vérification."""
        from subprocess import TimeoutExpired

        mock_subprocess.side_effect = TimeoutExpired(cmd=["tc"], timeout=10)

        manager = TrafficControlManager(interface="eth0")
        result = manager.is_active()

        assert result is False
