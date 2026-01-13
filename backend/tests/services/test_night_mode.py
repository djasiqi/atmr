# backend/tests/services/test_night_mode.py
"""Tests unitaires pour le service de mode nuit."""

from datetime import datetime, time
from unittest.mock import Mock, patch

import pytest

from services.events.night_mode import (
    NIGHT_END,
    NIGHT_START,
    is_night_time,
    should_send_night_notification,
)


class TestIsNightTime:
    """Tests pour is_night_time()."""

    def test_night_time_22h(self):
        """Test: 22h00 = nuit."""
        dt = datetime(2026, 1, 13, 22, 0)
        assert is_night_time(dt) is True

    def test_night_time_23h(self):
        """Test: 23h00 = nuit."""
        dt = datetime(2026, 1, 13, 23, 30)
        assert is_night_time(dt) is True

    def test_night_time_2h(self):
        """Test: 02h00 = nuit."""
        dt = datetime(2026, 1, 13, 2, 0)
        assert is_night_time(dt) is True

    def test_day_time_10h(self):
        """Test: 10h00 = jour."""
        dt = datetime(2026, 1, 13, 10, 0)
        assert is_night_time(dt) is False

    def test_day_time_14h(self):
        """Test: 14h00 = jour."""
        dt = datetime(2026, 1, 13, 14, 0)
        assert is_night_time(dt) is False

    def test_boundary_06h00(self):
        """Test: 06h00 exact = jour (fin de nuit)."""
        dt = datetime(2026, 1, 13, 6, 0)
        assert is_night_time(dt) is False

    def test_boundary_05h59(self):
        """Test: 05h59 = nuit."""
        dt = datetime(2026, 1, 13, 5, 59)
        assert is_night_time(dt) is True


class TestShouldSendNightNotification:
    """Tests pour should_send_night_notification()."""

    @patch("services.events.night_mode.is_night_time")
    def test_day_time_always_ok(self, mock_is_night):
        """Test: Jour = toujours OK."""
        mock_is_night.return_value = False

        assert should_send_night_notification("booking", driver_id=1) is True
        assert should_send_night_notification("message", driver_id=1) is True
        assert should_send_night_notification("info", driver_id=1) is True

    @patch("services.events.night_mode.is_night_time")
    def test_urgent_always_ok(self, mock_is_night):
        """Test: Urgences = toujours OK (même la nuit)."""
        mock_is_night.return_value = True

        assert should_send_night_notification("urgent_alert") is True
        assert should_send_night_notification("accident") is True
        assert should_send_night_notification("emergency") is True

    @patch("services.events.night_mode.is_night_time")
    @patch("services.events.night_mode.Driver")
    def test_mission_on_duty(self, mock_driver_model, mock_is_night):
        """Test: Mission la nuit = OK si chauffeur en service."""
        mock_is_night.return_value = True

        # Mock driver en service
        mock_driver = Mock()
        mock_driver.is_on_duty = True
        mock_driver_model.query.get.return_value = mock_driver

        assert should_send_night_notification("booking", driver_id=1) is True

    @patch("services.events.night_mode.is_night_time")
    @patch("services.events.night_mode.Driver")
    def test_mission_off_duty(self, mock_driver_model, mock_is_night):
        """Test: Mission la nuit = refusée si chauffeur hors service."""
        mock_is_night.return_value = True

        # Mock driver hors service
        mock_driver = Mock()
        mock_driver.is_on_duty = False
        mock_driver_model.query.get.return_value = mock_driver

        assert should_send_night_notification("booking", driver_id=1) is False

    @patch("services.events.night_mode.is_night_time")
    def test_message_refused(self, mock_is_night):
        """Test: Messages refusés la nuit."""
        mock_is_night.return_value = True

        assert should_send_night_notification("message") is False
        assert should_send_night_notification("team_chat_message") is False

    @patch("services.events.night_mode.is_night_time")
    def test_info_refused(self, mock_is_night):
        """Test: Infos refusées la nuit."""
        mock_is_night.return_value = True

        assert should_send_night_notification("dispatch_completed") is False
        assert should_send_night_notification("stats") is False
        assert should_send_night_notification("info") is False


# Exécuter les tests:
# pytest backend/tests/services/test_night_mode.py -v
