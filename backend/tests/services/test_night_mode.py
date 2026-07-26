# backend/tests/services/test_night_mode.py

"""Tests unitaires pour le service de mode nuit."""

from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from services.events.night_mode import (
    NIGHT_END,
    NIGHT_START,
    is_night_time,
    should_send_night_notification,
)

PARIS = ZoneInfo("Europe/Paris")


class TestIsNightTime:
    """Tests pour is_night_time()."""

    def test_night_time_22h_paris(self):
        dt = datetime(2026, 1, 13, 22, 0, tzinfo=PARIS)

        assert is_night_time(dt) is True

    def test_night_time_23h_paris(self):
        dt = datetime(2026, 1, 13, 23, 30, tzinfo=PARIS)

        assert is_night_time(dt) is True

    def test_night_time_2h_paris(self):
        dt = datetime(2026, 1, 13, 2, 0, tzinfo=PARIS)

        assert is_night_time(dt) is True

    def test_day_time_10h_paris(self):
        dt = datetime(2026, 1, 13, 10, 0, tzinfo=PARIS)

        assert is_night_time(dt) is False

    def test_day_time_14h_paris(self):
        dt = datetime(2026, 1, 13, 14, 0, tzinfo=PARIS)

        assert is_night_time(dt) is False

    def test_boundary_06h00_paris(self):
        dt = datetime(2026, 1, 13, 6, 0, tzinfo=PARIS)

        assert is_night_time(dt) is False

    def test_boundary_05h59_paris(self):
        dt = datetime(2026, 1, 13, 5, 59, tzinfo=PARIS)

        assert is_night_time(dt) is True

    def test_default_timezone_is_europe_paris(self):
        from services.events import night_mode as nm

        assert nm.DEFAULT_NIGHT_MODE_TZ == "Europe/Paris"


class TestShouldSendNightNotification:
    """Tests pour should_send_night_notification()."""

    @patch("services.events.night_mode.is_night_time")
    def test_day_time_always_ok(self, mock_is_night):
        mock_is_night.return_value = False

        assert should_send_night_notification("booking", driver_id=1) is True

        assert should_send_night_notification("message", driver_id=1) is True

        assert should_send_night_notification("info", driver_id=1) is True

    @patch("services.events.night_mode.is_night_time")
    def test_urgent_always_ok(self, mock_is_night):
        mock_is_night.return_value = True

        assert should_send_night_notification("urgent_alert") is True

        assert should_send_night_notification("accident") is True

        assert should_send_night_notification("emergency") is True

    @patch("services.events.night_mode.is_night_time")
    @patch("models.Driver")
    def test_mission_available_driver(self, mock_driver_model, mock_is_night):
        mock_is_night.return_value = True

        mock_driver = Mock()

        mock_driver.is_available = True

        mock_driver_model.query.get.return_value = mock_driver

        assert should_send_night_notification("booking", driver_id=1) is True

    @patch("services.events.night_mode.is_night_time")
    @patch("models.Driver")
    def test_mission_unavailable_driver(self, mock_driver_model, mock_is_night):
        mock_is_night.return_value = True

        mock_driver = Mock()

        mock_driver.is_available = False

        mock_driver_model.query.get.return_value = mock_driver

        assert should_send_night_notification("booking", driver_id=1) is False

    @patch("services.events.night_mode.is_night_time")
    def test_message_allowed(self, mock_is_night):
        mock_is_night.return_value = True

        assert should_send_night_notification("message") is True

        assert should_send_night_notification("team_chat_message") is True

        assert should_send_night_notification("chat_message") is True

    @patch("services.events.night_mode.is_night_time")
    def test_info_refused(self, mock_is_night):
        mock_is_night.return_value = True

        assert should_send_night_notification("dispatch_completed") is False

        assert should_send_night_notification("stats") is False

        assert should_send_night_notification("info") is False
