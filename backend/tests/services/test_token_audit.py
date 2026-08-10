# backend/tests/services/test_token_audit.py
"""Tests P0.4: token_audit (collision detection, recipient proof)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from services.notifications.token_audit import (
    _token_hash,
    check_token_collision,
    log_push_recipient_proof,
)


class TestTokenHash:
    """Tests _token_hash (jamais logger le token brut)."""

    def test_token_hash_deterministic(self) -> None:
        """Même token -> même hash."""
        h1 = _token_hash("ExponentPushToken[abc123]")
        h2 = _token_hash("ExponentPushToken[abc123]")
        assert h1 == h2

    def test_token_hash_different_tokens(self) -> None:
        """Tokens différents -> hashes différents."""
        h1 = _token_hash("token_a")
        h2 = _token_hash("token_b")
        assert h1 != h2

    def test_token_hash_truncated_8_chars(self) -> None:
        """Hash tronqué à 8 chars par défaut."""
        h = _token_hash("any_token")
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_token_hash_empty_returns_empty(self) -> None:
        """Token vide -> hash vide."""
        assert _token_hash("") == ""


class TestCheckTokenCollision:
    """Tests détection collision driver vs company."""

    @pytest.mark.parametrize("debug", [True, False])
    def test_no_collision_distinct_tokens(self, debug: bool) -> None:
        """Driver token X, company token Y (distinct) -> pas de collision."""
        with patch.dict(os.environ, {"DEBUG_NOTIF_ROUTING": "1" if debug else "0"}):
            # Reimport pour prendre en compte l'env
            import importlib

            import services.notifications.token_audit as m

            importlib.reload(m)
            result = m.check_token_collision(
                driver_tokens=["token_driver_a"],
                company_token="token_company_b",
                driver_id=33,
                company_user_id=42,
            )
            assert result is False

    def test_collision_same_token(self) -> None:
        """Driver token X, company token X -> collision détectée."""
        shared_token = "ExponentPushToken[same]"
        with patch.dict(os.environ, {"DEBUG_NOTIF_ROUTING": "1"}):
            import importlib

            import services.notifications.token_audit as m

            importlib.reload(m)
            with patch("ext.app_logger") as mock_log:
                result = m.check_token_collision(
                    driver_tokens=[shared_token],
                    company_token=shared_token,
                    driver_id=33,
                    company_user_id=42,
                    trace_id="trace-123",
                )
                assert result is True
                mock_log.warning.assert_called_once()
                call_args = str(mock_log.warning.call_args)
                assert "COLLISION" in call_args
                assert "trace-123" in call_args


class TestLogPushRecipientProof:
    """Tests log_push_recipient_proof (DEBUG_NOTIF_ROUTING)."""

    def test_log_when_debug_enabled(self) -> None:
        """Log appelé quand DEBUG_NOTIF_ROUTING=1."""
        with patch.dict(os.environ, {"DEBUG_NOTIF_ROUTING": "1"}):
            import importlib

            import services.notifications.token_audit as m

            importlib.reload(m)
            with patch("ext.app_logger") as mock_log:
                m.log_push_recipient_proof(
                    trace_id="t1",
                    booking_id=1,
                    status="in_progress",
                    recipient_role="company",
                    recipient_id=42,
                    token_count=1,
                    token_hashes=["abc12345"],
                    collapse_key="booking:1",
                )
                mock_log.info.assert_called_once()
                call_args = str(mock_log.info.call_args)
                assert "PUSH_RECIPIENT_PROOF" in call_args
                assert "company" in call_args
                assert "abc12345" in call_args

    def test_no_log_when_debug_disabled(self) -> None:
        """Pas de log quand DEBUG_NOTIF_ROUTING=0."""
        with patch.dict(os.environ, {"DEBUG_NOTIF_ROUTING": "0"}):
            import importlib

            import services.notifications.token_audit as m

            importlib.reload(m)
            with patch("ext.app_logger") as mock_log:
                m.log_push_recipient_proof(
                    trace_id="t1",
                    booking_id=1,
                    status="in_progress",
                    recipient_role="driver",
                    recipient_id=33,
                    token_count=2,
                    token_hashes=["a1", "b2"],
                )
                mock_log.info.assert_not_called()
