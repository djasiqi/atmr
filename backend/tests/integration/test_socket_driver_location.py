"""
Tests pour les handlers socket driver_location.

Vérifie notamment :
- test_driver_without_company : un chauffeur sans company_id ne peut pas envoyer de position.
  Attendu : emit("error", ...), aucune écriture Redis.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from ext import socketio
from models import Driver


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("os").environ.get("SKIP_SOCKETIO", "false").lower() == "true",
    reason="SKIP_SOCKETIO=1, socket tests désactivés",
)
class TestSocketDriverLocation:
    """Tests ownership et validation socket driver_location."""

    def test_driver_without_company_emits_error_and_no_redis_write(
        self, app, db, test_company, test_driver
    ):
        """Un chauffeur avec company_id=None ne doit pas pouvoir envoyer de position.

        Attendu :
        - emit("error", {"error": "Chauffeur non lié à une entreprise."})
        - Aucune écriture Redis (LocationService.update_driver_location non appelé)
        """
        from flask_jwt_extended import create_access_token

        if not test_driver or not test_driver.user:
            pytest.skip("test_driver fixture required")
        user = test_driver.user
        driver = test_driver
        db.session.refresh(driver)
        db.session.refresh(user)

        # JWT pour la connexion socket
        with app.app_context():
            token = create_access_token(
                identity=str(user.public_id),
                additional_claims={
                    "role": "driver",
                    "aud": "atmr-api",
                },
                expires_delta=timedelta(hours=1),
            )

        # Connexion socket (driver a company_id pour que connect réussisse)
        with app.app_context():
            client = socketio.test_client(
                app,
                auth={"token": token},
                flask_test_client=app.test_client(),
            )

        if not client.is_connected():
            pytest.skip(
                "Socket client could not connect. "
                "Vérifier que Socket.IO est initialisé (SKIP_SOCKETIO=false)."
            )

        # Mock : Driver sans company pour le handler
        mock_location_service = MagicMock()
        driver_no_company = MagicMock(spec=Driver)
        driver_no_company.id = driver.id
        driver_no_company.company_id = None

        with (
            patch(
                "sockets.chat.get_location_service",
                return_value=mock_location_service,
            ),
            patch.object(Driver, "query") as mock_driver_query,
        ):
            mock_filter = MagicMock()
            mock_filter.first.return_value = driver_no_company
            mock_driver_query.filter_by.return_value = mock_filter

            client.emit(
                "driver_location",
                {
                    "latitude": 46.2044,
                    "longitude": 6.1432,
                },
            )

        received = client.get_received()
        error_events = [e for e in received if e.get("name") == "error"]
        assert len(error_events) >= 1, (
            f"Expected 'error' event, got: {[e.get('name') for e in received]}"
        )
        payload = error_events[0].get("args", [{}])[0]
        assert "Chauffeur non lié à une entreprise" in str(
            payload.get("error", "")
        ), f"Expected error message, got: {payload}"

        mock_location_service.update_driver_location.assert_not_called()
