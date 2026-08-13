"""Tests de couverture pour backend/db.py (module critique ≥80 %)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from db import une_fonction_qui_cree_une_reservation


@pytest.fixture
def mock_booking_and_db():
    """Isole Booking + session DB pour des tests unitaires purs."""
    fake_booking = SimpleNamespace(id=42)
    booking_ctor = MagicMock(return_value=fake_booking)
    mock_session = MagicMock()
    mock_db = MagicMock()
    mock_db.session = mock_session
    mock_logger = MagicMock()

    with (
        patch("db.Booking", booking_ctor),
        patch("db.db", mock_db),
        patch("db.app_logger", mock_logger),
    ):
        yield {
            "booking_ctor": booking_ctor,
            "fake_booking": fake_booking,
            "session": mock_session,
            "logger": mock_logger,
        }


def test_missing_client_id_returns_400():
    body, status = une_fonction_qui_cree_une_reservation({"user_id": 1})
    assert status == 400
    assert body == {"error": "ID client ou utilisateur manquant"}


def test_missing_user_id_returns_400():
    body, status = une_fonction_qui_cree_une_reservation({"client_id": 1})
    assert status == 400
    assert body == {"error": "ID client ou utilisateur manquant"}


def test_invalid_client_id_returns_400():
    body, status = une_fonction_qui_cree_une_reservation(
        {"client_id": "abc", "user_id": 1}
    )
    assert status == 400
    assert body == {"error": "ID client ou utilisateur invalide"}


def test_invalid_user_id_returns_400():
    body, status = une_fonction_qui_cree_une_reservation(
        {"client_id": 1, "user_id": object()}
    )
    assert status == 400
    assert body == {"error": "ID client ou utilisateur invalide"}


def test_create_booking_success_with_defaults(mock_booking_and_db: dict[str, Any]):
    body, status = une_fonction_qui_cree_une_reservation(
        {"client_id": "10", "user_id": "20"}
    )

    assert status == 201
    assert body == {"message": "Réservation créée", "booking_id": 42}

    ctor = mock_booking_and_db["booking_ctor"]
    ctor.assert_called_once()
    payload = ctor.call_args.kwargs
    assert payload["client_id"] == 10
    assert payload["user_id"] == 20
    assert payload["customer_name"] == "John Doe"
    assert payload["pickup_location"] == "1 Rue de la Paix, 75002 Paris"
    assert payload["dropoff_location"] == "10 Avenue des Champs-Élysées, 75008 Paris"
    assert payload["amount"] == 50.0
    assert "company_id" not in payload

    session = mock_booking_and_db["session"]
    session.add.assert_called_once_with(mock_booking_and_db["fake_booking"])
    session.commit.assert_called_once()
    mock_booking_and_db["logger"].info.assert_called_once()


def test_create_booking_with_optional_fields(mock_booking_and_db: dict[str, Any]):
    body, status = une_fonction_qui_cree_une_reservation(
        {
            "client_id": 1,
            "user_id": 2,
            "customer_name": "Alice",
            "pickup_location": "A",
            "dropoff_location": "B",
            "amount": 12.5,
            "company_id": "99",
            "scheduled_time": "2026-08-12T10:00:00",
            "medical_facility": "Clinique X",
            "doctor_name": "Dr Y",
            "notes_medical": "Note Z",
        }
    )

    assert status == 201
    assert body["booking_id"] == 42

    payload = mock_booking_and_db["booking_ctor"].call_args.kwargs
    assert payload["customer_name"] == "Alice"
    assert payload["pickup_location"] == "A"
    assert payload["dropoff_location"] == "B"
    assert payload["amount"] == 12.5
    assert payload["company_id"] == 99
    assert payload["scheduled_time"] == "2026-08-12T10:00:00"
    assert payload["medical_facility"] == "Clinique X"
    assert payload["doctor_name"] == "Dr Y"
    assert payload["notes_medical"] == "Note Z"


def test_invalid_company_id_is_suppressed(mock_booking_and_db: dict[str, Any]):
    _body, status = une_fonction_qui_cree_une_reservation(
        {
            "client_id": 1,
            "user_id": 2,
            "company_id": "not-an-int",
        }
    )

    assert status == 201
    payload = mock_booking_and_db["booking_ctor"].call_args.kwargs
    assert "company_id" not in payload


def test_create_booking_rolls_back_on_commit_error(
    mock_booking_and_db: dict[str, Any],
):
    mock_booking_and_db["session"].commit.side_effect = RuntimeError("db down")

    body, status = une_fonction_qui_cree_une_reservation({"client_id": 1, "user_id": 2})

    assert status == 500
    assert body == {"error": "Une erreur interne est survenue"}
    mock_booking_and_db["session"].rollback.assert_called_once()
    mock_booking_and_db["logger"].error.assert_called_once()
    err_msg = mock_booking_and_db["logger"].error.call_args[0][0]
    assert "db down" in err_msg
