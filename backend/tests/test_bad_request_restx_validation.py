"""BadRequest handler : erreurs RESTX ne doivent pas être masquées en invalid_json."""

from __future__ import annotations

import json
import os

import pytest
from werkzeug.exceptions import BadRequest


@pytest.fixture
def app_client():
    os.environ["CSRF_ENABLED"] = "false"
    from app import create_app

    app = create_app("development")
    app.config["TESTING"] = True
    app.config["CSRF_ENABLED"] = False
    return app.test_client()


def test_missing_version_returns_marshmallow_error_not_invalid_json(
    app_client, monkeypatch
):
    """version manquante → erreur Marshmallow explicite (pas invalid_json)."""
    monkeypatch.setattr(
        "routes.institution_bookings.get_institution_booking_context",
        lambda: (1, 1, "institution_admin", "Test"),
    )

    response = app_client.patch(
        "/api/v1/institutions/bookings/31009",
        json={"pickup_location": "x"},
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body is not None
    assert body.get("error") != "invalid_json"
    assert body.get("error") in {"Données invalides", "validation_error"}


def test_string_version_coerced_by_marshmallow(app_client, monkeypatch):
    """version chaîne acceptée par Marshmallow (validate=False sur RESTX expect)."""
    from routes.institution_bookings import get_institution_booking_context

    monkeypatch.setattr(
        "routes.institution_bookings.get_institution_booking_context",
        lambda: (1, 1, "institution_admin", "Test"),
    )
    monkeypatch.setattr(
        "routes.institution_bookings.resolve_institution_booking",
        lambda *_a, **_k: None,
    )

    response = app_client.patch(
        "/api/v1/institutions/bookings/31009",
        data=json.dumps({"version": "1", "pickup_location": "x"}),
        content_type="application/json",
    )
    assert response.status_code == 404
    body = response.get_json()
    assert body.get("error") != "invalid_json"


def test_bad_request_handler_reads_restx_data():
    """Le handler BadRequest expose e.data RESTX."""
    from app import create_app

    app = create_app("development")
    handlers = app.error_handler_spec[None][400]
    handler = handlers[BadRequest]
    exc = BadRequest()
    exc.data = {
        "message": "Input payload validation failed",
        "errors": {"version": "'version' is a required property"},
    }
    with app.test_request_context(
        "/api/v1/institutions/bookings/1",
        method="PATCH",
        json={"pickup_location": "x"},
    ):
        response, status = handler(exc)
        payload = response.get_json()
        assert status == 400
        assert payload["error"] == "validation_error"
        assert "version" in payload["message"].lower()
