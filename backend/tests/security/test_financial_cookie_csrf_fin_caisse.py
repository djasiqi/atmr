"""FIN-CAISSE-2 : CSRF sur écritures financières cookie ; Bearer exempt.

Ne retire pas l'exemption historique de /api/v1/driver/ pour le GPS.
"""

from __future__ import annotations

import pytest
from flask import Flask

from security.jwt_token_locations import resolve_jwt_token_locations
from services.security.csrf import (
    financial_cookie_csrf_required,
    generate_csrf_token,
    is_financial_settlement_mutation_path,
    setup_csrf_protection,
)


@pytest.fixture
def csrf_app(monkeypatch):
    monkeypatch.setenv("CSRF_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET_KEY", "fin-caisse-2-csrf-secret-key!!")
    monkeypatch.setenv("SECRET_KEY", "fin-caisse-2-secret-key!!!!!!!!!!")
    monkeypatch.delenv("CSRF_STRICT", raising=False)

    app = Flask(__name__)
    app.config["TESTING"] = False
    app.config["CSRF_ENABLED"] = True
    app.config["WTF_CSRF_ENABLED"] = True
    setup_csrf_protection(app)

    @app.route(
        "/api/v1/driver/me/bookings/<int:booking_id>/cash-collection",
        methods=["POST"],
    )
    def cash_open(booking_id: int):
        return {"ok": True, "booking_id": booking_id}, 200

    @app.route(
        "/api/v1/driver/me/bookings/<int:booking_id>/cash-collection/transitions",
        methods=["POST"],
    )
    def cash_transition(booking_id: int):
        return {"ok": True, "action": "declare"}, 200

    @app.route("/api/v1/driver/me/location", methods=["PUT"])
    def driver_location():
        return {"ok": True}, 200

    @app.route(
        "/api/v1/companies/me/cash-register/<int:collection_id>/confirm-handover",
        methods=["POST"],
    )
    def company_confirm(collection_id: int):
        return {"ok": True, "collection_id": collection_id}, 200

    return app


class TestFinancialPathHelpers:
    def test_driver_cash_is_financial(self):
        assert is_financial_settlement_mutation_path(
            "/api/v1/driver/me/bookings/9/cash-collection"
        )
        assert is_financial_settlement_mutation_path(
            "/api/v1/driver/me/bookings/9/cash-collection/transitions"
        )

    def test_driver_location_not_financial(self):
        assert not is_financial_settlement_mutation_path(
            "/api/v1/driver/me/location"
        )

    def test_company_confirm_is_financial(self):
        assert is_financial_settlement_mutation_path(
            "/api/v1/companies/me/cash-register/3/confirm-handover"
        )

    def test_cookie_requires_csrf_bearer_skips(self):
        path = "/api/v1/driver/me/bookings/1/cash-collection"
        assert financial_cookie_csrf_required(path, authorization=None) is True
        assert (
            financial_cookie_csrf_required(path, authorization="Bearer abc") is False
        )


class TestAuthProA1Locations:
    def test_driver_bearer_headers_only(self):
        assert resolve_jwt_token_locations(
            "/api/v1/driver/me/bookings/1/cash-collection",
            authorization="Bearer token",
        ) == ["headers"]

    def test_driver_cookie_web_allows_cookies(self):
        assert resolve_jwt_token_locations(
            "/api/v1/driver/me/bookings/1/cash-collection",
            authorization=None,
        ) == ["cookies", "headers"]


class TestFinancialCookieCsrfMiddleware:
    def test_cookie_cash_without_csrf_403(self, csrf_app):
        client = csrf_app.test_client()
        resp = client.post(
            "/api/v1/driver/me/bookings/42/cash-collection",
            json={"exceptional": False},
        )
        assert resp.status_code == 403
        assert "CSRF" in (resp.get_json() or {}).get("error", "")

    def test_cookie_only_csrf_cookie_rejected(self, csrf_app):
        """Le cookie csrf_token seul ne protège pas une écriture financière."""
        client = csrf_app.test_client()
        token = generate_csrf_token()
        client.set_cookie("csrf_token", token)
        resp = client.post(
            "/api/v1/driver/me/bookings/42/cash-collection",
            json={"exceptional": False},
        )
        assert resp.status_code == 403

    def test_cookie_cash_with_header_csrf_200(self, csrf_app):
        client = csrf_app.test_client()
        token = generate_csrf_token()
        resp = client.post(
            "/api/v1/driver/me/bookings/42/cash-collection",
            json={"exceptional": False},
            headers={"X-CSRF-Token": token},
        )
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_cookie_cash_invalid_csrf_403(self, csrf_app):
        client = csrf_app.test_client()
        resp = client.post(
            "/api/v1/driver/me/bookings/42/cash-collection/transitions",
            json={"action": "declare_cash", "idempotency_key": "k1"},
            headers={"X-CSRF-Token": "not-a-valid-token"},
        )
        assert resp.status_code == 403
        assert "invalide" in (resp.get_json() or {}).get("error", "").lower()

    def test_bearer_cash_without_csrf_200(self, csrf_app):
        client = csrf_app.test_client()
        resp = client.post(
            "/api/v1/driver/me/bookings/42/cash-collection",
            json={"exceptional": False},
            headers={"Authorization": "Bearer mobile-token"},
        )
        assert resp.status_code == 200

    def test_driver_location_still_exempt(self, csrf_app):
        client = csrf_app.test_client()
        resp = client.put(
            "/api/v1/driver/me/location",
            json={"lat": 46.2, "lng": 6.1},
        )
        assert resp.status_code == 200

    def test_company_confirm_cookie_requires_csrf(self, csrf_app):
        client = csrf_app.test_client()
        denied = client.post(
            "/api/v1/companies/me/cash-register/7/confirm-handover",
            json={"idempotency_key": "h1"},
        )
        assert denied.status_code == 403
        token = generate_csrf_token()
        ok = client.post(
            "/api/v1/companies/me/cash-register/7/confirm-handover",
            json={"idempotency_key": "h1"},
            headers={"X-CSRF-Token": token},
        )
        assert ok.status_code == 200
