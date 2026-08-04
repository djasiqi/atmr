"""Tests d'intégration : GET /api/v1/admin/bookings et GET .../bookings/:id."""

from __future__ import annotations

import pytest


_PII_FORBIDDEN_KEYS = {
    "birth_date",
    "notes_medical",
    "door_code",
    "pickup_door_code",
    "dropoff_door_code",
    "contact_phone",
    "gp_phone",
    "pickup_lat",
    "pickup_lon",
    "dropoff_lat",
    "dropoff_lon",
    "online_payment",
}


def _assert_no_forbidden_keys(obj, *, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert k not in _PII_FORBIDDEN_KEYS, f"PII key at {path}.{k}"
            assert k != "booking", "Ancien payload booking.serialize interdit"
            if k == "links":
                pytest.fail(f"Clé legacy {k} interdite dans le détail")
            _assert_no_forbidden_keys(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _assert_no_forbidden_keys(item, path=f"{path}[{i}]")


@pytest.mark.integration
class TestAdminPlatformBookingsList:
    def test_list_shape_and_keys(
        self,
        client,
        admin_headers,
        requires_postgresql,
    ):
        response = client.get(
            "/api/v1/admin/bookings?page=1&per_page=5",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        if response.status_code == 403:
            pytest.skip("IP whitelist ou JWT : ajuster environnement de test")
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json()
        assert data is not None
        assert "summary" in data
        s = data["summary"]
        for key in (
            "total",
            "unassigned",
            "canceled",
            "transferred",
            "incomplete_data",
            "needs_investigation",
        ):
            assert key in s
            assert isinstance(s[key], int)
        assert "items" in data
        assert isinstance(data["items"], list)
        assert "pagination" in data
        p = data["pagination"]
        assert "page" in p
        assert "per_page" in p
        assert "total_items" in p
        if data["items"]:
            row = data["items"][0]
            for key in (
                "id",
                "status",
                "status_label",
                "client_name",
                "current_company_name",
                "created_by",
                "investigation_reasons",
                "needs_investigation",
            ):
                assert key in row
            assert isinstance(row["investigation_reasons"], list)

    def test_detail_404_unknown_id(
        self,
        client,
        admin_headers,
        requires_postgresql,
    ):
        response = client.get(
            "/api/v1/admin/bookings/999999999",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        if response.status_code == 403:
            pytest.skip("IP whitelist ou JWT : ajuster environnement de test")
        assert response.status_code == 404

    def test_detail_support_contract_shape(
        self,
        client,
        admin_headers,
        requires_postgresql,
    ):
        list_resp = client.get(
            "/api/v1/admin/bookings?page=1&per_page=1",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        if list_resp.status_code == 403:
            pytest.skip("IP whitelist ou JWT : ajuster environnement de test")
        assert list_resp.status_code == 200
        items = (list_resp.get_json() or {}).get("items") or []
        if not items:
            pytest.skip("Aucune réservation en base de test")
        booking_id = items[0]["id"]
        response = client.get(
            f"/api/v1/admin/bookings/{booking_id}",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json()
        assert data["id"] == booking_id
        for key in (
            "transport",
            "support_diagnostic",
            "actors",
            "timeline",
            "references",
        ):
            assert key in data
        assert "booking" not in data
        assert "links" not in data
        diag = data["support_diagnostic"]
        assert diag["status"] in ("action_required", "attention", "ok")
        assert isinstance(diag["reasons"], list)
        assert data["references"]["booking_id"] == booking_id
        payload_str = response.get_data(as_text=True)
        assert "/platform-ops" not in payload_str
        assert "/dashboard/admin/" not in payload_str
        _assert_no_forbidden_keys(data)
