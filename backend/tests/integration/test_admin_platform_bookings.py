"""Tests d'intégration : GET /api/v1/admin/bookings et GET .../bookings/:id."""

from __future__ import annotations

import pytest


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
        assert "page" in p and "per_page" in p and "total_items" in p
        if data["items"]:
            row = data["items"][0]
            for key in (
                "id",
                "status",
                "status_label",
                "client_name",
                "current_company_name",
                "created_by",
            ):
                assert key in row

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
