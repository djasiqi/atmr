"""Tests d'intégration : GET /api/v1/admin/dashboard/summary."""

from __future__ import annotations

import pytest


@pytest.mark.integration
class TestAdminDashboardSummaryEndpoint:
    """Contrat JSON stable pour le tableau de bord admin."""

    def test_summary_shape_and_keys(
        self,
        client,
        admin_headers,
        requires_postgresql,
    ):
        response = client.get(
            "/api/v1/admin/dashboard-summary",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        if response.status_code == 403:
            pytest.skip("IP whitelist ou JWT : ajuster environnement de test")
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json()
        assert data is not None

        assert "priorities" in data
        p = data["priorities"]
        for key in (
            "bookings_pending_action",
            "demo_requests_open",
            "tenants_suspended",
            "platform_alerts_open",
        ):
            assert key in p
            assert isinstance(p[key], int)
            assert p[key] >= 0

        assert "kpi_business" in data
        k = data["kpi_business"]
        for key in (
            "bookings_created_7d",
            "bookings_completed_7d",
            "bookings_canceled_7d",
            "active_users_30d",
            "invoices_current_month",
            "revenue_current_month_chf",
        ):
            assert key in k
        assert isinstance(k["revenue_current_month_chf"], (int, float))

        assert "platform_snippet" in data
        s = data["platform_snippet"]
        for key in ("overall_status", "open_alerts", "runbooks_today", "tenants_in_drift"):
            assert key in s
        assert s["overall_status"] in ("ok", "degraded", "unknown")

        assert "booking_trends" in data
        assert isinstance(data["booking_trends"], list)
        assert len(data["booking_trends"]) == 12

        assert "recent_activity" in data
        assert isinstance(data["recent_activity"], list)
        for item in data["recent_activity"]:
            assert set(item.keys()) >= {"type", "label", "status", "occurred_at", "href"}
            assert item["type"] in ("booking", "demo_request", "tenant_governance", "runbook")
