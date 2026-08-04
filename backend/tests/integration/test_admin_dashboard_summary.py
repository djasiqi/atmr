"""Tests d'intégration : GET /api/v1/admin/dashboard-summary."""

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

        assert "generated_at" in data

        assert "priorities" in data
        p = data["priorities"]
        for key in (
            "bookings_pending_action",
            "demo_requests_open",
            "tenants_suspended",
            "platform_alerts_open",
            "billing_to_review",
            "critical_attention_count",
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
            "bookings_canceled_from_created_7d",
            "cancellation_rate_7d",
            "active_users_30d",
            "invoices_current_month",
            "revenue_current_month_chf",
            "platform_invoiced_current_month_chf",
        ):
            assert key in k
        assert isinstance(k["revenue_current_month_chf"], (int, float))
        assert isinstance(k["platform_invoiced_current_month_chf"], (int, float))
        assert isinstance(k["cancellation_rate_7d"], (int, float))
        assert 0.0 <= float(k["cancellation_rate_7d"]) <= 1.0

        assert "platform_snippet" in data
        s = data["platform_snippet"]
        for key in (
            "overall_status",
            "open_alerts",
            "runbooks_today",
            "tenants_in_drift",
            "critical_attention_count",
        ):
            assert key in s
        assert s["overall_status"] in ("ok", "degraded", "unknown")

        # Déprécié : toujours une liste vide (plus de série 12 mois).
        assert "booking_trends" in data
        assert data["booking_trends"] == []

        assert "recent_activity" in data
        assert isinstance(data["recent_activity"], list)
        assert len(data["recent_activity"]) <= 5
        for item in data["recent_activity"]:
            assert set(item.keys()) >= {
                "type",
                "entity_id",
                "label",
                "status",
                "occurred_at",
                "action",
            }
            assert "href" not in item
            assert item["type"] in (
                "booking",
                "demo_request",
                "tenant_governance",
                "runbook",
            )
            assert item["action"] == "open_booking"
