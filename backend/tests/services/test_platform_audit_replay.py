"""Tests unitaires — replay audit par correlation_id (contrat V1)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from services.platform_audit_events import replay_timeline_by_correlation_id


@patch("services.platform_audit_events.AuditLog")
def test_replay_timeline_shape_and_order(mock_al):
    """Ordre id asc ; clés alignées spec §6 bis."""
    dt = datetime.now(UTC)
    r1 = MagicMock()
    r1.id = 5
    r1.created_at = dt
    r1.action_type = "platform_runbook_execution"
    r1.result_status = "success"
    r1.company_id = 1
    r1.resource_type = "runbook_execution"
    r1.resource_id = "ex-1"

    r2 = MagicMock()
    r2.id = 12
    r2.created_at = dt
    r2.action_type = "platform_runbook_rollback"
    r2.result_status = "success"
    r2.company_id = 1
    r2.resource_type = "runbook_execution"
    r2.resource_id = "ex-1"

    mock_al.query.filter.return_value.order_by.return_value.all.return_value = [r1, r2]

    out = replay_timeline_by_correlation_id("cid-test")
    assert out["correlation_id"] == "cid-test"
    assert out["count"] == 2
    assert len(out["events"]) == 2
    assert out["events"][0]["id"] == 5
    assert out["events"][1]["id"] == 12
    for ev in out["events"]:
        assert set(ev.keys()) == {
            "id",
            "created_at",
            "action_type",
            "result_status",
            "company_id",
            "resource_type",
            "resource_id",
        }
