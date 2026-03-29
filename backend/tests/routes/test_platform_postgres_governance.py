"""Tests d'intégration gouvernance — marqueur `postgres` (voir tests/README.md)."""

from __future__ import annotations

import pytest

from models.company import Company


@pytest.mark.postgres
def test_platform_tenant_detail_roundtrip(admin_tenant_client, sample_company):
    """Smoke persistance : GET tenant retourne les champs gouvernance."""
    tid = sample_company.id
    rv = admin_tenant_client.get(f"/api/v1/platform/tenants/{tid}")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["tenant_id"] == tid
    assert "effective_state" in data


@pytest.mark.postgres
def test_replay_contract_shape(admin_tenant_client, sample_company, db):
    """Contrat replay : racine correlation_id, count, events[]."""
    tid = sample_company.id
    Company.query.filter_by(id=tid).update({"platform_suspended": True})
    db.session.commit()

    cid = "test-replay-contract-1"
    headers = {**admin_tenant_client._h, "X-Correlation-Id": cid}
    rv = admin_tenant_client.post(
        "/api/v1/platform/runbooks/tenant_post_suspend_verify/executions",
        json={"tenant_id": tid},
        headers=headers,
    )
    assert rv.status_code == 200

    rv2 = admin_tenant_client.get(
        f"/api/v1/platform/audit-events/replay?correlation_id={cid}"
    )
    assert rv2.status_code == 200
    rep = rv2.get_json()
    assert rep["correlation_id"] == cid
    assert "count" in rep and isinstance(rep["count"], int)
    assert "events" in rep and isinstance(rep["events"], list)
    if rep["events"]:
        ev0 = rep["events"][0]
        for k in (
            "id",
            "created_at",
            "action_type",
            "result_status",
            "company_id",
            "resource_type",
            "resource_id",
        ):
            assert k in ev0
