"""Tests API plateforme — policy, tenants, runbooks (auth admin)."""

from __future__ import annotations

import pytest
from sqlalchemy import delete, inspect

from ext import db
from models.company import Company
from models.platform_admin_permission_grant import PlatformAdminPermissionGrant
from services.platform_authz import PERM_OBSERVE_TENANT_READ, PERM_POLICY_EXPLAIN


def _platform_rbac_table_ready() -> bool:
    return bool(inspect(db.engine).has_table("platform_admin_permission_grant"))


@pytest.fixture
def admin_tenant_client(client, admin_headers):
    """Client avec en-têtes admin pour routes /platform/*."""

    class H:
        def __init__(self, c, headers):
            self._c = c
            self._h = headers

        def _merge(self, kw):
            h = dict(self._h)
            h.update(kw.pop("headers", None) or {})
            kw["headers"] = h
            return kw

        def get(self, path, **kw):
            return self._c.get(path, **self._merge(kw))

        def post(self, path, **kw):
            return self._c.post(path, **self._merge(kw))

    return H(client, admin_headers)


def test_platform_me_ok(admin_tenant_client):
    rv = admin_tenant_client.get("/api/v1/platform/me")
    assert rv.status_code == 200
    data = rv.get_json()
    assert "role" in data
    assert data["role"] == "admin"


def test_platform_me_includes_permissions_effective(admin_tenant_client):
    rv = admin_tenant_client.get("/api/v1/platform/me")
    assert rv.status_code == 200
    perms = rv.get_json().get("platform", {}).get("permissions_effective")
    assert isinstance(perms, list)
    assert len(perms) >= 1


def test_policies_evaluate_allow(admin_tenant_client):
    rv = admin_tenant_client.post(
        "/api/v1/platform/policies/evaluate",
        json={
            "action_type": "governance.tenant.suspend",
            "scope_type": "tenant",
            "scope_id": "1",
        },
    )
    assert rv.status_code == 200
    body = rv.get_json()
    assert body["policy_evaluation_result"]["decision"] == "allow"


def test_tenants_list_contains_sample_company(admin_tenant_client, sample_company):
    rv = admin_tenant_client.get("/api/v1/platform/tenants?per_page=50")
    assert rv.status_code == 200
    data = rv.get_json()
    ids = {x["tenant_id"] for x in data["items"]}
    assert sample_company.id in ids


def test_tenant_detail_governance_shape(admin_tenant_client, sample_company):
    tid = sample_company.id
    rv = admin_tenant_client.get(f"/api/v1/platform/tenants/{tid}")
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["tenant_id"] == tid
    assert "desired_state" in data
    assert "observed_state" in data
    assert "reconciliation_status" in data
    assert "effective_state" in data


def test_suspend_preview(admin_tenant_client, sample_company):
    tid = sample_company.id
    rv = admin_tenant_client.post(
        f"/api/v1/platform/tenants/{tid}/suspend/preview",
        json={},
    )
    assert rv.status_code == 200
    data = rv.get_json()
    assert "blast_radius" in data
    assert data["tenant_id"] == tid


def test_suspend_requires_justification(admin_tenant_client, sample_company):
    tid = sample_company.id
    rv = admin_tenant_client.post(
        f"/api/v1/platform/tenants/{tid}/suspend",
        json={"justification": "no"},
    )
    assert rv.status_code == 400


def test_suspend_applies_platform_flag(admin_tenant_client, sample_company, db):
    tid = sample_company.id
    Company.query.filter_by(id=tid).update({"platform_suspended": False})
    db.session.commit()

    rv = admin_tenant_client.post(
        f"/api/v1/platform/tenants/{tid}/suspend",
        json={"justification": "Test suspension gouvernance plateforme."},
    )
    assert rv.status_code == 200
    data = rv.get_json()
    assert data["decision"] in ("applied", "partially_applied")
    row = db.session.get(Company, tid)
    assert row is not None
    assert row.platform_suspended is True


def test_runbooks_list(admin_tenant_client):
    rv = admin_tenant_client.get("/api/v1/platform/runbooks")
    assert rv.status_code == 200
    data = rv.get_json()
    assert len(data["items"]) >= 1


def test_runbook_execution_verify(admin_tenant_client, sample_company, db):
    tid = sample_company.id
    Company.query.filter_by(id=tid).update({"platform_suspended": True})
    db.session.commit()

    rv = admin_tenant_client.post(
        "/api/v1/platform/runbooks/tenant_post_suspend_verify/executions",
        json={"tenant_id": tid},
    )
    assert rv.status_code == 200
    data = rv.get_json()
    assert data.get("verification_status") in ("passed", "partial", "failed")
    assert data.get("status") == "completed"
    assert "id" in data

    ex_id = data["id"]
    rv2 = admin_tenant_client.get(f"/api/v1/platform/runbooks/executions/{ex_id}")
    assert rv2.status_code == 200


def test_runbook_rollback_audit_carries_correlation_id(
    admin_tenant_client, sample_company, db
):
    """Execute + rollback + replay: même correlation_id sur le log rollback."""
    tid = sample_company.id
    Company.query.filter_by(id=tid).update({"platform_suspended": True})
    db.session.commit()

    cid = "test-corr-rb-rollback-1"
    headers = {**admin_tenant_client._h, "X-Correlation-Id": cid}

    rv = admin_tenant_client.post(
        "/api/v1/platform/runbooks/tenant_post_suspend_verify/executions",
        json={"tenant_id": tid},
        headers=headers,
    )
    assert rv.status_code == 200
    ex_id = rv.get_json()["id"]

    rv_rb = admin_tenant_client.post(
        f"/api/v1/platform/runbooks/executions/{ex_id}/rollback",
        json={},
        headers=headers,
    )
    assert rv_rb.status_code == 200

    rv_replay = admin_tenant_client.get(
        f"/api/v1/platform/audit-events/replay?correlation_id={cid}"
    )
    assert rv_replay.status_code == 200
    rep = rv_replay.get_json()
    assert rep["correlation_id"] == cid
    types = {e["action_type"] for e in rep["events"]}
    assert "platform_runbook_execution" in types
    assert "platform_runbook_rollback" in types


def test_suspend_denied_when_observe_only_grants(
    admin_tenant_client, sample_company, sample_admin_user, db
):
    """Un admin avec grants lecture seule ne peut pas suspendre (DEC-011)."""
    if not _platform_rbac_table_ready():
        pytest.skip("Migration 20260329_plat_admin_perm non appliquée")

    db.session.execute(
        delete(PlatformAdminPermissionGrant).where(
            PlatformAdminPermissionGrant.user_id == sample_admin_user.id
        )
    )
    db.session.add(
        PlatformAdminPermissionGrant(
            user_id=sample_admin_user.id,
            permission=PERM_OBSERVE_TENANT_READ,
        )
    )
    db.session.add(
        PlatformAdminPermissionGrant(
            user_id=sample_admin_user.id,
            permission=PERM_POLICY_EXPLAIN,
        )
    )
    db.session.commit()

    tid = sample_company.id
    Company.query.filter_by(id=tid).update({"platform_suspended": False})
    db.session.commit()

    rv = admin_tenant_client.post(
        f"/api/v1/platform/tenants/{tid}/suspend",
        json={"justification": "Test refus RBAC observe-only."},
    )
    assert rv.status_code == 403
    body = rv.get_json()
    assert body.get("decision") == "denied"
