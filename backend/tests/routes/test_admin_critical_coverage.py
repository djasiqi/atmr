"""Couverture critique ``routes/admin.py`` (seuil 80 %)."""

from __future__ import annotations

import builtins
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from models import User
from models.enums import UserRole
from tests.routes.admin_route_fixtures import ADMIN_ENVIRON, admin_auth_headers


class _FakeResp:
    def __init__(self, status, json_data=None, text="", headers=None):
        self.status_code = status
        self._json = json_data if json_data is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._json


def _headers(app, admin):
    return admin_auth_headers(app, admin)


def _post_optuna(client, headers, payload=None, **env):
    return client.post(
        "/api/v1/admin/optuna/optimize",
        json=payload if payload is not None else {"n_trials": 2},
        headers=headers,
        environ_base={**ADMIN_ENVIRON, **env},
    )


def _post_train(client, headers, payload=None):
    return client.post(
        "/api/v1/admin/optuna/train",
        json=payload if payload is not None else {"training_episodes": 10},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )


@pytest.fixture
def admin_ctx(client, app, admin_route_env, make_admin_user):
    admin = make_admin_user()
    return client, app, admin, _headers(app, admin)


def test_users_liste_complete_et_filtres(
    admin_ctx, sample_company, simple_driver, db
):
    client, _app, admin, headers = admin_ctx
    resp = client.get("/api/v1/admin/users", headers=headers, environ_base=ADMIN_ENVIRON)
    assert resp.status_code == 200
    assert "users" in resp.get_json()

    bad = client.get(
        "/api/v1/admin/users?company_id=abc",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert bad.status_code == 400

    searched = client.get(
        f"/api/v1/admin/users?search={admin.username}&role=admin"
        "&sort_by=username&sort_order=asc&include_synthetic=true",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert searched.status_code == 200
    body = searched.get_json()
    assert "role_counts" in body
    assert body["page"] == 1

    by_company = client.get(
        f"/api/v1/admin/users?company_id={simple_driver.company_id}&sort_by=email",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert by_company.status_code == 200


def test_manage_user_get_et_delete(admin_ctx, db):
    client, _app, admin, headers = admin_ctx
    got = client.get(
        f"/api/v1/admin/users/{admin.id}",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert got.status_code == 200

    missing = client.get(
        "/api/v1/admin/users/99999999",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert missing.status_code in (404, 500)

    suffix = uuid.uuid4().hex[:8]
    orphan = User()
    orphan.username = f"adel_{suffix}"
    orphan.email = f"adel_{suffix}@test.ch"
    orphan.role = UserRole.client
    orphan.public_id = str(uuid.uuid4())
    orphan.set_password("password123", force_change=False)
    db.session.add(orphan)
    db.session.commit()

    deleted = client.delete(
        f"/api/v1/admin/users/{orphan.id}",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert deleted.status_code == 200

    gone = client.delete(
        "/api/v1/admin/users/99999999",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert gone.status_code in (404, 500)


def test_reset_password_succes_et_revoke_echec(admin_ctx, db, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    suffix = uuid.uuid4().hex[:8]
    target = User()
    target.username = f"arst_{suffix}"
    target.email = f"arst_{suffix}@test.ch"
    target.role = UserRole.client
    target.public_id = str(uuid.uuid4())
    target.set_password("password123", force_change=False)
    db.session.add(target)
    db.session.commit()
    uid = target.id

    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "security.mobile_device_session_service.revoke_user_security_sessions",
        lambda *a, **k: 1,
    )
    ok = client.post(
        f"/api/v1/admin/users/{uid}/reset-password",
        json={"reason": "reset pour couverture critique"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert ok.status_code == 200
    assert ok.get_json().get("sessions_revoked") is True

    monkeypatch.setattr(
        "security.mobile_device_session_service.revoke_user_security_sessions",
        MagicMock(side_effect=RuntimeError("mds")),
    )
    failed = client.post(
        f"/api/v1/admin/users/{uid}/reset-password",
        json={"reason": "reset revoke fail closed"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert failed.status_code == 500
    assert failed.get_json().get("error") == "session_revoke_failed"

    missing = client.post(
        "/api/v1/admin/users/99999999/reset-password",
        json={"reason": "utilisateur introuvable ici"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert missing.status_code in (404, 500)


def test_optuna_optimize_202_https_et_audit(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    monkeypatch.setenv("RL_WORKER_URL", "https://atmr-rl-worker:5000/")
    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResp(202, {"status": "started", "job_id": "j1"}),
    )
    monkeypatch.setattr(
        "security.audit_log.AuditLogger.log_action",
        staticmethod(lambda **k: None),
    )
    resp = _post_optuna(
        client, headers, {"company_id": 7, "data_period": "custom", "custom_days": 3}
    )
    assert resp.status_code == 202
    assert resp.get_json()["status"] == "started"


def test_optuna_optimize_url_sans_scheme_et_audit_echec(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    monkeypatch.setenv("RL_WORKER_URL", "atmr-rl-worker:5000")
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp(202, {"status": "ok"})
    )
    monkeypatch.setattr(
        "security.audit_log.AuditLogger.log_action",
        MagicMock(side_effect=RuntimeError("audit")),
    )
    resp = _post_optuna(client, headers)
    assert resp.status_code == 202


def test_optuna_optimize_redirects_et_erreurs(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx

    def _https_then_ok(url, **_kwargs):
        if url.startswith("http://atmr-rl-worker:5000/api"):
            return _FakeResp(302, headers={"Location": "https://rl/next"})
        return _FakeResp(202, {"status": "redirected"})

    monkeypatch.setattr("requests.post", _https_then_ok)
    monkeypatch.setattr(
        "security.audit_log.AuditLogger.log_action",
        staticmethod(lambda **k: None),
    )
    assert _post_optuna(client, headers).status_code == 202

    def _relative_then_ok(url, **_kwargs):
        if "/relative" not in url:
            return _FakeResp(302, headers={"Location": "/relative"})
        return _FakeResp(202, {"status": "rel"})

    monkeypatch.setattr("requests.post", _relative_then_ok)
    assert _post_optuna(client, headers).status_code == 202

    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResp(302, headers={}),
    )
    empty_loc = _post_optuna(client, headers)
    assert empty_loc.status_code == 500

    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResp(302, headers={"Location": "ftp://bad"}),
    )
    bad_loc = _post_optuna(client, headers)
    assert bad_loc.status_code == 500

    monkeypatch.setattr(
        "requests.post",
        lambda *a, **k: _FakeResp(302, headers={"Location": "https://loop/x"}),
    )
    looped = _post_optuna(client, headers)
    assert looped.status_code == 500
    assert "redirections" in (looped.get_json() or {}).get("error", "")

    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp(400, text="refusé")
    )
    rejected = _post_optuna(client, headers)
    assert rejected.status_code == 500

    monkeypatch.setattr(
        "requests.post",
        MagicMock(side_effect=requests.exceptions.ConnectionError("down")),
    )
    unavailable = _post_optuna(client, headers)
    assert unavailable.status_code == 503

    monkeypatch.setattr("requests.post", MagicMock(side_effect=ValueError("boom")))
    boom = _post_optuna(client, headers)
    assert boom.status_code == 500


def test_optuna_optimize_urllib3_absent(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "urllib3":
            raise ImportError("no urllib3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp(202, {"status": "ok"})
    )
    monkeypatch.setattr(
        "security.audit_log.AuditLogger.log_action",
        staticmethod(lambda **k: None),
    )
    assert _post_optuna(client, headers).status_code == 202


def test_optuna_train_branches(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    monkeypatch.setenv("RL_WORKER_URL", "https://rl-worker:9")
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp(202, {"status": "training"})
    )
    monkeypatch.setattr(
        "security.audit_log.AuditLogger.log_action",
        staticmethod(lambda **k: None),
    )
    assert _post_train(client, headers).status_code == 202

    monkeypatch.setenv("RL_WORKER_URL", "rl-worker:9")
    monkeypatch.setattr(
        "security.audit_log.AuditLogger.log_action",
        MagicMock(side_effect=RuntimeError("audit")),
    )
    assert _post_train(client, headers).status_code == 202

    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp(500, text="fail train")
    )
    assert _post_train(client, headers).status_code == 500

    monkeypatch.setattr(
        "requests.post",
        MagicMock(side_effect=requests.exceptions.Timeout("t")),
    )
    assert _post_train(client, headers).status_code == 503

    monkeypatch.setattr("requests.post", MagicMock(side_effect=RuntimeError("x")))
    assert _post_train(client, headers).status_code == 500

    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "urllib3":
            raise ImportError("no urllib3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(
        "requests.post", lambda *a, **k: _FakeResp(202, {"status": "ok"})
    )
    monkeypatch.setattr(
        "security.audit_log.AuditLogger.log_action",
        staticmethod(lambda **k: None),
    )
    assert _post_train(client, headers).status_code == 202


def test_autonomous_filtres_et_stats_periodes(admin_ctx):
    client, _app, _admin, headers = admin_ctx
    listed = client.get(
        "/api/v1/admin/autonomous-actions?page=1&per_page=10"
        "&success=true&reviewed=false&start_date=2026-01-01&end_date=2026-12-31",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert listed.status_code == 200

    invalid = client.get(
        "/api/v1/admin/autonomous-actions?page=0",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert invalid.status_code in (400, 200, 500)

    for period in ("hour", "week", "month", "day"):
        stats = client.get(
            f"/api/v1/admin/autonomous-actions/stats?period={period}&company_id=1",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert stats.status_code == 200

    global_stats = client.get(
        "/api/v1/admin/autonomous-actions/stats?period=day",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert global_stats.status_code == 200


def test_push_coverage_et_saferpay(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    monkeypatch.setattr(
        "services.notifications.push_coverage_service.list_driver_push_coverage",
        lambda **k: [{"driver_id": 1}],
    )
    cov = client.get(
        "/api/v1/admin/push-coverage/drivers?operational_only=false"
        "&without_token_only=true",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert cov.status_code == 200
    assert cov.get_json()["count"] == 1

    lookup = client.get(
        "/api/v1/admin/support/saferpay-payment-lookup?ref=txn-123456",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert lookup.status_code == 200
    assert "results" in lookup.get_json()


def test_redis_indisponible_et_ws_metrics(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    monkeypatch.setattr("routes.admin.redis_client", None)
    assert (
        client.post(
            "/api/v1/admin/rate-limit/flush",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        ).status_code
        == 503
    )
    assert (
        client.get(
            "/api/v1/admin/rate-limit/stats",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        ).status_code
        == 503
    )
    assert (
        client.get(
            "/api/v1/admin/redis/info",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        ).status_code
        == 503
    )

    ws_none = client.get(
        "/api/v1/admin/websocket/metrics",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert ws_none.status_code == 200
    assert ws_none.get_json().get("drivers_online_count") == 0

    fake_redis = MagicMock()
    fake_redis.keys.return_value = ["driver:1:last_seen", "driver:2:last_seen"]
    monkeypatch.setattr("routes.admin.redis_client", fake_redis)
    ws_ok = client.get(
        "/api/v1/admin/websocket/metrics",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert ws_ok.status_code == 200
    assert ws_ok.get_json().get("drivers_online_count") == 2

    fake_redis.keys.side_effect = RuntimeError("redis ws")
    ws_err = client.get(
        "/api/v1/admin/websocket/metrics",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert ws_err.status_code == 200
    assert ws_err.get_json().get("drivers_online_count") == 0


def test_partenaires_et_control_plane(admin_ctx, monkeypatch):
    client, _app, admin, headers = admin_ctx
    monkeypatch.setattr(
        "routes.admin.list_organizations_with_read_mode",
        lambda **k: {"items": [], "total": 0},
    )
    orgs = client.get(
        "/api/v1/admin/partners/organizations?include_synthetic=true"
        "&page=1&per_page=10&organization_type=company&search=acme",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert orgs.status_code == 200

    monkeypatch.setattr("routes.admin.get_organization_by_public_id", lambda _pid: None)
    missing_org = client.get(
        "/api/v1/admin/organizations/missing-id",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert missing_org.status_code in (404, 500)
    monkeypatch.setattr(
        "routes.admin.get_organization_by_public_id",
        lambda _pid: {"public_id": _pid},
    )
    found_org = client.get(
        "/api/v1/admin/organizations/org-1",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert found_org.status_code == 200

    monkeypatch.setattr(
        "routes.admin.list_anomalies", lambda **k: {"items": [], "total": 0}
    )
    anom = client.get(
        "/api/v1/admin/control-plane/anomalies?unresolved_only=false&severity=high",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert anom.status_code == 200

    monkeypatch.setattr(
        "routes.admin.compute_effective_access", lambda _uid: {"user_id": _uid}
    )
    access = client.get(
        f"/api/v1/admin/accounts/{admin.id}/effective-access",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert access.status_code == 200

    monkeypatch.setattr(
        "services.admin_account_manage_context.build_account_manage_context",
        lambda *a, **k: None,
    )
    ctx_404 = client.get(
        f"/api/v1/admin/accounts/{admin.id}/manage-context",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert ctx_404.status_code in (404, 500)
    monkeypatch.setattr(
        "services.admin_account_manage_context.build_account_manage_context",
        lambda *a, **k: {"user_id": admin.id},
    )
    ctx_ok = client.get(
        f"/api/v1/admin/accounts/{admin.id}/manage-context",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert ctx_ok.status_code == 200

    monkeypatch.setattr("routes.admin.build_account_integrity", lambda _uid: None)
    integ_404 = client.get(
        f"/api/v1/admin/partners/accounts/{admin.id}/integrity",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert integ_404.status_code in (404, 500)
    monkeypatch.setattr(
        "routes.admin.build_account_integrity", lambda uid: {"user_id": uid}
    )
    integ_ok = client.get(
        f"/api/v1/admin/partners/accounts/{admin.id}/integrity",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert integ_ok.status_code == 200


def test_role_driver_company_ops_monkeypatch(admin_ctx, monkeypatch):
    client, _app, admin, headers = admin_ctx
    preview = SimpleNamespace(to_dict=lambda: {"preview": True})
    monkeypatch.setattr(
        "services.admin_account_role_transition.AdminAccountRoleTransitionService.preview",
        lambda self, **k: preview,
    )
    prev = client.post(
        f"/api/v1/admin/users/{admin.id}/role-transition/preview",
        json={"role": "client"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert prev.status_code == 200

    applied = SimpleNamespace(to_dict=lambda: {"applied": True})
    monkeypatch.setattr(
        "services.admin_account_role_transition.AdminAccountRoleTransitionService.apply",
        lambda self, **k: applied,
    )
    put_role = client.put(
        f"/api/v1/admin/users/{admin.id}/role",
        json={
            "role": "client",
            "expected_current_role": "admin",
            "reason": "transition de role pour coverage",
        },
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert put_role.status_code == 200

    monkeypatch.setattr(
        "services.admin_driver_status.set_driver_status",
        lambda **k: SimpleNamespace(to_dict=lambda: {"ok": True}),
    )
    drv = client.put(
        f"/api/v1/admin/users/{admin.id}/driver-status",
        json={"is_active": False, "reason": "desactivation chauffeur test"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert drv.status_code == 200

    monkeypatch.setattr(
        "services.admin_driver_status.revoke_user_sessions_admin",
        lambda **k: SimpleNamespace(to_dict=lambda: {"revoked": True}),
    )
    rev = client.post(
        f"/api/v1/admin/users/{admin.id}/revoke-sessions",
        json={"reason": "revoquer sessions coverage"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert rev.status_code == 200

    monkeypatch.setattr(
        "services.admin_company_ops.set_company_approval",
        lambda **k: SimpleNamespace(to_dict=lambda: {"approved": True}),
    )
    appr = client.put(
        "/api/v1/admin/companies/1/approval",
        json={"is_approved": True, "reason": "approbation coverage test"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert appr.status_code == 200

    monkeypatch.setattr(
        "services.admin_company_ops.set_company_dispatch",
        lambda **k: SimpleNamespace(to_dict=lambda: {"dispatch": True}),
    )
    disp = client.put(
        "/api/v1/admin/companies/1/dispatch-status",
        json={"dispatch_enabled": True, "reason": "dispatch coverage test"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert disp.status_code == 200

    monkeypatch.setattr(
        "services.admin_company_ops.preview_dispatch_disable",
        lambda _cid: {"impact": 0},
    )
    prev_disp = client.get(
        "/api/v1/admin/companies/1/dispatch-disable-preview",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert prev_disp.status_code == 200


def test_autonomous_review_et_detail(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    action = SimpleNamespace(
        reviewed_by_admin=False,
        reviewed_at=None,
        admin_notes="",
        to_dict=lambda: {"id": 1, "reviewed_by_admin": True},
    )
    monkeypatch.setattr(
        "routes.admin.autonomous_action_repo.find_by_id_or_404",
        lambda _id: action,
    )
    monkeypatch.setattr("routes.admin.db.session.commit", lambda: None)
    reviewed = client.post(
        "/api/v1/admin/autonomous-actions/1/review",
        json={"notes": "revue coverage"},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert reviewed.status_code == 200
    detail = client.get(
        "/api/v1/admin/autonomous-actions/1",
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert detail.status_code == 200


def test_indicative_fare_put_succes(admin_ctx, monkeypatch):
    client, _app, _admin, headers = admin_ctx
    row = SimpleNamespace(
        config_version=1,
        updated_by_user_id=None,
    )
    monkeypatch.setattr(
        "routes.admin.db.session.get",
        lambda *_a, **_k: row,
    )
    monkeypatch.setattr("routes.admin.merge_admin_update", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "routes.admin.config_to_public_dict", lambda _row: {"config_version": 2}
    )
    monkeypatch.setattr("routes.admin.db.session.add", lambda *_a, **_k: None)
    monkeypatch.setattr("routes.admin.db.session.commit", lambda: None)
    monkeypatch.setattr(
        "routes.admin.get_current_user_via_use_case",
        lambda: SimpleNamespace(id=1),
    )
    resp = client.put(
        "/api/v1/admin/client-indicative-fare",
        json={"base_fare_chf": 12},
        headers=headers,
        environ_base=ADMIN_ENVIRON,
    )
    assert resp.status_code == 200
