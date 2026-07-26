"""F-04 — feature flags : refus anonyme / non-admin, succès admin."""

from __future__ import annotations


def test_anonymous_enable_401(client, monkeypatch):
    monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
    resp = client.post("/api/feature-flags/ml/enable", json={"percentage": 10})
    assert resp.status_code == 401


def test_company_enable_403(client, auth_headers, monkeypatch):
    monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
    resp = client.post(
        "/api/feature-flags/ml/enable",
        json={"percentage": 10},
        headers=auth_headers,
    )
    assert resp.status_code == 403


def test_admin_enable_200(client, admin_headers, monkeypatch):
    monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
    resp = client.post(
        "/api/feature-flags/ml/enable",
        json={"percentage": 15},
        headers=admin_headers,
    )
    assert resp.status_code == 200
    data = resp.get_json() or {}
    assert data.get("success") is True
    assert data["status"]["config"]["ML_TRAFFIC_PERCENTAGE"] == 15
