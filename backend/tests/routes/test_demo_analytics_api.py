from __future__ import annotations


def test_demo_analytics_accepts_allowed_event(client):
    response = client.post(
        "/api/v1/demo_access/analytics",
        json={"event": "demo_session_start", "payload": {"role": "transporteur"}},
    )
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def test_demo_analytics_rejects_unknown_event(client):
    response = client.post(
        "/api/v1/demo_access/analytics",
        json={"event": "unknown_demo_event"},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["ok"] is False
    assert data["code"] == "invalid_event"

