class _DummyDemoRequest:
    def __init__(self, **kwargs):
        self.id = kwargs.pop("id", 1)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _payload(**overrides):
    data = {
        "name": "Marie Curie",
        "email": "marie.curie@example.com",
        "phone": "+41221234567",
        "organization": "Clinique Test",
        "organization_type": "clinic",
        "use_case": "planning_dispatch",
        "volume_range": "20_100",
        "integration_required": "yes",
        "integration_system": "ERP Test",
        "timing": "immediate",
        "preferred_slot": "this_week",
        "preferred_period": "morning",
        "comment": "Besoin de cadrer le deploiement rapidement.",
        "privacy_consent": True,
        "honeypot": "",
        "form_started_at_ms": 1,
    }
    data.update(overrides)
    return data


def _patch_db_layer(monkeypatch):
    from routes import demo_requests as demo_module

    monkeypatch.setattr("routes.demo_requests.DemoRequest", _DummyDemoRequest)

    def _add(obj):
        if not getattr(obj, "id", None):
            obj.id = 1

    monkeypatch.setattr(demo_module.db.session, "add", _add)
    monkeypatch.setattr(demo_module.db.session, "flush", lambda: None)
    monkeypatch.setattr(demo_module.db.session, "commit", lambda: None)
    monkeypatch.setattr(demo_module.db.session, "rollback", lambda: None)


def test_demo_request_create_success(client, monkeypatch):
    _patch_db_layer(monkeypatch)
    monkeypatch.setattr(
        "routes.demo_requests.send_demo_notification",
        lambda payload: {"ok": True, "provider": "smtp", "payload": payload},
    )

    response = client.post(
        "/api/v1/demo-requests",
        json=_payload(),
        environ_overrides={"REMOTE_ADDR": "10.30.0.1"},
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["ok"] is True
    assert body["request_id"] == 1
    assert body["priority"] in {"high", "medium", "standard"}


def test_demo_request_validation_error_without_consent(client, monkeypatch):
    _patch_db_layer(monkeypatch)
    response = client.post(
        "/api/v1/demo-requests",
        json=_payload(privacy_consent=False),
        environ_overrides={"REMOTE_ADDR": "10.30.0.2"},
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "validation_error"


def test_demo_request_honeypot_is_ignored(client, monkeypatch):
    was_called = {"value": False}

    def _mark_called(payload):
        _ = payload
        was_called["value"] = True
        return {"ok": True}

    monkeypatch.setattr("routes.demo_requests.send_demo_notification", _mark_called)

    response = client.post(
        "/api/v1/demo-requests",
        json=_payload(honeypot="https://spam.example"),
        environ_overrides={"REMOTE_ADDR": "10.30.0.3"},
    )
    assert response.status_code == 201
    assert was_called["value"] is False
