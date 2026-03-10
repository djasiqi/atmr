from types import SimpleNamespace


def _payload(**overrides):
    data = {
        "category": "support",
        "name": "Marie Curie",
        "email": "marie.curie@example.com",
        "organization": "Clinique Test",
        "phone": "+41221234567",
        "subject_detail": "bug",
        "message": "Bonjour, nous avons besoin d'aide sur notre configuration.",
        "privacy_consent": True,
        "website": "",
        "client_request_id": "req-123",
    }
    data.update(overrides)
    return data


class _DummyQuery:
    def __init__(self):
        self.first_value = None
        self.update_value = 1

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.first_value

    def update(self, *args, **kwargs):
        return self.update_value


class _DummyContactRequest:
    query = _DummyQuery()

    def __init__(self, **kwargs):
        self.id = kwargs.pop("id", 1)
        for key, value in kwargs.items():
            setattr(self, key, value)


def _patch_db_layer(monkeypatch):
    from routes import contact as contact_module

    query = _DummyQuery()
    _DummyContactRequest.query = query
    monkeypatch.setattr("routes.contact.ContactRequest", _DummyContactRequest)

    store = {"last": None}

    def _add(obj):
        store["last"] = obj
        if not getattr(obj, "id", None):
            obj.id = 1

    monkeypatch.setattr(contact_module.db.session, "add", _add)
    monkeypatch.setattr(contact_module.db.session, "commit", lambda: None)
    monkeypatch.setattr(contact_module.db.session, "rollback", lambda: None)
    return store


def _patch_pipeline(monkeypatch, duplicate=None, rate_limited=False):
    monkeypatch.setattr("routes.contact.find_recent_duplicate", lambda *args, **kwargs: duplicate)
    monkeypatch.setattr("routes.contact.in_cooldown", lambda *args, **kwargs: False)
    monkeypatch.setattr("routes.contact.hit_rate_limit", lambda *args, **kwargs: rate_limited)
    monkeypatch.setattr("routes.contact._acquire_email_send_lock", lambda *args, **kwargs: True)
    monkeypatch.setattr("routes.contact._has_sent_for_hash", lambda *args, **kwargs: False)
    monkeypatch.setattr("routes.contact._mark_email_sending", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        "routes.contact._extract_optional_auth_context",
        lambda: {
            "user_id": None,
            "user_public_id": None,
            "user_role": None,
            "company_id": None,
            "institution_id": None,
        },
    )
    monkeypatch.setattr("routes.contact._hash_ip", lambda _ip: "ip_hash_test")


def test_contact_request_create_success_all_categories(client, db, monkeypatch):
    _patch_db_layer(monkeypatch)
    _patch_pipeline(monkeypatch)
    calls = []

    def _send(payload):
        calls.append(payload)
        return {"ok": True}

    monkeypatch.setattr("routes.contact.send_contact_notification", _send)

    payloads = [
        _payload(category="support"),
        _payload(
            category="institution",
            organization_type="ems",
            integration_required="yes",
            integration_system="DPI",
        ),
        _payload(category="transport"),
        _payload(
            category="demo",
            organization_type="institution",
            timing="immediate",
            preferred_slot="this_week",
        ),
        _payload(category="billing"),
        _payload(category="family", organization=""),
    ]

    for payload in payloads:
        response = client.post("/api/v1/contact/requests", json=payload)
        assert response.status_code == 200
        body = response.get_json()
        assert body["ok"] is True
        assert body["trace_id"].startswith("ct_")

    assert len(calls) == 6


def test_contact_request_validation_error_institution_missing_required(client, monkeypatch):
    _patch_db_layer(monkeypatch)
    _patch_pipeline(monkeypatch)
    response = client.post(
        "/api/v1/contact/requests",
        json=_payload(category="institution", organization_type=None, integration_required=None),
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "validation_error"


def test_contact_request_silent_spam_no_email(client, monkeypatch):
    store = _patch_db_layer(monkeypatch)
    _patch_pipeline(monkeypatch)
    was_called = {"value": False}

    def _mark_called(payload):
        _ = payload
        was_called["value"] = True
        return {"ok": False}

    monkeypatch.setattr("routes.contact.send_contact_notification", _mark_called)

    response = client.post("/api/v1/contact/requests", json=_payload(website="https://spam.example"))
    assert response.status_code == 200
    assert was_called["value"] is False
    assert store["last"].status == "spam"


def test_contact_request_dedupe_returns_existing_trace(client, monkeypatch):
    _patch_db_layer(monkeypatch)
    existing = SimpleNamespace(trace_id="ct_EXISTING", status="new")
    _patch_pipeline(monkeypatch, duplicate=existing)
    monkeypatch.setattr("routes.contact.send_contact_notification", lambda payload: {"ok": True})

    response = client.post("/api/v1/contact/requests", json=_payload())
    assert response.status_code == 200
    assert response.get_json()["trace_id"] == "ct_EXISTING"


def test_contact_request_rate_limit_by_category(client, monkeypatch):
    _patch_db_layer(monkeypatch)
    monkeypatch.setattr("routes.contact.in_cooldown", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "routes.contact.hit_rate_limit",
        lambda _ip_hash, category: category == "demo",
    )
    response = client.post(
        "/api/v1/contact/requests",
        json=_payload(
            category="demo",
            organization_type="institution",
            timing="immediate",
            preferred_slot="this_week",
        ),
    )
    assert response.status_code == 429


def test_contact_request_concurrency_guard_suppresses_duplicate_email(client, monkeypatch):
    _patch_db_layer(monkeypatch)
    _patch_pipeline(monkeypatch)
    monkeypatch.setattr("routes.contact._has_sent_for_hash", lambda *args, **kwargs: True)
    called = {"count": 0}

    def _send(payload):
        called["count"] += 1
        return {"ok": True}

    monkeypatch.setattr("routes.contact.send_contact_notification", _send)
    response = client.post("/api/v1/contact/requests", json=_payload())
    assert response.status_code == 200
    assert called["count"] == 0

