from __future__ import annotations

from datetime import UTC, datetime

from flask_jwt_extended import create_access_token

from tests.e2e.helpers.e2e_helpers import create_test_company, create_test_driver


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    def get(self, key: str):
        value = self.store.get(key)
        return value.encode("utf-8") if value is not None else None

    def setex(self, key: str, _ttl: int, value: str):
        self.store[key] = value
        return True

    def incr(self, key: str):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def expire(self, _key: str, _ttl: int):
        return True

    def exists(self, key: str):
        return 1 if key in self.store else 0


def _driver_auth_headers(client, driver) -> dict[str, str]:
    with client.application.app_context():
        token = create_access_token(
            identity=str(driver.user.public_id),
            additional_claims={
                "role": "driver",
                "company_id": driver.company_id,
                "driver_id": driver.id,
                "aud": "atmr-api",
            },
        )
    return {"Authorization": f"Bearer {token}"}


def test_batch_requires_tracking_session_id(client, db, monkeypatch):
    company = create_test_company(db)
    driver = create_test_driver(db, company=company)
    monkeypatch.setattr("routes.driver.TRACKING_INGEST_ASYNC_ENABLED", True)
    response = client.post(
        "/api/v1/driver/me/locations/batch",
        json={
            "positions": [{"latitude": 46.2, "longitude": 6.1}],
        },
        headers=_driver_auth_headers(client, driver),
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "tracking_session_id_required"


def test_batch_single_active_session_conflict_returns_409(client, db, monkeypatch):
    company = create_test_company(db)
    driver = create_test_driver(db, company=company)
    fake_redis = _FakeRedis()
    fake_redis.setex(f"driver:{driver.id}:active_tracking_session", 1800, "sess-a")
    monkeypatch.setattr("routes.driver.TRACKING_INGEST_ASYNC_ENABLED", True)
    monkeypatch.setattr("routes.driver.redis_client", fake_redis)
    response = client.post(
        "/api/v1/driver/me/locations/batch",
        json={
            "tracking_session_id": "sess-b",
            "positions": [{"latitude": 46.2, "longitude": 6.1}],
        },
        headers=_driver_auth_headers(client, driver),
    )
    assert response.status_code == 409
    assert response.get_json()["error"] == "tracking_session_conflict"


def test_batch_duplicate_position_id_absorbed_and_ordered_by_sequence(
    client, db, monkeypatch
):
    company = create_test_company(db)
    driver = create_test_driver(db, company=company)
    fake_redis = _FakeRedis()
    monkeypatch.setattr("routes.driver.TRACKING_INGEST_ASYNC_ENABLED", True)
    monkeypatch.setattr("routes.driver.redis_client", fake_redis)
    calls: list[dict] = []

    def _enqueue_tracking_event(**kwargs):
        calls.append(kwargs["payload"])
        return {"queued": True, "trace_id": "t-1"}

    monkeypatch.setattr("routes.driver.enqueue_tracking_event", _enqueue_tracking_event)
    now = datetime.now(UTC).isoformat()
    response = client.post(
        "/api/v1/driver/me/locations/batch",
        json={
            "tracking_session_id": "sess-1",
            "batch_id": "batch-1",
            "positions": [
                {
                    "latitude": 46.2,
                    "longitude": 6.1,
                    "position_id": "p-2",
                    "sequence_id": 2,
                    "recorded_at": now,
                },
                {
                    "latitude": 46.3,
                    "longitude": 6.2,
                    "position_id": "p-1",
                    "sequence_id": 1,
                    "recorded_at": now,
                },
                {
                    "latitude": 46.3,
                    "longitude": 6.2,
                    "position_id": "p-1",
                    "sequence_id": 1,
                    "recorded_at": now,
                },
            ],
        },
        headers=_driver_auth_headers(client, driver),
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["accepted_count"] >= 2
    assert len(calls) == 2
    assert calls[0]["sequence_id"] == 1
    assert calls[1]["sequence_id"] == 2
