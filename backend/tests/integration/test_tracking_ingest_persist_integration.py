"""Intégration : persistance Kafka ingest_consumer → DB + message processed + dédup."""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from application.drivers.update_driver_location import UpdateDriverLocationUseCase
from models import Driver
from models.enums import AssignmentStatus
from models.trip_tracking import TripTracking
from repositories.assignment_repository import AssignmentRepository
from services.geolocation.location import LocationService, TRIP_TRACKING_ASSIGNMENT_STATUSES
from services.tracking.ingest_consumer import TrackingIngestConsumer
from services.tracking.kafka_topics import TOPIC_DRIVER_LOCATION_PROCESSED
from tests.factories import create_assignment_with_booking_driver


class _TrackingIntegrationRedis:
    """Redis minimal : SET NX (dédup event_id) + hash canonique LocationService."""

    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}

    def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
        _ = ex
        if nx:
            if key in self._strings:
                return None
            self._strings[key] = str(value)
            return True
        self._strings[key] = str(value)
        return True

    def get(self, key: str):
        val = self._strings.get(key)
        return val.encode("utf-8") if val is not None else None

    def hgetall(self, key: str):
        h = self._hashes.get(key, {})
        return {k.encode("utf-8"): v.encode("utf-8") for k, v in h.items()}

    def hset(self, key: str, mapping: dict[str, Any] | None = None, **kwargs: Any):
        payload = dict(mapping or {})
        payload.update(kwargs)
        bucket = self._hashes.setdefault(key, {})
        bucket.update({str(k): str(v) for k, v in payload.items()})
        return len(payload)

    def expire(self, _key: str, _ttl: int):
        return True

    def geoadd(self, _key: str, *_values: Any, **_kwargs: Any):
        return 1

    def xadd(self, _key: str, _fields: dict[str, Any], **_kwargs: Any):
        return b"1-0"

    def lpush(self, _key: str, _value: str):
        return 1

    def ltrim(self, _key: str, _start: int, _end: int):
        return True

    def lrange(self, _key: str, _start: int, _end: int):
        return []

    def ping(self):
        return True


def _kafka_record(message: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        topic="driver.location.raw",
        partition=0,
        offset=1,
        key=f"driver_{message['driver_id']}",
        timestamp=int(message.get("received_at_ms", 1_700_000_000_000)),
        value=message,
    )


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


@pytest.mark.integration
def test_ingest_consumer_persist_and_dedup_integration(app, db, sample_company, monkeypatch):
    """Raw Kafka → persist → processed ; rejeu même event_id → skipped sans nouvelle ligne."""
    assignment = create_assignment_with_booking_driver(
        company=sample_company,
        status=AssignmentStatus.EN_ROUTE_PICKUP,
    )
    driver = assignment.driver
    booking = assignment.booking
    assert driver is not None and booking is not None

    booking.driver_id = driver.id
    db.session.flush()

    active_assignments = [
        dto
        for dto in AssignmentRepository().find_by_driver_id(driver.id)
        if dto.status in TRIP_TRACKING_ASSIGNMENT_STATUSES
    ]
    assert len(active_assignments) >= 1

    fake_redis = _TrackingIntegrationRedis()
    monkeypatch.setattr("ext.redis_client", fake_redis)
    monkeypatch.setattr("services.geolocation.location._location_service_instance", None)
    loc_svc = LocationService(redis_client_instance=fake_redis)
    monkeypatch.setattr("services.geolocation.location.get_location_service", lambda: loc_svc)

    _real_update_location = LocationService.update_driver_location

    def _update_location_with_test_session(self, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("db_session", db.session)
        return _real_update_location(self, **kwargs)

    monkeypatch.setattr(
        LocationService, "update_driver_location", _update_location_with_test_session
    )
    # Évite un app_context imbriqué qui isole la session pytest (savepoint).
    monkeypatch.setattr(app, "app_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr("celery_app.get_flask_app", lambda: app)
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", True
    )

    uc_executions: list[Any] = []
    real_execute = UpdateDriverLocationUseCase.execute

    def _spy_execute(self, cmd):  # type: ignore[no-untyped-def]
        uc_executions.append(cmd)
        return real_execute(self, cmd)

    monkeypatch.setattr(UpdateDriverLocationUseCase, "execute", _spy_execute)

    consumer = TrackingIngestConsumer()
    published: list[dict[str, Any]] = []

    def _capture_publish(*, topic: str, key: str, message: dict[str, Any], retry_count: int):
        published.append(
            {"topic": topic, "key": key, "message": message, "retry_count": retry_count}
        )

    monkeypatch.setattr(consumer, "_publish_with_ack", _capture_publish)
    monkeypatch.setattr(consumer, "_commit_current", lambda: None)

    location_event_id = f"integration-{uuid.uuid4()}"
    recorded_at = datetime.now(UTC).isoformat()
    raw_message = {
        "driver_id": driver.id,
        "company_id": sample_company.id,
        "location_event_id": location_event_id,
        "received_at_ms": 1_718_000_000_000,
        "source": "http_async",
        "payload": {
            "latitude": 46.2044,
            "longitude": 6.1432,
            "recorded_at": recorded_at,
            "location_mode": "mission_live",
            "mission_id": booking.id,
        },
    }
    record = _kafka_record(raw_message)

    trip_count_before = TripTracking.query.filter_by(driver_id=driver.id).count()
    driver_row = db.session.get(Driver, driver.id)
    assert driver_row is not None
    last_pos_before = driver_row.last_position_update

    assert consumer._process_record(record) is True
    db.session.flush()

    assert len(published) == 1
    first_processed = published[0]
    persist_result = first_processed["message"].get("persist_result")
    assert isinstance(persist_result, dict), first_processed["message"]

    assert len(uc_executions) == 1
    assert uc_executions[0].driver_id == driver.id
    assert uc_executions[0].metrics_transport == "kafka"
    assert uc_executions[0].location_event_id == location_event_id

    trip_count_after_first = TripTracking.query.filter_by(driver_id=driver.id).count()
    assert trip_count_after_first == trip_count_before + 1

    db.session.expire_all()
    driver_after = db.session.get(Driver, driver.id)
    assert driver_after is not None
    assert driver_after.last_position_update is not None
    if last_pos_before is not None:
        assert _aware(driver_after.last_position_update) >= _aware(last_pos_before)
    assert driver_after.latitude == pytest.approx(46.2044, abs=0.01)
    assert driver_after.longitude == pytest.approx(6.1432, abs=0.01)

    assert first_processed["topic"] == TOPIC_DRIVER_LOCATION_PROCESSED
    assert first_processed["key"] == f"driver_{driver.id}"
    assert persist_result.get("accept_status") in (
        "accepted_canonical",
        "accepted_observability_only",
    )
    assert persist_result.get("dedup_skipped") is False
    assert first_processed["message"].get("location_event_id") == location_event_id

    # --- Rejeu identique (même location_event_id) ---
    assert consumer._process_record(record) is True
    db.session.flush()

    assert len(uc_executions) == 2
    assert uc_executions[1].location_event_id == location_event_id

    trip_count_after_second = TripTracking.query.filter_by(driver_id=driver.id).count()
    assert trip_count_after_second == trip_count_after_first

    rows_for_assignment = TripTracking.query.filter_by(
        assignment_id=assignment.id,
        driver_id=driver.id,
    ).count()
    assert rows_for_assignment == 1

    assert len(published) == 2
    second_persist = published[1]["message"].get("persist_result")
    assert isinstance(second_persist, dict)
    assert second_persist.get("dedup_skipped") is True
    assert second_persist.get("accept_status") == "skipped"
