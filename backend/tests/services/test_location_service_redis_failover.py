from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from services.geolocation.location import LocationService


class _FailingRedis:
    def hgetall(self, _key):  # type: ignore[no-untyped-def]
        raise ConnectionError("redis down")


@dataclass
class _DriverDto:
    id: int
    company_id: int


def test_store_location_marks_non_canonical_when_redis_unavailable() -> None:
    service = LocationService(redis_client_instance=_FailingRedis())
    now = datetime.now(UTC)

    accept_status, accept_reason, _received_at = service._store_location(
        driver_id=42,
        latitude=46.2,
        longitude=6.1,
        speed=10.0,
        heading=120.0,
        accuracy=5.0,
        source="raw",
        timestamp=now,
        location_mode="mission_live",
        recorded_at=now,
        sent_at=now,
        is_background=False,
        mission_id=123,
        company_id=1,
    )

    assert accept_status == "accepted_observability_only"
    assert accept_reason == "redis_unavailable_no_arbitration"


def test_update_driver_location_disables_canonical_outputs_when_non_canonical(
    monkeypatch,
) -> None:
    service = LocationService(redis_client_instance=None)
    driver_query_get_calls = {"count": 0}

    class _Repo:
        def find_by_id(self, _driver_id):  # type: ignore[no-untyped-def]
            return _DriverDto(id=99, company_id=1)

    monkeypatch.setattr(
        "services.geolocation.location.DriverRepository",
        _Repo,
    )
    monkeypatch.setattr(
        "services.geolocation.location.Driver",
        type(
            "DriverFake",
            (),
            {
                "query": type(
                    "Q",
                    (),
                    {
                        "get": lambda _cls, _id: driver_query_get_calls.__setitem__(
                            "count", driver_query_get_calls["count"] + 1
                        )
                    },
                )()
            },
        ),
    )
    monkeypatch.setattr(
        "services.geolocation.location.get_geofencing_service",
        lambda: type(
            "G", (), {"check_active_assignment_geofencing": lambda *a, **k: []}
        )(),
    )
    monkeypatch.setattr(service, "_log_trip_tracking", lambda **_kwargs: False)

    result = service.update_driver_location(
        driver_id=99,
        latitude=46.2,
        longitude=6.1,
        location_mode="availability_presence",
        recorded_at=datetime.now(UTC),
        sent_at=datetime.now(UTC),
        mission_id=None,
    )
    assert driver_query_get_calls["count"] == 0

    assert result.accept_status == "accepted_observability_only"
    assert result.accept_reason == "redis_unavailable_no_arbitration"
    assert result.should_fanout is False
    assert result.should_persist_db is False
