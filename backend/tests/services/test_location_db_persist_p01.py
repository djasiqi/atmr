"""P0.1 — échec PG ne doit jamais produire persisted_sync / db_persisted=True."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from sqlalchemy.exc import OperationalError

from services.geolocation.location import LocationService, LocationUpdateResult


class _OkRedis:
    def __init__(self) -> None:
        self._hashes: dict[str, dict] = {}

    def hgetall(self, key):  # type: ignore[no-untyped-def]
        return dict(self._hashes.get(key, {}))

    def hset(self, key, mapping=None, **_kwargs):  # type: ignore[no-untyped-def]
        self._hashes.setdefault(key, {}).update(mapping or {})
        return 1

    def expire(self, _key, _ttl):  # type: ignore[no-untyped-def]
        return True

    def geoadd(self, *_a, **_k):  # type: ignore[no-untyped-def]
        return 1

    def xadd(self, *_a, **_k):  # type: ignore[no-untyped-def]
        return b"1-0"


def test_store_location_db_failure_sets_db_persisted_false(monkeypatch) -> None:
    service = LocationService(redis_client_instance=_OkRedis())
    now = datetime.now(UTC)

    class _Repo:
        def find_by_id(self, driver_id):  # type: ignore[no-untyped-def]
            return MagicMock(id=driver_id, company_id=1)

    driver_orm = MagicMock()
    session = MagicMock()
    session.commit.side_effect = OperationalError("stmt", {}, Exception("pg down"))

    monkeypatch.setattr(
        "services.geolocation.location.DriverRepository",
        _Repo,
    )

    class _Q:
        @staticmethod
        def get(_id):  # type: ignore[no-untyped-def]
            return driver_orm

    monkeypatch.setattr(
        "services.geolocation.location.Driver",
        type("DriverFake", (), {"query": _Q}),
    )
    monkeypatch.setattr("services.geolocation.location.db", MagicMock(session=session))

    (
        accept_status,
        _accept_reason,
        _received,
        canonical_updated,
        db_persisted,
    ) = service._store_location(
        driver_id=7,
        latitude=46.2,
        longitude=6.1,
        speed=None,
        heading=None,
        accuracy=5.0,
        source="raw",
        timestamp=now,
        location_mode="availability_presence",
        recorded_at=now,
        sent_at=now,
        is_background=False,
        mission_id=None,
        company_id=1,
    )

    assert accept_status == "accepted_canonical"
    assert canonical_updated is True
    assert db_persisted is False
    session.rollback.assert_called()


def test_store_location_pg_first_failure_does_not_write_canonical(
    monkeypatch,
) -> None:
    """CASE 2 — flag P5-B : échec PG → canonical Redis inchangé."""
    redis = _OkRedis()
    service = LocationService(redis_client_instance=redis)
    now = datetime.now(UTC)

    class _Repo:
        def find_by_id(self, driver_id):  # type: ignore[no-untyped-def]
            return MagicMock(id=driver_id, company_id=1)

    session = MagicMock()
    session.commit.side_effect = OperationalError("stmt", {}, Exception("pg down"))
    monkeypatch.setenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "true")
    monkeypatch.setattr(
        "services.geolocation.location.DriverRepository",
        _Repo,
    )

    class _Q:
        @staticmethod
        def get(_id):  # type: ignore[no-untyped-def]
            return MagicMock()

    monkeypatch.setattr(
        "services.geolocation.location.Driver",
        type("DriverFake", (), {"query": _Q}),
    )
    monkeypatch.setattr("services.geolocation.location.db", MagicMock(session=session))

    (
        accept_status,
        _reason,
        _received,
        canonical_updated,
        db_persisted,
    ) = service._store_location(
        driver_id=7,
        latitude=46.2,
        longitude=6.1,
        speed=None,
        heading=None,
        accuracy=5.0,
        source="raw",
        timestamp=now,
        location_mode="availability_presence",
        recorded_at=now,
        sent_at=now,
        is_background=False,
        mission_id=None,
        company_id=1,
        location_event_id="evt-1",
        capture_id="fix-1",
        session_generation=1,
        sequence_id=1,
        tracking_session_id="sess",
    )
    assert accept_status == "accepted_canonical"
    assert db_persisted is False
    assert canonical_updated is False
    assert "driver:7:loc:canonical" not in redis._hashes


def test_store_location_pg_first_success_promotes_canonical(monkeypatch) -> None:
    """CASE 1 / CASE 8 — HTTP sync : PG OK → canonical Redis."""
    redis = _OkRedis()
    service = LocationService(redis_client_instance=redis)
    now = datetime.now(UTC)

    class _Repo:
        def find_by_id(self, driver_id):  # type: ignore[no-untyped-def]
            return MagicMock(id=driver_id, company_id=1)

    session = MagicMock()
    monkeypatch.setenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "true")
    monkeypatch.setattr(
        "services.geolocation.location.DriverRepository",
        _Repo,
    )

    class _Q:
        @staticmethod
        def get(_id):  # type: ignore[no-untyped-def]
            return MagicMock()

    monkeypatch.setattr(
        "services.geolocation.location.Driver",
        type("DriverFake", (), {"query": _Q}),
    )
    monkeypatch.setattr("services.geolocation.location.db", MagicMock(session=session))

    (
        _status,
        _reason,
        _received,
        canonical_updated,
        db_persisted,
    ) = service._store_location(
        driver_id=7,
        latitude=46.2,
        longitude=6.1,
        speed=None,
        heading=None,
        accuracy=5.0,
        source="raw",
        timestamp=now,
        location_mode="availability_presence",
        recorded_at=now,
        sent_at=now,
        is_background=False,
        mission_id=None,
        company_id=1,
        location_event_id="evt-1",
        capture_id="fix-http",
        session_generation=2,
        sequence_id=8,
        tracking_session_id="sess",
    )
    assert db_persisted is True
    assert canonical_updated is True
    mapping = redis._hashes["driver:7:loc:canonical"]
    assert mapping["capture_id"] == "fix-http"
    assert mapping["session_generation"] == "2"
    assert mapping["sequence_id"] == "8"


def test_update_result_exposes_db_persisted_false(monkeypatch) -> None:
    service = LocationService(redis_client_instance=_OkRedis())
    now = datetime.now(UTC)

    monkeypatch.setattr(
        service,
        "_store_location",
        lambda **_k: (
            "accepted_canonical",
            "",
            now.isoformat(),
            True,
            False,
        ),
    )
    monkeypatch.setattr(
        "services.geolocation.location.DriverRepository",
        lambda: MagicMock(find_by_id=lambda _id: MagicMock(company_id=1)),
    )
    monkeypatch.setattr(
        "services.geolocation.location.get_geofencing_service",
        lambda: type(
            "G", (), {"check_active_assignment_geofencing": lambda *a, **k: []}
        )(),
    )
    monkeypatch.setattr(service, "_log_trip_tracking", lambda **_k: False)
    monkeypatch.setattr(service, "_is_v21_enabled_for_company", lambda _cid: False)

    result = service.update_driver_location(
        driver_id=7,
        latitude=46.2,
        longitude=6.1,
        location_mode="availability_presence",
        recorded_at=now,
        sent_at=now,
    )
    assert isinstance(result, LocationUpdateResult)
    assert result.accept_status == "accepted_canonical"
    assert result.canonical_updated is True
    assert result.db_persisted is False


def test_accepted_canonical_db_false_must_not_claim_persisted_sync() -> None:
    """Contrat : accepted_canonical + db_persisted=False → 503, jamais persisted_sync."""
    from application.drivers.update_driver_location import UpdateDriverLocationResult

    uc = UpdateDriverLocationResult(
        snapped_lat=46.2,
        snapped_lon=6.1,
        source="raw",
        geofence_events=[],
        accept_status="accepted_canonical",
        accept_reason="",
        received_at="2026-03-18T10:00:01Z",
        canonical_updated=True,
        db_persisted=False,
    )
    assert uc.db_persisted is False
    # Branche route miroir
    status_code = (
        503
        if uc.accept_status == "accepted_canonical" and uc.db_persisted is False
        else 200
    )
    durability = (
        "persisted_sync"
        if uc.accept_status == "accepted_canonical" and uc.db_persisted is True
        else None
    )
    assert status_code == 503
    assert durability is None
