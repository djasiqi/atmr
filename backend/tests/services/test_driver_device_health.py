"""Tests services driver_device_health."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from services.driver_device_health import (
    ingest_driver_device_health,
    purge_old_device_health_events,
    read_driver_device_health_snapshot,
    resolve_tracking_display_status,
)


def test_resolve_tracking_display_status_live():
    assert (
        resolve_tracking_display_status(
            location_status="live",
            health_snapshot=None,
        )
        == "live"
    )


def test_resolve_tracking_display_status_stale():
    assert (
        resolve_tracking_display_status(
            location_status="stale",
            health_snapshot={"battery_optimized": "1"},
        )
        == "stale"
    )


def test_resolve_tracking_display_status_degraded_constrained():
    assert (
        resolve_tracking_display_status(
            location_status="offline",
            health_snapshot={
                "battery_optimized": "1",
                "constraint_reason": "battery_optimized",
            },
        )
        == "degraded_constrained"
    )


def test_resolve_tracking_display_status_offline_unknown():
    assert (
        resolve_tracking_display_status(
            location_status="offline",
            health_snapshot=None,
        )
        == "offline_unknown"
    )


def test_ingest_driver_device_health_persists_and_writes_redis(db, sample_driver):
    mock_redis = MagicMock()
    mock_event = MagicMock()
    with (
        patch("services.driver_device_health.redis_client", mock_redis),
        patch(
            "services.geolocation.device_health.write_device_health", return_value=True
        ),
        patch(
            "services.monitoring.driver_device_health_metrics.record_device_health_report"
        ),
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent",
            return_value=mock_event,
        ),
        patch("services.driver_device_health.db.session") as mock_session,
    ):
        from services.driver_device_health import ingest_driver_device_health

        snapshot = ingest_driver_device_health(
            sample_driver.id,
            {
                "manufacturer": "Xiaomi",
                "platform": "android",
                "tracking_active": True,
                "last_fix_age_seconds": 20,
            },
        )

    assert snapshot["manufacturer"] == "Xiaomi"
    mock_session.add.assert_called_once_with(mock_event)
    mock_session.commit.assert_called_once()
    assert mock_redis.hset.called


def test_ingest_driver_device_health_parses_diagnostic_lot1_fields(db, sample_driver):
    """Lot 1 : versions + signaux iOS background remontés dans le snapshot."""
    mock_redis = MagicMock()
    mock_event = MagicMock()
    with (
        patch("services.driver_device_health.redis_client", mock_redis),
        patch(
            "services.geolocation.device_health.write_device_health", return_value=True
        ),
        patch(
            "services.monitoring.driver_device_health_metrics.record_device_health_report"
        ),
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent",
            return_value=mock_event,
        ),
        patch("services.driver_device_health.db.session"),
    ):
        snapshot = ingest_driver_device_health(
            sample_driver.id,
            {
                "manufacturer": "Apple",
                "platform": "ios",
                "tracking_active": True,
                "last_fix_age_seconds": 6000,
                "location_fix_age_seconds": 15,
                "app_version": "1.42.3",
                "os_version": "17.4",
                "task_invoke_age_seconds": 12,
                "native_last_fix_age_seconds": 999,
                "observability_class": "RUNTIME_ONLY",
                "native_task_running": True,
                "ios_accuracy_authorization": "reduced",
                "ios_low_power_mode": True,
                "ios_background_refresh_status": "denied",
            },
        )

    assert snapshot["app_version"] == "1.42.3"
    assert snapshot["os_version"] == "17.4"
    # Prefer location_fix / task_invoke ; native_last_fix kept as compat alias
    assert snapshot["location_fix_age_seconds"] == "15"
    assert snapshot["last_fix_age_seconds"] == "15"
    assert snapshot["task_invoke_age_seconds"] == "12"
    assert snapshot["native_last_fix_age_seconds"] == "12"
    assert snapshot["observability_class"] == "RUNTIME_ONLY"
    assert snapshot["native_task_running"] == "1"
    assert snapshot["ios_accuracy_authorization"] == "reduced"
    assert snapshot["ios_low_power_mode"] == "1"
    assert snapshot["ios_background_refresh_status"] == "denied"


def test_ingest_prefers_legacy_native_last_fix_when_task_invoke_absent(
    db, sample_driver
):
    mock_redis = MagicMock()
    mock_event = MagicMock()
    with (
        patch("services.driver_device_health.redis_client", mock_redis),
        patch(
            "services.geolocation.device_health.write_device_health", return_value=True
        ),
        patch(
            "services.monitoring.driver_device_health_metrics.record_device_health_report"
        ),
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent",
            return_value=mock_event,
        ),
        patch("services.driver_device_health.db.session"),
    ):
        snapshot = ingest_driver_device_health(
            sample_driver.id,
            {
                "platform": "android",
                "native_last_fix_age_seconds": 42,
                "last_fix_age_seconds": 7,
            },
        )

    assert snapshot["task_invoke_age_seconds"] == "42"
    assert snapshot["native_last_fix_age_seconds"] == "42"
    assert snapshot["location_fix_age_seconds"] == "7"
    assert snapshot["last_fix_age_seconds"] == "7"


def test_canary_observability_backend_compat_new_and_legacy(db, sample_driver):
    """Canary OBSERVABILITY — nouveaux champs prioritaires ; legacy sans exception."""
    mock_redis = MagicMock()
    mock_event = MagicMock()

    def _ingest(payload: dict):
        return ingest_driver_device_health(sample_driver.id, payload)

    with (
        patch("services.driver_device_health.redis_client", mock_redis),
        patch(
            "services.geolocation.device_health.write_device_health", return_value=True
        ),
        patch(
            "services.monitoring.driver_device_health_metrics.record_device_health_report"
        ),
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent",
            return_value=mock_event,
        ),
        patch("services.driver_device_health.db.session"),
    ):
        new_snap = _ingest(
            {
                "platform": "android",
                "location_fix_age_seconds": 11,
                "last_fix_age_seconds": 999,
                "task_invoke_age_seconds": 22,
                "native_last_fix_age_seconds": 888,
                "observability_class": "PIPELINE",
                "oldest_queue_item_age_seconds": 200,
                "persistence_lag_seconds": 300,
            }
        )
        legacy_snap = _ingest(
            {
                "platform": "android",
                "last_fix_age_seconds": 9,
                "native_last_fix_age_seconds": 33,
            }
        )
        empty_snap = _ingest({"platform": "android"})

    assert new_snap["location_fix_age_seconds"] == "11"
    assert new_snap["last_fix_age_seconds"] == "11"
    assert new_snap["task_invoke_age_seconds"] == "22"
    assert new_snap["native_last_fix_age_seconds"] == "22"
    assert new_snap["observability_class"] == "PIPELINE"
    assert new_snap["oldest_queue_item_age_seconds"] == "200"
    assert new_snap["persistence_lag_seconds"] == "300"

    assert legacy_snap["location_fix_age_seconds"] == "9"
    assert legacy_snap["task_invoke_age_seconds"] == "33"
    assert legacy_snap["native_last_fix_age_seconds"] == "33"
    assert legacy_snap["observability_class"] == ""

    assert empty_snap["location_fix_age_seconds"] == ""
    assert empty_snap["task_invoke_age_seconds"] == ""
    assert empty_snap["native_last_fix_age_seconds"] == ""


def test_canary_observability_legacy_write_device_health_passthrough():
    """Canary : write_device_health legacy accepte nouveaux champs sans erreur."""
    from services.geolocation.device_health import (
        parse_device_health,
        write_device_health,
    )

    store: dict[str, dict[str, str]] = {}

    class FakeRedis:
        def hset(self, key, mapping=None, **_kwargs):
            store[key] = dict(mapping or {})
            return True

        def expire(self, *_args, **_kwargs):
            return True

        def hgetall(self, key):
            return store.get(key, {})

    redis = FakeRedis()
    ok = write_device_health(
        redis,
        7,
        {
            "fgs_running": True,
            "battery_optimized": False,
            "constraint_reason": None,
            "fg_permission": "granted",
            "bg_permission": "granted",
            "gps_provider_enabled": True,
            "location_fix_age_seconds": 14,
            "task_invoke_age_seconds": 40,
            "observability_class": "RUNTIME_ONLY",
        },
    )
    assert ok is True
    parsed = parse_device_health(store["driver:7:device_health"])
    assert parsed is not None
    assert parsed["location_fix_age_seconds"] == 14
    assert parsed["task_invoke_age_seconds"] == 40
    assert parsed["native_last_fix_age_seconds"] == 40
    assert parsed["observability_class"] == "RUNTIME_ONLY"


def test_read_driver_device_health_snapshot_empty():
    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {}
    with patch("services.driver_device_health.redis_client", mock_redis):
        assert read_driver_device_health_snapshot(1) is None


def test_read_driver_device_health_snapshot_present():
    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {b"platform": b"ios", b"tracking_active": b"1"}
    with patch("services.driver_device_health.redis_client", mock_redis):
        snap = read_driver_device_health_snapshot(1)
    assert snap is not None
    assert snap["platform"] == "ios"


def test_ingest_driver_device_health_persists_tracking_pipeline(db, sample_driver):
    """JZ-R1 : tracking_pipeline JSONB optionnel, backward-compatible."""
    mock_redis = MagicMock()
    pipeline = {
        "pipeline_snapshot_version": 1,
        "bridge_last_fix_age_s": 5,
        "durable_ack_age_s": 1900,
        "first_suspect": "FLUSH",
    }
    with (
        patch("services.driver_device_health.redis_client", mock_redis),
        patch(
            "services.geolocation.device_health.write_device_health", return_value=True
        ),
        patch(
            "services.monitoring.driver_device_health_metrics.record_device_health_report"
        ),
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent"
        ) as mock_event_cls,
        patch("services.driver_device_health.db.session") as mock_session,
    ):
        mock_event = MagicMock()
        mock_event.id = 4242
        mock_event_cls.return_value = mock_event
        snapshot = ingest_driver_device_health(
            sample_driver.id,
            {
                "platform": "android",
                "tracking_active": True,
                "tracking_pipeline": pipeline,
            },
        )

    assert snapshot["tracking_pipeline"] == pipeline
    assert snapshot["device_health_event_id"] == 4242
    assert snapshot["pipeline_snapshot_version"] == "1"
    mock_event_cls.assert_called_once()
    assert mock_event_cls.call_args.kwargs["tracking_pipeline"] == pipeline
    mock_session.add.assert_called_once_with(mock_event)
    mock_session.commit.assert_called_once()


def test_ingest_driver_device_health_legacy_without_tracking_pipeline(
    db, sample_driver
):
    """Ancien client sans tracking_pipeline → 2xx path inchangé."""
    mock_redis = MagicMock()
    with (
        patch("services.driver_device_health.redis_client", mock_redis),
        patch(
            "services.geolocation.device_health.write_device_health", return_value=True
        ),
        patch(
            "services.monitoring.driver_device_health_metrics.record_device_health_report"
        ),
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent"
        ) as mock_event_cls,
        patch("services.driver_device_health.db.session"),
    ):
        mock_event = MagicMock()
        mock_event.id = None
        mock_event_cls.return_value = mock_event
        snapshot = ingest_driver_device_health(
            sample_driver.id,
            {"platform": "android", "last_fix_age_seconds": 9},
        )

    assert "tracking_pipeline" not in snapshot
    assert mock_event_cls.call_args.kwargs["tracking_pipeline"] is None


def test_purge_old_device_health_events():
    with (
        patch("services.driver_device_health.db.session") as mock_session,
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent.query"
        ) as mock_query,
    ):
        mock_query.filter.return_value.delete.return_value = 2
        from services.driver_device_health import purge_old_device_health_events

        deleted = purge_old_device_health_events()

    assert deleted == 2
    mock_session.commit.assert_called_once()
