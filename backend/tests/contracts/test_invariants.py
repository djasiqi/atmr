"""Tests contractuels invariants INV-1 à INV-3 (backend)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_inv3_mission_live_missing_mission_id_logs_and_metrics() -> None:
    from services.geolocation.location import LocationService

    svc = LocationService(redis_client_instance=None)
    with (
        patch("services.geolocation.location.DriverRepository") as mock_repo_cls,
        patch.object(
            svc,
            "_store_location",
            return_value=("accepted", "", None, False, None),
        ),
        patch.object(svc, "_is_v21_enabled_for_company", return_value=True),
        patch(
            "services.monitoring.driver_location_metrics.inc_tracking_mission_live_missing_mission_id"
        ) as mock_inc,
        patch(
            "services.monitoring.driver_location_metrics.inc_tracking_invariant_violation"
        ) as mock_inv,
    ):
        mock_repo = mock_repo_cls.return_value
        mock_repo.find_by_id.return_value = MagicMock(company_id=1)
        svc.update_driver_location(
            driver_id=1,
            latitude=46.2,
            longitude=6.1,
            mission_id=None,
            location_mode="mission_live",
            transport="http",
        )
        mock_inc.assert_called_once()
        mock_inv.assert_called_once()
        assert mock_inv.call_args.kwargs.get("invariant_id") == "INV-3"


def test_kafka_partition_key_for_driver_location() -> None:
    from services.region_router import kafka_partition_key_for_driver_location

    key = kafka_partition_key_for_driver_location(region_id="CH", driver_id=7514)
    assert key == "CH:driver:7514"
    assert "7514" in key
