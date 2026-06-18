from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.geolocation.location import LocationService


def test_mission_live_without_mission_id_downgraded_to_availability(app) -> None:
    svc = LocationService(redis_client_instance=None)
    mock_store = MagicMock(
        return_value=("accepted_canonical", "", "2026-06-17T12:00:00Z")
    )
    mock_driver = MagicMock()
    mock_driver.company_id = 1
    with app.app_context():
        with patch.object(svc, "_store_location", mock_store):
            with patch.object(svc, "_is_v21_enabled_for_company", return_value=True):
                with patch(
                    "services.geolocation.location.DriverRepository"
                ) as mock_repo_cls:
                    mock_repo_cls.return_value.find_by_id.return_value = mock_driver
                    with patch(
                        "services.geolocation.location.publish_event"
                    ):
                        with patch(
                            "services.monitoring.driver_location_metrics."
                            "inc_tracking_mission_live_missing_mission_id"
                        ) as mock_inc:
                            svc.update_driver_location(
                                driver_id=1,
                                latitude=46.2,
                                longitude=6.1,
                                location_mode="mission_live",
                                mission_id=None,
                            )
    mock_inc.assert_called_once_with(transport="http", action="downgraded")
    call_kwargs = mock_store.call_args.kwargs
    assert call_kwargs["location_mode"] == "availability_presence"
