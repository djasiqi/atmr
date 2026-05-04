"""Tests P1 — logique delta / throttle importée via comportement public (sans socket réel)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from application.drivers.get_driver_bookings_eta import (
    BookingEtaItem,
    DriverBookingsEtaResponse,
)


def _make_resp(
    items: list[tuple[int, int | None, int | None]],
) -> DriverBookingsEtaResponse:
    return DriverBookingsEtaResponse(
        has_gps=True,
        driver_position={"lat": 46.0, "lon": 6.0},
        bookings=[
            BookingEtaItem(
                id=bid,
                eta_to_pickup_seconds=pu,
                eta_to_dropoff_seconds=do,
                duration_seconds=100,
                distance_meters=1000,
                estimated_arrival="2026-01-01T12:00:00",
                estimated_arrival_dropoff=None,
            )
            for bid, pu, do in items
        ],
    )


def test_significant_delta_requires_10s_change() -> None:
    from services.geolocation import driver_eta_socket_fanout as m

    prev = {1: (100, None)}
    new_same = {1: (105, None)}
    assert m._significant_delta(prev, new_same) is False

    new_big = {1: (120, None)}
    assert m._significant_delta(prev, new_big) is True


@pytest.mark.parametrize(
    "accept_status",
    ["accepted_observability_only", "rejected"],
)
def test_maybe_emit_skips_non_canonical(accept_status: str) -> None:
    from services.geolocation.driver_eta_socket_fanout import (
        maybe_emit_eta_changed_after_driver_location,
    )

    with patch(
        "services.geolocation.driver_eta_socket_fanout.BookingRepository"
    ) as repo_cls:
        maybe_emit_eta_changed_after_driver_location(
            driver_id=1,
            driver_lat=46.0,
            driver_lon=6.0,
            accept_status=accept_status,
        )
        repo_cls.assert_not_called()


def test_maybe_emit_throttles_second_call_within_window() -> None:
    from services.geolocation import driver_eta_socket_fanout as m
    from services.geolocation.driver_eta_socket_fanout import (
        maybe_emit_eta_changed_after_driver_location,
    )

    m._last_compute_monotonic.clear()
    m._last_eta_emitted_by_driver.clear()

    fake_booking = MagicMock()
    fake_booking.id = 1
    fake_booking.status = "ASSIGNED"
    fake_booking.pickup_lat = 46.0
    fake_booking.pickup_lon = 6.0
    fake_booking.dropoff_lat = 46.1
    fake_booking.dropoff_lon = 6.1
    fake_booking.duration_seconds = 100
    fake_booking.distance_meters = 1000

    resp_a = _make_resp([(1, 100, None)])
    resp_b = _make_resp([(1, 500, None)])

    with (
        patch.object(m, "_ETA_COMPUTE_MIN_INTERVAL_SEC", 60.0),
        patch(
            "services.geolocation.driver_eta_socket_fanout.BookingRepository"
        ) as repo_cls,
        patch(
            "services.geolocation.driver_eta_socket_fanout.GetDriverBookingsETAUseCase"
        ) as uc_cls,
        patch(
            "services.geolocation.driver_eta_socket_fanout.emit_driver_event"
        ) as emit,
    ):
        repo_inst = MagicMock()
        repo_inst.find_models_by_driver_with_statuses_and_time_range.return_value = [
            fake_booking
        ]
        repo_cls.return_value = repo_inst

        uc_inst = MagicMock()
        uc_inst.execute.side_effect = [resp_a, resp_b]
        uc_cls.return_value = uc_inst

        t0 = time.monotonic()
        with patch("services.geolocation.driver_eta_socket_fanout.time") as time_mod:
            time_mod.monotonic.side_effect = [t0, t0 + 1.0, t0 + 2.0]
            maybe_emit_eta_changed_after_driver_location(
                driver_id=7,
                driver_lat=46.0,
                driver_lon=6.0,
                accept_status="accepted_canonical",
            )
            maybe_emit_eta_changed_after_driver_location(
                driver_id=7,
                driver_lat=46.0,
                driver_lon=6.0,
                accept_status="accepted_canonical",
            )

        assert emit.call_count == 1
