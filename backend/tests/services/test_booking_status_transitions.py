"""P0-A MISSION-STATE : transitions Booking centralisées."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from models.enums import BookingStatus
from services.booking.status_transitions import (
    BookingStatusTransitionError,
    transition_booking_status,
)


@dataclass
class _FakeBooking:
    id: int = 1
    status: BookingStatus = BookingStatus.PENDING
    driver_id: int | None = 10

    def validate_status_transition(
        self, new_status: BookingStatus
    ) -> tuple[bool, str | None]:
        # Réutilise la vraie machine du modèle via une instance minimale.
        from models.booking import Booking

        return Booking.validate_status_transition(
            self,  # type: ignore[arg-type]
            new_status,
        )


def test_forward_progress_applies() -> None:
    booking = _FakeBooking(status=BookingStatus.PENDING)
    assert (
        transition_booking_status(
            booking, BookingStatus.ACCEPTED, source="test"
        )
        is True
    )
    assert booking.status == BookingStatus.ACCEPTED


def test_same_status_is_noop() -> None:
    booking = _FakeBooking(status=BookingStatus.ACCEPTED)
    assert (
        transition_booking_status(
            booking, BookingStatus.ACCEPTED, source="test"
        )
        is False
    )


@pytest.mark.parametrize(
    "terminal",
    [BookingStatus.COMPLETED, BookingStatus.RETURN_COMPLETED, BookingStatus.CANCELED],
)
@pytest.mark.parametrize(
    "target",
    [
        BookingStatus.PENDING,
        BookingStatus.ACCEPTED,
        BookingStatus.ASSIGNED,
        BookingStatus.EN_ROUTE,
        BookingStatus.IN_PROGRESS,
    ],
)
def test_terminal_states_are_immutable(
    terminal: BookingStatus, target: BookingStatus
) -> None:
    """C4/C9 : COMPLETED / RETURN_COMPLETED / CANCELED = terminaux, toute sortie 409."""
    booking = _FakeBooking(status=terminal)
    with pytest.raises(BookingStatusTransitionError) as exc_info:
        transition_booking_status(booking, target, source="test")
    assert exc_info.value.http_status == 409
    assert exc_info.value.code == "terminal_state"
    assert booking.status == terminal


def test_backward_progress_is_stale_409() -> None:
    booking = _FakeBooking(status=BookingStatus.IN_PROGRESS)
    with pytest.raises(BookingStatusTransitionError) as exc_info:
        transition_booking_status(booking, BookingStatus.ASSIGNED, source="test")
    assert exc_info.value.http_status == 409
    assert exc_info.value.code == "stale_transition"
    assert booking.status == BookingStatus.IN_PROGRESS


def test_forward_skip_is_invalid_400() -> None:
    booking = _FakeBooking(status=BookingStatus.ASSIGNED)
    with pytest.raises(BookingStatusTransitionError) as exc_info:
        transition_booking_status(booking, BookingStatus.COMPLETED, source="test")
    assert exc_info.value.http_status == 400
    assert exc_info.value.code == "invalid_transition"


def test_cancel_intent_allowed_from_non_terminal() -> None:
    booking = _FakeBooking(status=BookingStatus.EN_ROUTE)
    assert (
        transition_booking_status(
            booking, BookingStatus.CANCELED, source="test", intent="cancel"
        )
        is True
    )
    assert booking.status == BookingStatus.CANCELED


def test_deassign_allowed_pre_departure() -> None:
    booking = _FakeBooking(status=BookingStatus.ASSIGNED)
    assert (
        transition_booking_status(
            booking, BookingStatus.ACCEPTED, source="test", intent="deassign"
        )
        is True
    )
    assert booking.status == BookingStatus.ACCEPTED


def test_deassign_forbidden_when_onboard() -> None:
    """Patient à bord : aucune désassignation possible."""
    booking = _FakeBooking(status=BookingStatus.IN_PROGRESS)
    with pytest.raises(BookingStatusTransitionError) as exc_info:
        transition_booking_status(
            booking, BookingStatus.PENDING, source="test", intent="deassign"
        )
    assert exc_info.value.http_status == 409
    assert exc_info.value.code == "deassign_forbidden"
    assert booking.status == BookingStatus.IN_PROGRESS


def test_error_payload_shape() -> None:
    booking = _FakeBooking(status=BookingStatus.COMPLETED)
    try:
        transition_booking_status(booking, BookingStatus.ASSIGNED, source="test")
    except BookingStatusTransitionError as exc:
        payload: dict[str, Any] = exc.to_payload()
        assert payload["error_code"] == "terminal_state"
        assert payload["retryable"] is False
        assert isinstance(payload["error"], str)
    else:  # pragma: no cover
        raise AssertionError("expected BookingStatusTransitionError")
