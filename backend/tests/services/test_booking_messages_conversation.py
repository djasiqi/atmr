"""Tests clôture conversation booking (aller-retour et multi-étapes)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.booking import Booking
from routes.booking_messages import (
    _is_conversation_closed,
    _route_group_has_active_leg,
)


def _booking(**kwargs):
    return SimpleNamespace(**kwargs)


@pytest.mark.parametrize(
    ("outbound_status", "return_status", "expected_closed"),
    [
        ("COMPLETED", "ASSIGNED", False),
        ("COMPLETED", "IN_PROGRESS", False),
        ("COMPLETED", "COMPLETED", True),
        ("COMPLETED", "RETURN_COMPLETED", True),
        ("COMPLETED", "CANCELED", True),
        ("ASSIGNED", None, False),
    ],
)
def test_conversation_open_until_return_terminal(
    outbound_status, return_status, expected_closed
):
    outbound = _booking(
        id=1,
        is_return=False,
        is_round_trip=True,
        status=outbound_status,
    )
    return_booking = (
        _booking(id=2, is_return=True, parent_booking_id=1, status=return_status)
        if return_status is not None
        else None
    )

    with (
        patch(
            "routes.booking_messages._route_group_has_active_leg",
            return_value=False,
        ),
        patch(
            "routes.booking_messages.resolve_return_child_booking",
            return_value=return_booking,
        ),
    ):
        assert _is_conversation_closed(outbound) is expected_closed


def test_conversation_closed_one_way_completed():
    outbound = _booking(
        id=10,
        is_return=False,
        is_round_trip=False,
        status="COMPLETED",
    )
    with (
        patch(
            "routes.booking_messages._route_group_has_active_leg",
            return_value=False,
        ),
        patch(
            "routes.booking_messages.resolve_return_child_booking",
            return_value=None,
        ),
    ):
        assert _is_conversation_closed(outbound) is True


def test_conversation_open_round_trip_without_return_segment():
    outbound = _booking(
        id=11,
        is_return=False,
        is_round_trip=True,
        status="COMPLETED",
    )
    with (
        patch(
            "routes.booking_messages._route_group_has_active_leg",
            return_value=False,
        ),
        patch(
            "routes.booking_messages.resolve_return_child_booking",
            return_value=None,
        ),
    ):
        assert _is_conversation_closed(outbound) is False


def test_conversation_open_when_multi_leg_route_group_active():
    booking = _booking(
        id=35210,
        route_group_id="grp-karan",
        status="COMPLETED",
        is_return=False,
        is_round_trip=False,
    )
    with patch(
        "routes.booking_messages._route_group_has_active_leg",
        return_value=True,
    ):
        assert _is_conversation_closed(booking) is False


def test_route_group_has_active_leg_second_leg_in_progress():
    leg1 = _booking(id=35210, status="COMPLETED", route_group_id="grp-karan")
    leg2 = _booking(id=35211, status="IN_PROGRESS", route_group_id="grp-karan")
    primary = _booking(id=35210, route_group_id="grp-karan", status="COMPLETED")

    mock_query = MagicMock()
    mock_query.filter_by.return_value.order_by.return_value.all.return_value = [
        leg1,
        leg2,
    ]

    with (
        patch.object(Booking, "query", mock_query),
        patch(
            "routes.booking_messages.resolve_return_child_booking",
            return_value=None,
        ),
    ):
        assert _route_group_has_active_leg(primary) is True


def test_route_group_all_legs_terminal():
    leg1 = _booking(id=1, status="COMPLETED", route_group_id="grp-done")
    leg2 = _booking(id=2, status="COMPLETED", route_group_id="grp-done")
    primary = _booking(id=1, route_group_id="grp-done", status="COMPLETED")

    mock_query = MagicMock()
    mock_query.filter_by.return_value.order_by.return_value.all.return_value = [
        leg1,
        leg2,
    ]

    with (
        patch.object(Booking, "query", mock_query),
        patch(
            "routes.booking_messages.resolve_return_child_booking",
            return_value=None,
        ),
    ):
        assert _route_group_has_active_leg(primary) is False
