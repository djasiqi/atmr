"""Tests clôture conversation booking (aller-retour)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from routes.booking_messages import _is_conversation_closed


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

    with patch(
        "routes.booking_messages.resolve_return_child_booking",
        return_value=return_booking,
    ):
        assert _is_conversation_closed(outbound) is expected_closed


def test_conversation_closed_one_way_completed():
    outbound = _booking(
        id=10,
        is_return=False,
        is_round_trip=False,
        status="COMPLETED",
    )
    with patch(
        "routes.booking_messages.resolve_return_child_booking",
        return_value=None,
    ):
        assert _is_conversation_closed(outbound) is True


def test_conversation_open_round_trip_without_return_segment():
    outbound = _booking(
        id=11,
        is_return=False,
        is_round_trip=True,
        status="COMPLETED",
    )
    with patch(
        "routes.booking_messages.resolve_return_child_booking",
        return_value=None,
    ):
        assert _is_conversation_closed(outbound) is False
