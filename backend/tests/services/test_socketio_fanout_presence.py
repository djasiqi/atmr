from __future__ import annotations

from unittest.mock import patch

import pytest

from services.realtime.socketio import fanout_driver_location_update


def test_fanout_requires_explicit_accept_status_keyword() -> None:
    with pytest.raises(TypeError):
        fanout_driver_location_update(  # type: ignore[call-arg]
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
        )


def test_fanout_rejects_empty_accept_status() -> None:
    with pytest.raises(ValueError, match="accept_status is required"):
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="",
        )


def test_fanout_skips_non_canonical_status() -> None:
    with patch("services.realtime.socketio._safe_emit") as safe_emit:
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="accepted_observability_only",
        )
    safe_emit.assert_not_called()


def test_fanout_emits_both_events_for_canonical() -> None:
    with patch("services.realtime.socketio._safe_emit") as safe_emit:
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="accepted_canonical",
        )

    assert safe_emit.call_count == 2
