from __future__ import annotations

from unittest.mock import patch

import pytest

from services.realtime.socketio import fanout_driver_location_update


@patch("services.realtime.socketio.inc_fanout")
def test_fanout_increments_metrics(mock_fanout) -> None:
    with patch("services.realtime.socketio._safe_emit"):
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="accepted_observability_only",
        )
    assert mock_fanout.call_count == 1


@patch("services.realtime.socketio.inc_fanout")
def test_fanout_metrics_two_emits_for_canonical(mock_fanout) -> None:
    with patch("services.realtime.socketio._safe_emit"):
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="accepted_canonical",
        )
    assert mock_fanout.call_count == 2


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


def test_fanout_emits_location_only_for_observability_status() -> None:
    """Observabilité : géométrie uniquement (pas driver_live_state_update — statut métier)."""
    with patch("services.realtime.socketio._safe_emit") as safe_emit:
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="accepted_observability_only",
        )
    assert safe_emit.call_count == 1
    assert safe_emit.call_args[0][0] == "driver_location_update"
    assert safe_emit.call_args[0][1].get("accept_status") == "accepted_observability_only"


def test_fanout_skips_rejected_invalid() -> None:
    with patch("services.realtime.socketio._safe_emit") as safe_emit:
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="rejected_invalid",
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
