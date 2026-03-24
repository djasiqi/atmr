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


def test_fanout_emits_for_observability_status() -> None:
    """L'observabilité doit fanouter aussi (carte entreprise), pas seulement le canon Redis."""
    with patch("services.realtime.socketio._safe_emit") as safe_emit:
        fanout_driver_location_update(
            1,
            {"driver_id": 10, "company_id": 1},
            {"driver_id": 10, "company_id": 1},
            accept_status="accepted_observability_only",
        )
    assert safe_emit.call_count == 2
    payloads = [c.args[1] for c in safe_emit.call_args_list]
    assert all(p.get("accept_status") == "accepted_observability_only" for p in payloads)


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
