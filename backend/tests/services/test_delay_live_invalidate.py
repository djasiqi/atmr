"""P3 — throttle emit_delay_live_invalidate."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_throttle_state():
    from services.realtime import socketio as sio_mod

    sio_mod._DELAY_LIVE_LAST_EMIT.clear()
    yield
    sio_mod._DELAY_LIVE_LAST_EMIT.clear()


def test_emit_delay_live_invalidate_second_call_throttled_within_window():
    from services.realtime import socketio as sio_mod

    with patch.object(sio_mod, "emit_company_event") as em:
        sio_mod.emit_delay_live_invalidate(1, "2025-06-01", "r1")
        sio_mod.emit_delay_live_invalidate(1, "2025-06-01", "r2")
        assert em.call_count == 1


def test_emit_delay_live_invalidate_different_date_emits_twice():
    from services.realtime import socketio as sio_mod

    with patch.object(sio_mod, "emit_company_event") as em:
        sio_mod.emit_delay_live_invalidate(1, "2025-06-01", "a")
        sio_mod.emit_delay_live_invalidate(1, "2025-06-02", "b")
        assert em.call_count == 2
