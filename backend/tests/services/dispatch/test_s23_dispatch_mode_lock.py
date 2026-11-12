"""Verrou S23 dispatch mode (LOT-BOOKING-DISPATCH-1)."""

from __future__ import annotations

import pytest

from services.dispatch import s23_dispatch_mode_lock as lock


def test_manual_always_allowed():
    lock.assert_dispatch_mode_switch_allowed("manual")
    lock.assert_dispatch_mode_switch_allowed("")


def test_semi_auto_locked_by_default():
    with pytest.raises(lock.DispatchModeLockedError) as exc:
        lock.assert_dispatch_mode_switch_allowed("semi_auto")
    assert exc.value.mode == "semi_auto"


def test_fully_auto_locked_by_default():
    with pytest.raises(lock.DispatchModeLockedError) as exc:
        lock.assert_dispatch_mode_switch_allowed("fully_auto")
    assert "automatique" in str(exc.value).lower()


def test_semi_auto_allowed_when_flag_on(monkeypatch):
    monkeypatch.setattr(lock, "SEMI_AUTO_DISPATCH_ENABLED", True)
    lock.assert_dispatch_mode_switch_allowed("semi_auto")
