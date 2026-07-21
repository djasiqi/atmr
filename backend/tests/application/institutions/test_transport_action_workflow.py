"""Tests invariants TransportAction V1.1."""

from __future__ import annotations

from application.institutions.transport_action_workflow import (
    ALLOWED_STATUS_EFFECT,
    assert_status_effect_combo,
    classify_action_type,
    is_counter_enabled,
)
from models.booking_change_request import (
    TransportActionEffectStatus,
    TransportActionStatus,
    TransportActionType,
)


def test_status_effect_combinations_allowed():
    for status, effect in ALLOWED_STATUS_EFFECT:
        assert_status_effect_combo(status, effect)


def test_status_effect_invalid_rejected():
    try:
        assert_status_effect_combo(
            TransportActionStatus.COMPLETED, TransportActionEffectStatus.FAILED
        )
        assert False, "devrait lever"
    except ValueError:
        pass


def test_classify_cancellation():
    assert classify_action_type(set(), is_cancellation=True) == TransportActionType.CANCELLATION


def test_classify_time_change():
    assert (
        classify_action_type({"scheduled_time": True}) == TransportActionType.CHANGE_TIME
    )


def test_counter_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRANSPORT_ACTION_COUNTER_ENABLED", raising=False)
    assert is_counter_enabled() is False


def test_counter_enabled_flag(monkeypatch):
    monkeypatch.setenv("TRANSPORT_ACTION_COUNTER_ENABLED", "true")
    assert is_counter_enabled() is True


def test_open_statuses_include_requested():
    assert TransportActionStatus.REQUESTED in TransportActionStatus.OPEN
    assert TransportActionStatus.PENDING in TransportActionStatus.OPEN
