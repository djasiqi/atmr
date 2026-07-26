"""Tests sérialisation FCM data JSON-safe."""

from __future__ import annotations

from services.notifications.firebase_push import _fcm_str_data, fcm_data_value


def test_fcm_data_value_serializes_dict_as_json():
    value = fcm_data_value({"missions": [{"id": 1}]})
    assert value.startswith("{")
    assert "'missions'" not in value


def test_fcm_data_value_serializes_list_as_json():
    value = fcm_data_value(["a", "b"])
    assert value == '["a", "b"]'


def test_fcm_str_data_mixed_values():
    out = _fcm_str_data({"count": 3, "labels": ["x"], "nested": {"k": "v"}})
    assert out["count"] == "3"
    assert out["labels"].startswith("[")
    assert out["nested"].startswith("{")
