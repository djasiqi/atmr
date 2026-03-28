"""Tests enrichissement erreurs FCM."""

from __future__ import annotations

from services.notifications.firebase_push import _fcm_generic_error_result


def test_fcm_generic_error_result_includes_error_class():
    class CustomErr(Exception):
        pass

    out = _fcm_generic_error_result(CustomErr("boom"))
    assert out["ok"] is False
    assert out["error"] == "fcm_send_error"
    assert out["error_class"] == "CustomErr"
    assert "boom" in out["error_message"]


def test_fcm_generic_error_marks_token_invalid_heuristic():
    out = _fcm_generic_error_result(
        Exception("Requested entity was not found for registration token")
    )
    assert out.get("token_invalid") is True
