"""Tests enrichissement erreurs FCM."""

from __future__ import annotations

from unittest.mock import patch

from services.notifications.firebase_push import (
    _fcm_generic_error_result,
    send_fcm_android,
    send_fcm_ios,
)


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


@patch("services.notifications.firebase_push._send_with_retry")
@patch("services.notifications.firebase_push._init_firebase", return_value=True)
def test_send_fcm_android_data_only_when_title_and_body_empty(_init, send_mock):
    """P0: pas de bloc Notification visible si title/body vides → message data-only."""
    send_mock.return_value = {"ok": True}
    res = send_fcm_android(
        token="tok",
        title="",
        body="",
        data={"type": "silent_update", "sync_type": "profile"},
    )
    assert res["ok"] is True
    msg = send_mock.call_args.args[0]
    assert getattr(msg, "notification", None) is None
    str_data = msg.data
    assert "title" not in str_data
    assert "body" not in str_data
    assert str_data.get("type") == "silent_update"


@patch("services.notifications.firebase_push._send_with_retry")
@patch("services.notifications.firebase_push._init_firebase", return_value=True)
def test_send_fcm_android_visible_when_title_present(_init, send_mock):
    """Android visible : data-only (title/body dans data), pas de bloc notification FCM."""
    send_mock.return_value = {"ok": True}
    res = send_fcm_android(token="tok", title="Hello", body="World", data={})
    assert res["ok"] is True
    msg = send_mock.call_args.args[0]
    assert getattr(msg, "notification", None) is None
    assert msg.data.get("title") == "Hello"
    assert msg.data.get("body") == "World"


@patch("services.notifications.firebase_push._send_with_retry")
@patch("services.notifications.firebase_push._init_firebase", return_value=True)
def test_send_fcm_ios_data_only_when_title_and_body_empty(_init, send_mock):
    send_mock.return_value = {"ok": True}
    res = send_fcm_ios(token="tok", title="", body="", data={"type": "silent_update"})
    assert res["ok"] is True
    msg = send_mock.call_args.args[0]
    assert getattr(msg, "notification", None) is None
    # APNs headers: background + priority 5 attendus pour silent push iOS.
    assert msg.apns.headers["apns-push-type"] == "background"
    assert msg.apns.headers["apns-priority"] == "5"
