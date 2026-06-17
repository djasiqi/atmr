"""Tests shared.sentry_init before_send filter."""

from __future__ import annotations

from shared.sentry_init import before_send


def test_before_send_drops_expired_signature_error():
    event = {"request": {"url": "http://api.example.com/notifications"}}
    hint = {"exc_info": (type("ExpiredSignatureError", (Exception,), {}), None, None)}
    hint["exc_info"] = (__import__("builtins").type("ExpiredSignatureError", (Exception,), {}), None, None)

    class ExpiredSignatureError(Exception):
        pass

    assert before_send(event, {"exc_info": (ExpiredSignatureError, None, None)}) is None


def test_before_send_drops_stopiteration_on_socketio():
    event = {"request": {"url": "http://api.example.com/socket.io/"}}
    assert before_send(event, {"exc_info": (StopIteration, None, None)}) is None


def test_before_send_drops_greenletexit_on_socketio():
    class GreenletExit(BaseException):
        pass

    event = {
        "request": {"url": "http://api.lirie.ch/socket.io/?transport=websocket"},
    }
    assert before_send(event, {"exc_info": (GreenletExit, None, None)}) is None


def test_before_send_keeps_greenletexit_off_socketio():
    class GreenletExit(BaseException):
        pass

    event = {"request": {"url": "http://api.example.com/api/v1/dispatch"}}
    assert before_send(event, {"exc_info": (GreenletExit, None, None)}) == event


def test_before_send_keeps_other_errors():
    event = {"request": {"url": "http://api.example.com/api/v1/dispatch"}}
    assert before_send(event, {"exc_info": (RuntimeError, None, None)}) == event


def test_before_send_drops_institution_notifications_jwt_log_noise():
    event = {
        "logentry": {
            "message": "[InstitutionNotifications] GET error: Signature has expired",
        },
        "request": {"url": "http://api.example.com/api/v1/institutions/notifications"},
    }
    assert before_send(event, None) is None


def test_before_send_drops_institution_read_all_jwt_log_noise():
    event = {
        "logentry": {
            "message": "[InstitutionNotifications] PUT read-all error: Signature has expired",
        },
        "request": {
            "url": "http://api.example.com/api/v1/institutions/notifications/read-all",
        },
    }
    assert before_send(event, None) is None


def test_before_send_drops_concurrent_object_use_error():
    class ConcurrentObjectUseError(Exception):
        pass

    event = {"message": "Error handling request (no URI read)"}
    hint = {"exc_info": (ConcurrentObjectUseError, None, None)}
    assert before_send(event, hint) is None


def test_before_send_drops_gunicorn_no_uri_read_log():
    event = {
        "logentry": {"message": "Error handling request (no URI read)"},
        "logger": "gunicorn.error",
    }
    assert before_send(event, None) is None
