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


def test_before_send_keeps_other_errors():
    event = {"request": {"url": "http://api.example.com/api/v1/dispatch"}}
    assert before_send(event, {"exc_info": (RuntimeError, None, None)}) == event
