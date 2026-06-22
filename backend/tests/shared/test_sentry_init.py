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


def test_before_send_drops_kafka_bootstrap_dns_noise():
    event = {
        "logentry": {
            "message": "DNS lookup failed for kafka-broker-1:29092, gaierror(-3, 'Temporary failure in name resolution')",
        },
        "logger": "kafka.net.inet",
    }
    assert before_send(event, None) is None


def test_before_send_drops_kafka_bootstrap_nodenotready_noise():
    event = {
        "logentry": {
            "message": "Bootstrap attempt to bootstrap-2 failed: NodeNotReadyError: bootstrap-2",
        },
        "logger": "kafka.net.manager",
    }
    assert before_send(event, None) is None


def test_before_send_drops_kafka_task_already_done_selector_noise():
    event = {
        "logentry": {"message": "Task is already done!"},
        "logger": "kafka.net.selector",
    }
    assert before_send(event, None) is None


def test_before_send_drops_kafka_task_already_done_runtime():
    event = {
        "logentry": {"message": "Task is already done!"},
        "logger": "app.worker",
    }
    assert (
        before_send(event, {"exc_info": (RuntimeError, RuntimeError("Task is already done!"), None)})
        is None
    )


def test_before_send_keeps_non_kafka_runtime_error():
    event = {
        "logentry": {"message": "Unhandled exception in request handler"},
        "logger": "app.errors",
    }
    assert before_send(event, None) == event
