"""Processus Socket.IO réel pour la preuve inter-workers. Pas un test pytest."""

from __future__ import annotations

import json
import os


def _serve_probe_emits() -> None:
    import redis

    from ext import socketio

    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    client = redis.Redis.from_url(url)
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe("portal_grant:test_probe")
    for message in pubsub.listen():
        if not isinstance(message, dict) or message.get("type") != "message":
            continue
        raw = message.get("data")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        room = payload.get("room") if isinstance(payload, dict) else None
        marker = payload.get("marker") if isinstance(payload, dict) else None
        if not isinstance(room, str) or not room:
            continue
        socketio.emit(
            "portal_probe",
            {"marker": marker},
            to=room,
            namespace="/",
        )


def main() -> None:
    os.environ["DISABLE_EVENTLET"] = "1"
    port = int(os.environ.get("PORTAL_GRANT_WORKER_PORT", "5099"))
    from threading import Thread

    from werkzeug.serving import make_server

    from app import create_app

    application = create_app("testing")
    application.config["JWT_SECRET_KEY"] = os.getenv(
        "PORTAL_GRANT_TEST_JWT_SECRET", "test-secret-key"
    )
    Thread(
        target=_serve_probe_emits, name="portal-grant-test-probe", daemon=True
    ).start()
    server = make_server("127.0.0.1", port, application)
    print("LISTENER_READY", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
