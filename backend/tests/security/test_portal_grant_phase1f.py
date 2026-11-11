"""Phase 1F — Redis coupé pendant une révocation, sur deux processus."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from contextlib import suppress

import pytest
import redis

from models import UserRole
from models.enums import PortalKind
from models.portal_grant import PortalGrant
from security.portal_grant_access import revoke_portal_grant
from security.portal_grant_socket_bus import LISTENER_CLIENT_NAME
from sockets.chat import _SID_INDEX
from tests.security.test_portal_grant_phase1 import _company, _grant, _user
from tests.security.test_portal_grant_phase1e import (
    _delete_account,
    _open_socket,
    _poll_socket,
    _redis_client,
    _token,
)

pytestmark = pytest.mark.integration

_WORKER_PORT = "5100"
_PROBE_CHANNEL = "portal_grant:test_probe"


@pytest.fixture
def portal_grant_table(db):
    PortalGrant.__table__.create(bind=db.engine, checkfirst=True)
    return db


@pytest.fixture
def enforce_grants(app):
    had_key = "PORTAL_GRANT_ENFORCEMENT" in app.config
    previous = app.config.get("PORTAL_GRANT_ENFORCEMENT")
    app.config["PORTAL_GRANT_ENFORCEMENT"] = True
    yield
    if had_key:
        app.config["PORTAL_GRANT_ENFORCEMENT"] = previous
    else:
        app.config.pop("PORTAL_GRANT_ENFORCEMENT", None)


def _wait_log(logs: list[str], marker: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(marker in line for line in logs):
            return True
        time.sleep(0.05)
    return False


def _logged_stamp(logs: list[str], marker: str) -> float | None:
    for line in logs:
        if marker not in line or "at=" not in line:
            continue
        tail = line.split("at=", 1)[1].strip().split()[0]
        try:
            return float(tail)
        except ValueError:
            continue
    return None


def _named_listener_ids(client: redis.Redis) -> list[str]:
    raw = client.execute_command("CLIENT", "LIST")
    if isinstance(raw, bytes):
        raw = raw.decode()
    found: list[str] = []
    for line in str(raw).splitlines():
        fields: dict[str, str] = {}
        for part in line.split(" "):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        if fields.get("name") == LISTENER_CLIENT_NAME and fields.get("id"):
            found.append(fields["id"])
    return found


def _interrupt_revocation_channel(client: redis.Redis) -> int:
    killed = 0
    for client_id in _named_listener_ids(client):
        client.execute_command("CLIENT", "KILL", "ID", client_id)
        killed += 1
    return killed


def _start_worker() -> tuple[subprocess.Popen[str], list[str], threading.Event]:
    env = os.environ.copy()
    env["DISABLE_EVENTLET"] = "1"
    env["FLASK_CONFIG"] = "testing"
    env["PORTAL_GRANT_TEST_JWT_SECRET"] = "test-secret-key"
    env["PORTAL_GRANT_ENFORCEMENT"] = "true"
    env["PORTAL_GRANT_WORKER_PORT"] = _WORKER_PORT
    env["PYTHONUNBUFFERED"] = "1"
    logs: list[str] = []
    ready = threading.Event()
    process = subprocess.Popen(
        ["python", "-m", "tests.security.portal_grant_socket_worker"],
        cwd="/app",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def _read() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            logs.append(line)
            if "LISTENER_READY" in line:
                ready.set()

    threading.Thread(target=_read, daemon=True).start()
    return process, logs, ready


def _probe(bus: redis.Redis, room: str, marker: str) -> None:
    bus.publish(_PROBE_CHANNEL, json.dumps({"room": room, "marker": marker}))


def test_redis_coupe_pendant_revocation_deux_processus(
    app, portal_grant_table, enforce_grants
):
    owner_a = _user(portal_grant_table, UserRole.COMPANY)
    company_a = _company(portal_grant_table, owner_a, approved=True)
    grant_a = _grant(
        portal_grant_table,
        owner_a,
        PortalKind.COMPANY.value,
        company_id=company_a.id,
    )
    owner_b = _user(portal_grant_table, UserRole.COMPANY)
    company_b = _company(portal_grant_table, owner_b, approved=True)
    grant_b = _grant(
        portal_grant_table,
        owner_b,
        PortalKind.COMPANY.value,
        company_id=company_b.id,
    )
    owner_c = _user(portal_grant_table, UserRole.COMPANY)
    company_c = _company(portal_grant_table, owner_c, approved=True)
    _grant(
        portal_grant_table,
        owner_c,
        PortalKind.COMPANY.value,
        company_id=company_c.id,
    )
    portal_grant_table.session.commit()
    token_a = _token(app, owner_a, role="COMPANY", company_id=company_a.id)
    token_b = _token(app, owner_b, role="COMPANY", company_id=company_b.id)
    token_c = _token(app, owner_c, role="COMPANY", company_id=company_c.id)
    process, logs, ready = _start_worker()
    sessions = []
    try:
        assert ready.wait(120), "".join(logs[-40:])
        assert process.pid != os.getpid()
        if process.poll() is not None:
            raise AssertionError("".join(logs[-60:]))
        base = f"http://127.0.0.1:{_WORKER_PORT}/socket.io/"
        bus = _redis_client()
        deadline = time.time() + 15
        while time.time() < deadline and not _named_listener_ids(bus):
            time.sleep(0.1)
        assert _named_listener_ids(bus), "abonné de révocation absent"

        room_a = f"company_{company_a.id}"
        session_a, sid_a, handshake_a = _open_socket(base, token_a)
        sessions.append(session_a)
        assert room_a in handshake_a, handshake_a[:400]
        assert all(
            claims.get("user_id") != owner_a.id for claims in _SID_INDEX.values()
        )
        _probe(bus, room_a, "avant-a")
        status_a, body_a = _poll_socket(session_a, base, sid_a)
        assert status_a == 200
        assert "avant-a" in body_a
        revoke_portal_grant(grant_a)
        _probe(bus, room_a, "pendant-a")
        _pending_status, pending_body = _poll_socket(session_a, base, sid_a)
        assert "pendant-a" in pending_body
        committed_at = time.time()
        portal_grant_table.session.commit()
        assert _wait_log(logs, "portal_grant_socket_revoked", 5), "".join(logs[-40:])
        healthy_stamp = _logged_stamp(logs, "portal_grant_socket_revoked")
        assert healthy_stamp is not None
        healthy_ms = (healthy_stamp - committed_at) * 1000
        _probe(bus, room_a, "apres-a")
        _late_a_status, late_a = _poll_socket(session_a, base, sid_a, timeout=3)
        assert "apres-a" not in late_a
        again_a, _again_sid_a, again_body_a = _open_socket(base, token_a)
        sessions.append(again_a)
        assert room_a not in again_body_a
        print(f"HEALTHY_REVOCATION_LATENCY_MS={healthy_ms:.1f}")

        room_b = f"company_{company_b.id}"
        session_b, sid_b, handshake_b = _open_socket(base, token_b)
        sessions.append(session_b)
        assert room_b in handshake_b, handshake_b[:400]
        _probe(bus, room_b, "avant-b")
        status_b, body_b = _poll_socket(session_b, base, sid_b)
        assert status_b == 200
        assert "avant-b" in body_b
        killed_at = time.time()
        killed = _interrupt_revocation_channel(bus)
        assert killed >= 1
        revoked_at = time.time()
        revoke_portal_grant(grant_b)
        portal_grant_table.session.commit()
        _probe(bus, room_b, "apres-b")
        _late_b_status, late_b = _poll_socket(session_b, base, sid_b, timeout=3)
        assert "apres-b" not in late_b
        assert _wait_log(logs, "portal_grant_channel_unhealthy", 5), "".join(logs[-50:])
        assert _wait_log(logs, "portal_grant_professional_sockets_dropped", 5)
        drop_stamp = _logged_stamp(logs, "portal_grant_professional_sockets_dropped")
        assert drop_stamp is not None
        detection_ms = (drop_stamp - killed_at) * 1000
        revocation_gap_ms = (drop_stamp - revoked_at) * 1000
        print(f"REDIS_DOWN_DETECTION_MS={detection_ms:.1f}")
        print(f"REVOCATION_TO_DROP_MS={revocation_gap_ms:.1f}")

        assert _wait_log(logs, "portal_grant_channel_recovered", 10), "".join(
            logs[-40:]
        )
        assert _wait_log(logs, "portal_grant_reconciliation_closed", 5)
        again_b, _again_sid_b, again_body_b = _open_socket(base, token_b)
        sessions.append(again_b)
        assert room_b not in again_body_b
        room_c = f"company_{company_c.id}"
        session_c, _sid_c, handshake_c = _open_socket(base, token_c)
        sessions.append(session_c)
        assert room_c in handshake_c, handshake_c[:400]
    finally:
        for http_session in sessions:
            http_session.close()
        process.terminate()
        with suppress(Exception):
            process.wait(timeout=10)
        if process.poll() is None:
            process.kill()
        for user_id in (owner_a.id, owner_b.id, owner_c.id):
            _delete_account(portal_grant_table.engine, user_id)
