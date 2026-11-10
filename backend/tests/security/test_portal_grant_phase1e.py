"""Phase 1E — socket inter-processus, HTTP réel, garde SQL."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import pytest
import redis
import requests
from flask_jwt_extended import create_access_token
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from models import Institution, UserRole, WebSession
from models.enums import PortalGrantStatus, PortalKind
from models.portal_grant import (
    REVOKED_DELETE_GUARD_DROP_SQL,
    REVOKED_DELETE_GUARD_SQL,
    PortalGrant,
)
from security.portal_grant_access import revoke_portal_grant
from security.portal_scope import authorized_company_id
from security.web_handoff_service import _resolve_company_id
from sockets.chat import _SID_INDEX
from tests.security.test_portal_grant_phase1 import _company, _grant, _user

pytestmark = pytest.mark.integration

_WORKER_PORT = "5099"
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


def _token(app, user, **claims) -> str:
    payload = {"role": user.role.value, "aud": "atmr-api"}
    payload.update(claims)
    with app.app_context():
        return create_access_token(
            identity=str(user.public_id),
            fresh=True,
            additional_claims=payload,
        )


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _institution_headers(app, db, user, institution_id: int) -> dict[str, str]:
    now = datetime.now(UTC)
    session = WebSession()
    session.id = str(uuid.uuid4())
    session.user_id = int(user.id)
    session.institution_id = institution_id
    session.created_at = now
    session.expires_at = now + timedelta(hours=8)
    session.last_interactive_activity_at = now
    db.session.add(session)
    db.session.flush()
    return _auth(
        _token(
            app,
            user,
            role="INSTITUTION",
            institution_id=institution_id,
            sid=session.id,
        )
    )


def _poll_socket(
    session: requests.Session, base: str, sid: str, *, timeout: int = 15
) -> tuple[int, str]:
    try:
        response = session.get(
            base,
            params={"transport": "polling", "EIO": "4", "sid": sid},
            timeout=timeout,
        )
    except requests.Timeout:
        return 0, ""
    return response.status_code, response.text


def _open_socket(base: str, token: str) -> tuple[requests.Session, str, str]:
    session = requests.Session()
    opened = session.get(
        base,
        params={"transport": "polling", "EIO": "4"},
        timeout=15,
    )
    opened.raise_for_status()
    sid = json.loads(opened.text[1:])["sid"]
    session.post(
        base,
        params={"transport": "polling", "EIO": "4", "sid": sid},
        data="40" + json.dumps({"token": token}),
        headers={"Content-Type": "text/plain;charset=UTF-8"},
        timeout=15,
    )
    status, body = _poll_socket(session, base, sid)
    return session, sid, f"{status}:{body}"


def _redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.Redis.from_url(url)


def _delete_account(engine, user_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM portal_grant WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        conn.execute(
            text("DELETE FROM company WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        conn.execute(
            text('DELETE FROM "user" WHERE id = :user_id'),
            {"user_id": user_id},
        )


def test_http_courses_dispatch_institution(app, portal_grant_table, enforce_grants):
    company_user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, company_user, approved=True)
    _grant(
        portal_grant_table,
        company_user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    token = _token(app, company_user, role="COMPANY", company_id=company.id)
    http = app.test_client()
    reservations = http.get(
        "/api/v1/companies/me/reservations?date=2026-10-12&flat=true",
        headers=_auth(token),
    )
    dispatch = http.get(
        "/api/v1/company_dispatch/assignments?date=2026-10-12",
        headers=_auth(token),
    )
    assert reservations.status_code == 200, reservations.get_data(as_text=True)[:400]
    assert dispatch.status_code == 200, dispatch.get_data(as_text=True)[:400]
    stranger = _user(portal_grant_table, UserRole.CLIENT)
    stranger_token = _token(app, stranger, role="CLIENT")
    assert (
        http.get(
            "/api/v1/companies/me/reservations?date=2026-10-12&flat=true",
            headers=_auth(stranger_token),
        ).status_code
        == 403
    )

    pending_user = _user(portal_grant_table, UserRole.INSTITUTION)
    institution = Institution()
    institution.name = "Clinique locale"
    institution.public_id = str(uuid.uuid4())
    portal_grant_table.session.add(institution)
    portal_grant_table.session.flush()
    pending_user.institution_id = institution.id
    _grant(
        portal_grant_table,
        pending_user,
        PortalKind.INSTITUTION.value,
        institution_id=institution.id,
        status=PortalGrantStatus.PENDING_ACTIVATION.value,
        internal_role="institution_admin",
    )
    pending = http.get(
        "/api/v1/institutions/me",
        headers=_institution_headers(
            app, portal_grant_table, pending_user, institution.id
        ),
    )
    assert pending.status_code == 403, pending.get_data(as_text=True)[:300]

    active_user = _user(portal_grant_table, UserRole.INSTITUTION)
    active_user.institution_id = institution.id
    _grant(
        portal_grant_table,
        active_user,
        PortalKind.INSTITUTION.value,
        institution_id=institution.id,
        internal_role="institution_admin",
    )
    opened = http.get(
        "/api/v1/institutions/me",
        headers=_institution_headers(
            app, portal_grant_table, active_user, institution.id
        ),
    )
    assert opened.status_code == 200, opened.get_data(as_text=True)[:400]
    body = opened.get_json()
    assert body["id"] == institution.id or body.get("data", {}).get("id") == institution.id


def test_lectures_company_id_passent_par_le_perimetre(app, portal_grant_table):
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    assert not hasattr(user, "company_id") or getattr(user, "company_id", None) is None
    assert authorized_company_id(user) == company.id
    assert _resolve_company_id(user) == company.id
    token = _token(app, user, role="COMPANY")
    health = app.test_client().get(
        "/api/v1/companies/me/tracking-health",
        headers=_auth(token),
    )
    assert health.status_code == 200
    assert health.get_json()["company_id"] == company.id
    client = _user(portal_grant_table, UserRole.CLIENT)
    refused = app.test_client().get(
        "/api/v1/companies/me/tracking-health",
        headers=_auth(_token(app, client, role="CLIENT")),
    )
    assert refused.status_code == 403


def test_delete_sql_d_une_revocation_est_refuse(portal_grant_table):
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    grant = _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
        status=PortalGrantStatus.REVOKED.value,
    )
    for statement in REVOKED_DELETE_GUARD_SQL:
        portal_grant_table.session.execute(text(statement))

    def _raw_delete() -> None:
        with portal_grant_table.session.begin_nested():
            portal_grant_table.session.execute(
                text("DELETE FROM portal_grant WHERE id = :grant_id"),
                {"grant_id": grant.id},
            )

    try:
        with pytest.raises(DBAPIError):
            _raw_delete()
        portal_grant_table.session.refresh(grant)
        assert grant.status == PortalGrantStatus.REVOKED.value
    finally:
        for statement in REVOKED_DELETE_GUARD_DROP_SQL:
            portal_grant_table.session.execute(text(statement))


def test_socket_deux_processus_coupure_reception(app, portal_grant_table, enforce_grants):
    user = _user(portal_grant_table, UserRole.COMPANY)
    company = _company(portal_grant_table, user, approved=True)
    grant = _grant(
        portal_grant_table,
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
    )
    portal_grant_table.session.commit()
    token = _token(app, user, role="COMPANY", company_id=company.id)
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

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    http_session: requests.Session | None = None
    try:
        assert ready.wait(120), "".join(logs[-40:])
        assert process.pid != os.getpid()
        if process.poll() is not None:
            raise AssertionError("".join(logs[-60:]))
        base = f"http://127.0.0.1:{_WORKER_PORT}/socket.io/"
        http_session, sid, handshake = _open_socket(base, token)
        assert f"company_{company.id}" in handshake, handshake[:400]
        assert all(
            claims.get("user_id") != user.id for claims in _SID_INDEX.values()
        )
        bus = _redis_client()
        room = f"company_{company.id}"
        bus.publish(
            _PROBE_CHANNEL,
            json.dumps({"room": room, "marker": "avant"}),
        )
        status, body = _poll_socket(http_session, base, sid)
        assert status == 200
        assert "avant" in body
        revoke_portal_grant(grant)
        portal_grant_table.session.commit()
        time.sleep(1)
        assert any("portal_grant_socket_revoked" in line for line in logs), "".join(
            logs[-20:]
        )
        bus.publish(
            _PROBE_CHANNEL,
            json.dumps({"room": room, "marker": "apres"}),
        )
        _late_status, late_body = _poll_socket(http_session, base, sid, timeout=3)
        assert "apres" not in late_body
        _again, _again_sid, again_body = _open_socket(base, token)
        assert f"company_{company.id}" not in again_body
    finally:
        if http_session is not None:
            http_session.close()
        process.terminate()
        with suppress(Exception):
            process.wait(timeout=10)
        if process.poll() is None:
            process.kill()
        _delete_account(portal_grant_table.engine, user.id)
