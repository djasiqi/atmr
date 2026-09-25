"""P0-2 FINAL GATE — preuves PostgreSQL concurrence + crash Redis→DB.

Le test ``test_f_concurrent_same_r0`` (suite grace) utilise PostgreSQL via
docker-compose.test.yml, 2 threads et ``app.test_client()``, mais :
- Redis reste mocké (dict process-global) ;
- FOR UPDATE n'est pas instrumenté ;
- assertions generation / successor_count incomplètes.

Ce module ajoute les preuves exigées pour CLOSED, sans retirer test_f.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest
from flask_jwt_extended import create_refresh_token
from sqlalchemy import event, text

from ext import db
from models import RefreshToken, User
from models.enums import UserRole
from models.mobile_device_session import AuthRotationResult
from security.mobile_device_session_service import create_or_reuse_session, get_session_by_id
from security.refresh_redis_rotation import (
    RedisRefreshState,
    classify_refresh_in_redis,
    publish_refresh_redis,
    rotate_refresh_redis,
    try_repair_orphan_redis_previous,
)
from security.refresh_token_service import store_refresh_token

REFRESH_URL = "/api/v1/auth/refresh-token"


@pytest.fixture
def fail_closed(monkeypatch):
    monkeypatch.setenv("REFRESH_FAIL_CLOSED", "true")


@pytest.fixture
def gate_user(db):
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"gate_{suffix}",
        email=f"gate_{suffix}@test.local",
        public_id=str(uuid.uuid4()),
        role=UserRole.driver,
    )
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.commit()
    return user


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _headers(device_id: str, idem: str) -> dict:
    return {
        "X-Requested-With": "Expo",
        "X-Client-Platform": "ios",
        "X-Device-ID": device_id,
        "X-Auth-Contract-Version": "mobile-device-session-v1",
        "Idempotency-Key": idem,
    }


def _login(client, user, device_id: str) -> dict:
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "password123"},
        headers=_headers(device_id, str(uuid.uuid4())),
    )
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


@pytest.mark.integration
def test_audit_existing_concurrency_test_environment(app, requires_postgresql):
    """Documente l'environnement du test_f existant (preuve de lecture)."""
    with app.app_context():
        bind = db.session.get_bind()
        dialect = bind.dialect.name
        assert dialect == "postgresql"
        # test_f : 2 threads × app.test_client() → sessions SQLAlchemy thread-local.
        # SELECT FOR UPDATE : get_session_by_id(..., for_update=True) sur le path refresh.
        # Redis conftest : mock process-global (pas redis_test réel).


@pytest.mark.integration
def test_p0_2_postgres_real_concurrent_refresh_for_update(
    app, db, gate_user, fail_closed, requires_postgresql
):
    """Deux connexions PG indépendantes + FOR UPDATE réel → un seul successeur."""
    device_id = f"dev-{uuid.uuid4()}"
    with app.test_client() as bootstrap:
        login = _login(bootstrap, gate_user, device_id)
    r0 = login["refresh_token"]
    session_id = login["session_id"]
    idem = str(uuid.uuid4())

    with app.app_context():
        sess = get_session_by_id(session_id)
        assert sess is not None
        gen_n = int(sess.refresh_generation or 1)
        dialect = db.session.get_bind().dialect.name
        assert dialect == "postgresql"

    for_update_statements: list[str] = []
    for_update_connection_ids: list[int] = []
    lock = threading.Lock()

    def _before_cursor(conn, cursor, statement, parameters, context, executemany):
        sql = statement if isinstance(statement, str) else str(statement)
        if "FOR UPDATE" not in sql.upper():
            return
        if "mobile_device_session" not in sql.lower():
            return
        try:
            raw = conn.connection.dbapi_connection
            cid = id(raw)
        except Exception:
            cid = id(conn)
        with lock:
            for_update_statements.append(sql)
            for_update_connection_ids.append(cid)

    with app.app_context():
        bind = db.session.get_bind()
    assert bind is not None
    event.listen(bind, "before_cursor_execute", _before_cursor)

    barrier = Barrier(2)
    results: list[tuple[int, dict]] = []

    def _worker() -> tuple[int, dict]:
        barrier.wait(timeout=30)
        with app.app_context():
            eng = db.session.get_bind()
            with eng.connect() as probe:
                probe.execute(text("SELECT 1"))
        with app.test_client() as c:
            resp = c.post(
                REFRESH_URL,
                json={"refresh_token": r0},
                headers=_headers(device_id, idem),
            )
            return resp.status_code, resp.get_json() or {}

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(_worker) for _ in range(2)]
            for f in as_completed(futs):
                results.append(f.result())
    finally:
        event.remove(bind, "before_cursor_execute", _before_cursor)

    assert len(results) == 2
    statuses = sorted(r[0] for r in results)
    bodies = [r[1] for r in results]
    assert statuses == [200, 200], bodies

    tokens = {b.get("refresh_token") for b in bodies}
    assert len(tokens) == 1
    r1 = next(iter(tokens))
    assert r1 and r1 != r0
    gens = {b.get("refresh_generation") for b in bodies}
    assert gens == {gen_n + 1}

    assert len(for_update_statements) >= 1, "FOR UPDATE non observé"
    # Deux threads → typiquement ≥1 acquisition ; 2 si B re-lock après A.
    assert len(for_update_connection_ids) >= 1

    with app.app_context():
        db.session.remove()
        sess = get_session_by_id(session_id)
        assert sess is not None
        assert int(sess.refresh_generation) == gen_n + 1

        # Un seul successeur DB de R0 (pas de fork)
        r0_row = RefreshToken.query.filter_by(token_hash=_sha(r0)).first()
        assert r0_row is not None
        assert r0_row.rotated_to_hash == _sha(r1)

        leaf_active = (
            RefreshToken.query.filter_by(session_id=session_id, is_revoked=False)
            .filter(RefreshToken.rotated_to_hash.is_(None))
            .all()
        )
        assert len(leaf_active) == 1
        assert leaf_active[0].token_hash == _sha(r1)

        # Aucun autre token issu de R0 (pas de R2)
        pointing_from_r0 = RefreshToken.query.filter_by(
            token_hash=_sha(r1)
        ).count()
        assert pointing_from_r0 == 1

        receipts = AuthRotationResult.query.filter_by(
            session_id=uuid.UUID(session_id),
            operation_type="refresh",
        ).all()
        succ_gens = {
            int(r.successor_generation)
            for r in receipts
            if r.successor_generation is not None
        }
        assert gen_n + 1 in succ_gens
        assert gen_n + 2 not in succ_gens


@pytest.mark.integration
def test_p0_2_process_crash_redis_before_db_commit_repair(
    app, db, gate_user, fail_closed, requires_postgresql
):
    """Crash après Redis rotate, avant COMMIT → orphelin réparé → R0→R2 une fois."""
    device_id = f"dev-{uuid.uuid4()}"
    with app.app_context():
        session, _rec, _rev, _ = create_or_reuse_session(
            user_id=gate_user.id,
            device_installation_id=device_id,
            role="driver",
        )
        db.session.commit()
        session_id = str(session.session_id)
        gen_n = int(session.refresh_generation or 1)
        epoch = int(session.session_epoch or 1)

        expires = datetime.now(UTC) + timedelta(days=30)
        r0 = create_refresh_token(
            identity=str(gate_user.public_id),
            additional_claims={
                "aud": "atmr-api",
                "session_id": session_id,
                "session_epoch": epoch,
                "refresh_generation": gen_n,
                "token_version": int(getattr(gate_user, "token_version", 0) or 0),
            },
        )
        store_refresh_token(
            r0,
            gate_user.id,
            expires,
            device_id=device_id,
            sync_redis=False,
        )
        row = RefreshToken.query.filter_by(token_hash=_sha(r0)).first()
        assert row is not None
        row.session_id = session_id
        db.session.commit()
        publish_refresh_redis(gate_user.id, r0, ttl_seconds=3600)

        # Rotation Redis R0→R1 SANS commit DB (crash process simulé)
        r1_orphan = create_refresh_token(
            identity=str(gate_user.public_id),
            additional_claims={
                "aud": "atmr-api",
                "session_id": session_id,
                "session_epoch": epoch,
                "refresh_generation": gen_n + 1,
                "token_version": int(getattr(gate_user, "token_version", 0) or 0),
            },
        )
        rotate_refresh_redis(
            gate_user.id,
            r0,
            r1_orphan,
            ttl_seconds=3600,
            session_id=session_id,
        )
        assert (
            classify_refresh_in_redis(r0, user_id=gate_user.id).state
            == RedisRefreshState.PREVIOUS_WITHIN_GRACE
        )
        assert RefreshToken.query.filter_by(token_hash=_sha(r1_orphan)).first() is None
        assert (
            AuthRotationResult.query.filter_by(
                session_id=uuid.UUID(session_id), operation_type="refresh"
            ).count()
            == 0
        )
        sess = get_session_by_id(session_id)
        assert int(sess.refresh_generation) == gen_n

    # Nouvelle requête avec R0 (nouvelle connexion / session request)
    with app.test_client() as client:
        resp = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_headers(device_id, str(uuid.uuid4())),
        )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    r2 = body["refresh_token"]
    assert r2 != r0
    assert r2 != r1_orphan
    assert body.get("refresh_generation") == gen_n + 1

    with app.app_context():
        db.session.remove()
        sess = get_session_by_id(session_id)
        assert int(sess.refresh_generation) == gen_n + 1
        assert RefreshToken.query.filter_by(token_hash=_sha(r2)).first() is not None
        assert RefreshToken.query.filter_by(token_hash=_sha(r1_orphan)).first() is None
        assert (
            classify_refresh_in_redis(r2, user_id=gate_user.id).state
            == RedisRefreshState.CURRENT
        )


@pytest.mark.integration
def test_p0_2_orphan_repair_ambiguous_fail_closed(app, db, gate_user, fail_closed):
    """Si generation déjà bumpée alors que Redis previous pointe vers un fantôme → fail closed."""
    with app.app_context():
        session, _, _, _ = create_or_reuse_session(
            user_id=gate_user.id,
            device_installation_id=f"dev-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()
        r0 = create_refresh_token(
            identity=str(gate_user.public_id),
            additional_claims={"aud": "atmr-api", "refresh_generation": 1},
        )
        store_refresh_token(
            r0,
            gate_user.id,
            datetime.now(UTC) + timedelta(days=1),
            sync_redis=False,
        )
        db.session.commit()
        publish_refresh_redis(gate_user.id, r0, ttl_seconds=600)
        r1 = "orphan-never-committed"
        rotate_refresh_redis(
            gate_user.id, r0, r1, ttl_seconds=600, session_id=str(session.session_id)
        )
        session.refresh_generation = 2
        db.session.commit()
        repair = try_repair_orphan_redis_previous(
            predecessor_token=r0,
            user_id=gate_user.id,
            session_id=str(session.session_id),
            claimed_refresh_generation=1,
            db_refresh_generation=2,
            successor_hash=_sha(r1),
        )
        assert repair.repaired is False
        assert repair.ambiguous is True
