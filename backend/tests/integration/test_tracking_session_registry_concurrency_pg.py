"""Tests registre sessions tracking — ouverture idempotente + concurrence PG."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import UTC, datetime
from threading import Barrier

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from services.tracking.session_registry import (
    SESSION_REGISTRY_LOCK_NAMESPACE,
    register_tracking_session,
)
from tests.factories import create_driver_with_position

REGISTRY_NS = SESSION_REGISTRY_LOCK_NAMESPACE


def _pg_dsn() -> str | None:
    """DSN PostgreSQL direct (pas PgBouncer) pour tests concurrence / advisory."""
    for key in (
        "POSTGRES_URL",
        "DATABASE_URL_DIRECT",
        "SQLALCHEMY_DATABASE_URI_DIRECT",
        "DATABASE_URL_TEST",
        "DATABASE_URL",
        "SQLALCHEMY_DATABASE_URI",
    ):
        url = os.getenv(key)
        if url:
            break
    else:
        return None
    url = (
        url.replace("postgresql+psycopg://", "postgresql://")
        .replace("postgres://", "postgresql://")
        .replace("postgresql+psycopg2://", "postgresql://")
    )
    if "@pgbouncer:" in url or "@atmr-pgbouncer" in url:
        url = url.replace("@pgbouncer:", "@postgres:").replace(
            "@atmr-pgbouncer:", "@postgres:"
        )
        url = url.replace(":6432/", ":5432/")
    return url


def _psycopg_connect(dsn: str):
    try:
        import psycopg

        return psycopg.connect(dsn)
    except ImportError:
        try:
            import psycopg2 as psycopg  # type: ignore

            return psycopg.connect(dsn)
        except ImportError:
            pytest.skip("psycopg unavailable")


def _count_sessions(session: Session, driver_id: int, sid: str) -> int:
    return int(
        session.execute(
            text(
                """
                SELECT COUNT(*) FROM tracking_sessions
                WHERE driver_id = :d AND tracking_session_id = :sid
                """
            ),
            {"d": driver_id, "sid": sid},
        ).scalar_one()
    )


def _count_state(session: Session, driver_id: int, sid: str) -> int:
    return int(
        session.execute(
            text(
                """
                SELECT COUNT(*) FROM tracking_session_state
                WHERE driver_id = :d AND tracking_session_id = :sid
                """
            ),
            {"d": driver_id, "sid": sid},
        ).scalar_one()
    )


def _status_counts(session: Session, driver_id: int) -> dict[str, int]:
    rows = session.execute(
        text(
            """
            SELECT status, COUNT(*) AS n FROM tracking_sessions
            WHERE driver_id = :d GROUP BY status
            """
        ),
        {"d": driver_id},
    ).mappings()
    return {str(r["status"]): int(r["n"]) for r in rows}


def _insert_user_psycopg(cur, *, suffix: str, role: str) -> int:
    cur.execute(
        """
        INSERT INTO "user" (
            public_id, username, email, password, role,
            force_password_change, encryption_migrated
        ) VALUES (
            %s, %s, %s, 'x', %s::user_role,
            false, false
        )
        RETURNING id
        """,
        (
            str(uuid.uuid4()),
            f"u_{role.lower()}_{suffix}",
            f"{role.lower()}_{suffix}@atmr-test.ch",
            role,
        ),
    )
    return int(cur.fetchone()[0])


def _create_committed_driver(dsn: str, tag: str) -> tuple[int, int, list[int]]:
    """Crée user+company+driver via psycopg (évite listeners SA hors app context)."""
    suffix = f"{tag}_{uuid.uuid4().hex[:10]}"
    conn = _psycopg_connect(dsn)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            user_company = _insert_user_psycopg(cur, suffix=suffix, role="COMPANY")
            cur.execute(
                "INSERT INTO company (name, user_id) VALUES (%s, %s) RETURNING id",
                (f"Co {suffix}"[:100], user_company),
            )
            company_id = int(cur.fetchone()[0])
            user_driver = _insert_user_psycopg(cur, suffix=suffix, role="DRIVER")
            cur.execute(
                "INSERT INTO driver (user_id, company_id) VALUES (%s, %s) RETURNING id",
                (user_driver, company_id),
            )
            driver_id = int(cur.fetchone()[0])
        conn.commit()
        return driver_id, company_id, [user_company, user_driver]
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _cleanup_driver(
    dsn: str, *, driver_id: int, company_id: int, user_ids: list[int]
) -> None:
    conn = _psycopg_connect(dsn)
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM tracking_session_state WHERE driver_id = %s", (driver_id,)
            )
            cur.execute(
                "DELETE FROM tracking_sessions WHERE driver_id = %s", (driver_id,)
            )
            cur.execute("DELETE FROM driver WHERE id = %s", (driver_id,))
            cur.execute("DELETE FROM company WHERE id = %s", (company_id,))
            for uid in user_ids:
                cur.execute('DELETE FROM "user" WHERE id = %s', (uid,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@pytest.mark.integration
def test_a_sequential_same_sid(db, sample_company):
    """TEST A — register X deux fois → même génération, 1 session, 1 state."""
    driver = create_driver_with_position(company=sample_company)
    db.session.flush()
    driver_id = int(driver.id)
    company_id = int(sample_company.id)
    sid = f"trk_test_a_{uuid.uuid4().hex[:12]}"

    r1 = register_tracking_session(
        db.session,
        driver_id=driver_id,
        company_id=company_id,
        tracking_session_id=sid,
        tracking_session_started_at="2026-08-11T20:00:00.000Z",
    )
    r2 = register_tracking_session(
        db.session,
        driver_id=driver_id,
        company_id=company_id,
        tracking_session_id=sid,
        tracking_session_started_at="2026-08-11T20:01:00.000Z",
    )
    assert r1["session_generation"] == r2["session_generation"]
    assert _count_sessions(db.session, driver_id, sid) == 1
    assert _count_state(db.session, driver_id, sid) == 1


@pytest.mark.integration
def test_b_concurrent_same_sid(app):
    """TEST B — deux TX concurrentes même SID → même gen, 1 session, 1 state."""
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip("DATABASE_URL absent")

    driver_id, company_id, user_ids = _create_committed_driver(dsn, "b")
    sid = f"trk_test_b_{uuid.uuid4().hex[:12]}"
    barrier = Barrier(2)
    results: list[dict] = []
    errors: list[BaseException] = []

    def _register() -> dict:
        barrier.wait(timeout=30)
        with app.app_context():
            engine = create_engine(dsn)
            SessionLocal = sessionmaker(bind=engine)
            try:
                with SessionLocal() as session:
                    try:
                        out = register_tracking_session(
                            session,
                            driver_id=driver_id,
                            company_id=company_id,
                            tracking_session_id=sid,
                            tracking_session_started_at="2026-08-11T20:09:12.558Z",
                        )
                        session.commit()
                        return out
                    except BaseException:
                        session.rollback()
                        raise
            finally:
                engine.dispose()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_register) for _ in range(2)]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except BaseException as exc:
                    errors.append(exc)

        assert not errors, errors
        assert len(results) == 2
        assert results[0]["session_generation"] == results[1]["session_generation"]

        with app.app_context():
            engine = create_engine(dsn)
            try:
                with sessionmaker(bind=engine)() as session:
                    assert _count_sessions(session, driver_id, sid) == 1
                    assert _count_state(session, driver_id, sid) == 1
            finally:
                engine.dispose()
    finally:
        _cleanup_driver(
            dsn, driver_id=driver_id, company_id=company_id, user_ids=user_ids
        )


@pytest.mark.integration
def test_c_concurrent_two_sids_one_active(app):
    """TEST C — X et Y concurrents → generations distinctes, 1 active, 1 superseded."""
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip("DATABASE_URL absent")

    driver_id, company_id, user_ids = _create_committed_driver(dsn, "c")
    sid_x = f"trk_test_c_x_{uuid.uuid4().hex[:10]}"
    sid_y = f"trk_test_c_y_{uuid.uuid4().hex[:10]}"
    barrier = Barrier(2)
    results: dict[str, dict] = {}
    errors: list[BaseException] = []

    def _register(sid: str) -> tuple[str, dict]:
        barrier.wait(timeout=30)
        with app.app_context():
            engine = create_engine(dsn)
            SessionLocal = sessionmaker(bind=engine)
            try:
                with SessionLocal() as session:
                    try:
                        out = register_tracking_session(
                            session,
                            driver_id=driver_id,
                            company_id=company_id,
                            tracking_session_id=sid,
                            tracking_session_started_at="2026-08-11T20:09:12.558Z",
                        )
                        session.commit()
                        return sid, out
                    except BaseException:
                        session.rollback()
                        raise
            finally:
                engine.dispose()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_register, sid_x), pool.submit(_register, sid_y)]
            for fut in as_completed(futures):
                try:
                    sid, out = fut.result()
                    results[sid] = out
                except BaseException as exc:
                    errors.append(exc)

        assert not errors, errors
        assert set(results) == {sid_x, sid_y}
        assert (
            results[sid_x]["session_generation"] != results[sid_y]["session_generation"]
        )

        with app.app_context():
            engine = create_engine(dsn)
            try:
                with sessionmaker(bind=engine)() as session:
                    counts = _status_counts(session, driver_id)
                    assert counts.get("active", 0) == 1
                    assert counts.get("superseded", 0) == 1
                    assert _count_state(session, driver_id, sid_x) == 1
                    assert _count_state(session, driver_id, sid_y) == 1
            finally:
                engine.dispose()
    finally:
        _cleanup_driver(
            dsn, driver_id=driver_id, company_id=company_id, user_ids=user_ids
        )


@pytest.mark.integration
def test_d_multi_driver_lock_keys_isolated():
    """TEST D — lock (42002, driver100) ≠ (42002, driver200) ; pas de timing."""
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip("DATABASE_URL absent")

    driver_a = 9100100
    driver_b = 9100200
    conn_a = _psycopg_connect(dsn)
    conn_b = _psycopg_connect(dsn)
    try:
        conn_a.autocommit = False
        conn_b.autocommit = False
        with conn_a.cursor() as cur:
            cur.execute("BEGIN")
            cur.execute(
                "SELECT pg_advisory_xact_lock(%s, %s)",
                (REGISTRY_NS, driver_a),
            )

        with conn_b.cursor() as cur:
            cur.execute("BEGIN")
            cur.execute(
                "SELECT pg_try_advisory_xact_lock(%s, %s)",
                (REGISTRY_NS, driver_b),
            )
            got_other = cur.fetchone()[0]
            cur.execute(
                "SELECT pg_try_advisory_xact_lock(%s, %s)",
                (REGISTRY_NS, driver_a),
            )
            got_same = cur.fetchone()[0]

        assert got_other is True
        assert got_same is False
    finally:
        with suppress(Exception):
            conn_a.rollback()
        with suppress(Exception):
            conn_b.rollback()
        conn_a.close()
        conn_b.close()


@pytest.mark.integration
def test_e_repair_missing_state(db, sample_company):
    """TEST E — session sans state → register répare le state avec génération canonique."""
    driver = create_driver_with_position(company=sample_company)
    db.session.flush()
    driver_id = int(driver.id)
    company_id = int(sample_company.id)
    sid = f"trk_test_e_{uuid.uuid4().hex[:12]}"
    started = datetime(2026, 8, 11, 20, 0, 0, tzinfo=UTC)
    generation = 77

    db.session.execute(
        text(
            """
            INSERT INTO tracking_sessions (
                driver_id, company_id, tracking_session_id, session_generation,
                status, started_at
            ) VALUES (
                :d, :c, :sid, :gen, 'active', :started
            )
            """
        ),
        {
            "d": driver_id,
            "c": company_id,
            "sid": sid,
            "gen": generation,
            "started": started,
        },
    )
    assert _count_state(db.session, driver_id, sid) == 0

    out = register_tracking_session(
        db.session,
        driver_id=driver_id,
        company_id=company_id,
        tracking_session_id=sid,
        tracking_session_started_at="2026-08-11T21:00:00.000Z",
    )
    assert out["session_generation"] == generation
    assert _count_sessions(db.session, driver_id, sid) == 1
    assert _count_state(db.session, driver_id, sid) == 1

    state = (
        db.session.execute(
            text(
                """
                SELECT session_generation, first_seen_at
                FROM tracking_session_state
                WHERE driver_id = :d AND tracking_session_id = :sid
                """
            ),
            {"d": driver_id, "sid": sid},
        )
        .mappings()
        .one()
    )
    assert int(state["session_generation"]) == generation
    first_seen = state["first_seen_at"]
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=UTC)
    assert first_seen == started
