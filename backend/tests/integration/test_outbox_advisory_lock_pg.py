"""Test intégration P0-5 : advisory lock 2-int sur vraie PG (Docker)."""

from __future__ import annotations

import os

import pytest

OUTBOX_LOCK_NAMESPACE = int(os.getenv("TRACKING_OUTBOX_LOCK_NAMESPACE", "42001"))


def _pg_dsn() -> str | None:
    """DSN session directe (pas PgBouncer transaction) pour advisory locks."""
    # Préférer l'URL postgres directe si dispo
    for key in (
        "POSTGRES_URL",
        "DATABASE_URL_DIRECT",
        "SQLALCHEMY_DATABASE_URI_DIRECT",
    ):
        url = os.getenv(key)
        if url:
            break
    else:
        url = os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")
    if not url:
        return None
    url = (
        url.replace("postgresql+psycopg://", "postgresql://")
        .replace("postgres://", "postgresql://")
        .replace("postgresql+psycopg2://", "postgresql://")
    )
    # Contournement pgbouncer : host postgres:5432
    if "@pgbouncer:" in url or "@atmr-pgbouncer" in url:
        url = url.replace("@pgbouncer:", "@postgres:").replace(
            "@atmr-pgbouncer:", "@postgres:"
        )
        # Port pgbouncer 6432 → 5432
        url = url.replace(":6432/", ":5432/")
    return url


@pytest.mark.integration
def test_advisory_lock_two_connections_same_driver():
    """Utilise psycopg brut pour éviter les listeners Flask/SQLAlchemy."""
    dsn = _pg_dsn()
    if not dsn:
        pytest.skip("DATABASE_URL absent")
    try:
        import psycopg
    except ImportError:
        try:
            import psycopg2 as psycopg  # type: ignore
        except ImportError:
            pytest.skip("psycopg unavailable")

    connect = psycopg.connect
    # Nettoyage préventif d'un lock orphelin (crash test précédent).
    cleanup = connect(dsn)
    try:
        cleanup.autocommit = True
        with cleanup.cursor() as cur:
            cur.execute(
                """
                SELECT pg_terminate_backend(l.pid)
                FROM pg_locks l
                WHERE l.locktype = 'advisory'
                  AND l.classid = %s AND l.objid = %s
                  AND l.pid <> pg_backend_pid()
                """,
                (OUTBOX_LOCK_NAMESPACE, 42),
            )
    finally:
        cleanup.close()

    conn_a = connect(dsn)
    conn_b = connect(dsn)
    try:
        conn_a.autocommit = True
        conn_b.autocommit = True
        with conn_a.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                (OUTBOX_LOCK_NAMESPACE, 42),
            )
            got_a = cur.fetchone()[0]
        assert got_a is True

        with conn_b.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                (OUTBOX_LOCK_NAMESPACE, 42),
            )
            got_b = cur.fetchone()[0]
        assert got_b is False

        with conn_a.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(%s, %s)",
                (OUTBOX_LOCK_NAMESPACE, 42),
            )
            unlocked = cur.fetchone()[0]
        assert unlocked is True

        with conn_b.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s, %s)",
                (OUTBOX_LOCK_NAMESPACE, 42),
            )
            got_b2 = cur.fetchone()[0]
            assert got_b2 is True
            cur.execute(
                "SELECT pg_advisory_unlock(%s, %s)",
                (OUTBOX_LOCK_NAMESPACE, 42),
            )
    finally:
        conn_a.close()
        conn_b.close()
