"""Post-mortem canary 135 #2 — PUT vs DLE gelé (docker backend)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app import create_app
from models import db

DRIVER = 20135
SESS = "trk_sess_1786985556979_ypmkdr5z"
HOME_START = datetime.fromisoformat("2026-08-17T16:55:06+00:00")


def main() -> None:
    app = create_app()
    with app.app_context():
        from ext import redis_client

        print("POSTMORTEM", datetime.now(UTC).isoformat())
        print("SESS", SESS, "HOME_START", HOME_START.isoformat())

        dle_n = db.session.execute(
            text(
                """
                SELECT COUNT(1), MIN(sequence_id), MAX(sequence_id),
                       MAX(created_at)
                FROM driver_location_events
                WHERE tracking_session_id = :s
                """
            ),
            {"s": SESS},
        ).first()
        print("DLE_SESSION", dle_n)

        recent = db.session.execute(
            text(
                """
                SELECT id, sequence_id, location_event_id, created_at, recorded_at
                FROM driver_location_events
                WHERE driver_id = :d
                ORDER BY id DESC LIMIT 8
                """
            ),
            {"d": DRIVER},
        ).mappings().all()
        for r in recent:
            print("DLE_ROW", dict(r))

        # outbox / pending if table exists
        for tbl in (
            "location_outbox",
            "driver_location_outbox",
            "tracking_outbox",
        ):
            try:
                n = db.session.execute(text(f"SELECT COUNT(1) FROM {tbl}")).scalar()
                print(f"TABLE {tbl} count={n}")
            except Exception as e:
                db.session.rollback()
                print(f"TABLE {tbl} NA {type(e).__name__}")

        # DLQ / conflict ledger if any
        for tbl in (
            "location_event_dlq",
            "tracking_event_dlq",
            "driver_location_event_conflicts",
        ):
            try:
                rows = db.session.execute(
                    text(
                        f"""
                        SELECT * FROM {tbl}
                        WHERE created_at >= :t
                        ORDER BY id DESC LIMIT 5
                        """
                    ),
                    {"t": HOME_START},
                ).mappings().all()
                print(f"DLQ_TBL {tbl} n={len(rows)}")
                for r in rows[:3]:
                    print(" ", {k: r[k] for k in list(r.keys())[:8]})
            except Exception as e:
                db.session.rollback()
                print(f"DLQ_TBL {tbl} NA {type(e).__name__}")

        # idempotency / first-seen for latest eids of session
        try:
            rows = db.session.execute(
                text(
                    """
                    SELECT location_event_id, sequence_id, created_at
                    FROM driver_location_events
                    WHERE tracking_session_id = :s
                    ORDER BY sequence_id DESC LIMIT 3
                    """
                ),
                {"s": SESS},
            ).mappings().all()
            for r in rows:
                eid = r["location_event_id"]
                # try common idempotency tables
                for itbl, col in (
                    ("location_event_idempotency", "event_id"),
                    ("driver_location_idempotency", "event_id"),
                    ("tracking_event_first_seen", "event_id"),
                ):
                    try:
                        hit = db.session.execute(
                            text(f"SELECT * FROM {itbl} WHERE {col}=:e LIMIT 1"),
                            {"e": eid},
                        ).mappings().first()
                        print(f"IDEM {itbl} {eid}", dict(hit) if hit else None)
                    except Exception:
                        db.session.rollback()
        except Exception as e:
            print("IDEM_ERR", e)

        key = f"driver:{DRIVER}:loc:canonical"
        canon = {
            (k.decode() if isinstance(k, bytes) else str(k)): (
                v.decode() if isinstance(v, bytes) else str(v)
            )
            for k, v in (redis_client.hgetall(key) or {}).items()
        }
        print("CANON", canon.get("sequence_id"), canon.get("location_event_id"), "ttl", redis_client.ttl(key))

        # session still active?
        a = db.session.execute(
            text(
                """
                SELECT tracking_session_id, status, started_at
                FROM tracking_sessions WHERE driver_id=:d ORDER BY id DESC LIMIT 2
                """
            ),
            {"d": DRIVER},
        ).mappings().all()
        for x in a:
            print("SESS_ROW", dict(x))


if __name__ == "__main__":
    main()
