"""Pré-canary gate post pm clear — session neuve + DLE (docker backend)."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from app import create_app
from models import db

DRIVER_ID = 20135
# pm clear ~18:48 local UTC+2 → 16:48Z
CLEAR_MARK = datetime.fromisoformat("2026-08-17T16:48:00+00:00")


def main() -> None:
    app = create_app()
    with app.app_context():
        from ext import redis_client
        from services.company_driver_locations import build_company_driver_locations_items

        print("PRECANARY_START", datetime.now(UTC).isoformat())
        print("CLEAR_MARK", CLEAR_MARK.isoformat())

        active = db.session.execute(
            text(
                """
                SELECT id, tracking_session_id, session_generation, status,
                       started_at
                FROM tracking_sessions
                WHERE driver_id = :d AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """
            ),
            {"d": DRIVER_ID},
        ).mappings().first()
        print("ACTIVE", dict(active) if active else None)
        if not active:
            print("GATE FAIL no_active_session")
            return

        sid = active["tracking_session_id"]
        started = active["started_at"]
        if started is not None and started.tzinfo is None:
            started_cmp = started.replace(tzinfo=UTC)
        else:
            started_cmp = started
        post_clear = bool(started_cmp and started_cmp >= CLEAR_MARK)
        print("SESSION_POST_CLEAR", post_clear, "started_at", started)

        dle = db.session.execute(
            text(
                """
                SELECT COUNT(1) AS n,
                       COALESCE(MIN(sequence_id), 0) AS min_seq,
                       COALESCE(MAX(sequence_id), 0) AS max_seq,
                       MIN(created_at) AS first_at,
                       MAX(created_at) AS last_at
                FROM driver_location_events
                WHERE tracking_session_id = :s
                """
            ),
            {"s": sid},
        ).mappings().first()
        print(
            "DLE",
            f"n={dle['n']}",
            f"min_seq={dle['min_seq']}",
            f"max_seq={dle['max_seq']}",
            f"first={dle['first_at']}",
            f"last={dle['last_at']}",
        )

        samples = db.session.execute(
            text(
                """
                SELECT location_event_id AS event_id, sequence_id, created_at,
                       recorded_at
                FROM driver_location_events
                WHERE tracking_session_id = :s
                ORDER BY sequence_id ASC
                LIMIT 5
                """
            ),
            {"s": sid},
        ).mappings().all()
        for s in samples:
            print("SAMPLE", dict(s))

        # Any DLE for this driver after clear belonging to OTHER sessions?
        other = db.session.execute(
            text(
                """
                SELECT tracking_session_id, COUNT(1) AS n
                FROM driver_location_events
                WHERE driver_id = :d
                  AND created_at >= :mark
                  AND tracking_session_id <> :s
                GROUP BY tracking_session_id
                ORDER BY n DESC
                LIMIT 5
                """
            ),
            {"d": DRIVER_ID, "mark": CLEAR_MARK.replace(tzinfo=None), "s": sid},
        ).mappings().all()
        print("OTHER_SESS_DLE_POST_CLEAR", [dict(x) for x in other])

        key = f"driver:{DRIVER_ID}:loc:canonical"
        canon = redis_client.hgetall(key) or {}
        c = {
            (k.decode() if isinstance(k, bytes) else str(k)): (
                v.decode() if isinstance(v, bytes) else str(v)
            )
            for k, v in canon.items()
        }
        ttl = redis_client.ttl(key)
        print(
            "CANON",
            f"seq={c.get('sequence_id')}",
            f"sess={c.get('tracking_session_id')}",
            f"eid={c.get('location_event_id')}",
            f"ttl={ttl}",
        )

        company_id = db.session.execute(
            text("SELECT company_id FROM driver WHERE id=:d"), {"d": DRIVER_ID}
        ).scalar()
        items = build_company_driver_locations_items(
            int(company_id or 1), is_demo_company=False
        )
        hit = [
            i
            for i in items
            if int(i.get("driver_id") or i.get("id") or 0) == DRIVER_ID
        ]
        rest = hit[0] if hit else {}
        print(
            "REST",
            f"status={rest.get('location_status')}",
            f"age={rest.get('last_seen_seconds')}",
            f"src={rest.get('position_source')}",
        )

        ok_session = post_clear and active["status"] == "active"
        ok_dle = int(dle["n"] or 0) > 0
        ok_seq = int(dle["min_seq"] or 0) >= 1 and int(dle["max_seq"] or 0) >= int(
            dle["min_seq"] or 0
        )
        ok_other = len(other) == 0
        ok_canon = c.get("tracking_session_id") == sid
        print(
            "GATE",
            f"session_post_clear={ok_session}",
            f"dle_gt0={ok_dle}",
            f"seq_ok={ok_seq}",
            f"no_other_sess={ok_other}",
            f"canon_match={ok_canon}",
        )
        if ok_session and ok_dle and ok_seq and ok_other:
            print("PRECANARY PASS")
        else:
            print("PRECANARY FAIL")


if __name__ == "__main__":
    main()
