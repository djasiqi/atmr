"""Gate stabilité session — canary Q3 (build ≥133) AVANT tout re-GO PG_FIRST.

Échantillonne périodiquement pendant STABLE_SEC (défaut 75s) :

- tracking_session active inchangée
- aucune nouvelle ligne tracking_sessions (POST /sessions spontané)
- nouvelles DLE uniquement sur cette session
- aucune DLE sur session superseded pendant la fenêtre
- location_event_id / capture_id uniques sur les nouvelles DLE
- seq max augmente (ou au moins MIN_NEW_DLE nouveaux events)
- PG_FIRST doit rester false pendant le gate Q3

Exit 0 = STABLE_Q3_PASS (alors seulement envisager re-GO P5-B)
Exit 2 = UNSTABLE / NOT_READY
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from app import create_app
from sqlalchemy import text

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
MISSION_ID = int(os.getenv("P0E_MISSION_ID", "38243"))
STABLE_SEC = int(os.getenv("P0E_STABLE_SEC", "75"))
POLL_SEC = float(os.getenv("P0E_POLL_SEC", "5"))
MIN_NEW_DLE = int(os.getenv("P0E_MIN_NEW_DLE", "3"))


def _now() -> datetime:
    return datetime.now(UTC)


app = create_app()
with app.app_context():
    from models import db

    pg_first = os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "false")
    print("GATE_SESSION_STABILITY_Q3")
    print(f"  now={_now().isoformat()}")
    print(f"  driver_id={DRIVER_ID} mission_id={MISSION_ID}")
    print(f"  stable_sec={STABLE_SEC} poll_sec={POLL_SEC} min_new_dle={MIN_NEW_DLE}")
    print(f"  PG_FIRST={pg_first}")

    if str(pg_first).lower() not in ("false", "0", "no", ""):
        print("VERDICT UNSTABLE pg_first_must_be_off")
        raise SystemExit(2)

    active = db.session.execute(
        text(
            """
            SELECT id, tracking_session_id, session_generation, status, started_at
            FROM tracking_sessions
            WHERE driver_id=:d AND status='active'
            ORDER BY id DESC LIMIT 1
            """
        ),
        {"d": DRIVER_ID},
    ).mappings().first()
    if not active:
        print("VERDICT UNSTABLE no_active_session")
        raise SystemExit(2)

    sid = active["tracking_session_id"]
    gen = active["session_generation"]
    anchor_session_row_id = int(active["id"])
    print(
        f"ANCHOR sid={sid} gen={gen} row_id={anchor_session_row_id} "
        f"started_at={active['started_at']}"
    )

    baseline = db.session.execute(
        text(
            """
            SELECT COALESCE(MAX(id), 0) AS max_id,
                   COALESCE(MAX(sequence_id), 0) AS max_seq,
                   COUNT(*) AS n
            FROM driver_location_events
            WHERE driver_id=:d AND tracking_session_id=:s
            """
        ),
        {"d": DRIVER_ID, "s": sid},
    ).mappings().first()
    print(f"BASELINE {dict(baseline)}")
    if int(baseline["n"] or 0) < 1:
        print("VERDICT UNSTABLE active_has_zero_dle_at_start")
        raise SystemExit(2)

    t0 = time.monotonic()
    deadline = t0 + STABLE_SEC
    last_max_id = int(baseline["max_id"])
    samples = 0

    while time.monotonic() < deadline:
        time.sleep(POLL_SEC)
        samples += 1
        db.session.expire_all()

        cur = db.session.execute(
            text(
                """
                SELECT tracking_session_id, session_generation, status
                FROM tracking_sessions
                WHERE driver_id=:d AND status='active'
                ORDER BY id DESC LIMIT 1
                """
            ),
            {"d": DRIVER_ID},
        ).mappings().first()
        if not cur or cur["tracking_session_id"] != sid:
            print(
                "VERDICT UNSTABLE active_rotated",
                f"expected={sid}",
                f"got={dict(cur) if cur else None}",
            )
            raise SystemExit(2)

        # POST /tracking/sessions spontané = nouvelle ligne tracking_sessions
        spawned = db.session.execute(
            text(
                """
                SELECT id, tracking_session_id, status, created_at
                FROM tracking_sessions
                WHERE driver_id=:d AND id > :anchor_id
                ORDER BY id ASC
                """
            ),
            {"d": DRIVER_ID, "anchor_id": anchor_session_row_id},
        ).mappings().all()
        if spawned:
            print(
                "VERDICT UNSTABLE spontaneous_tracking_sessions",
                [dict(r) for r in spawned],
            )
            raise SystemExit(2)

        foreign = db.session.execute(
            text(
                """
                SELECT tracking_session_id, COUNT(*) AS n, MIN(id) AS min_id, MAX(id) AS max_id
                FROM driver_location_events
                WHERE driver_id=:d AND id > :min_id
                  AND tracking_session_id <> :s
                GROUP BY tracking_session_id
                """
            ),
            {"d": DRIVER_ID, "min_id": int(baseline["max_id"]), "s": sid},
        ).mappings().all()
        if foreign:
            print("VERDICT UNSTABLE dle_on_other_sessions", [dict(r) for r in foreign])
            raise SystemExit(2)

        superseded_hits = db.session.execute(
            text(
                """
                SELECT e.tracking_session_id, COUNT(*) AS n
                FROM driver_location_events e
                JOIN tracking_sessions ts
                  ON ts.tracking_session_id = e.tracking_session_id
                 AND ts.driver_id = e.driver_id
                WHERE e.driver_id=:d AND e.id > :min_id
                  AND ts.status = 'superseded'
                GROUP BY e.tracking_session_id
                """
            ),
            {"d": DRIVER_ID, "min_id": int(baseline["max_id"])},
        ).mappings().all()
        if superseded_hits:
            print(
                "VERDICT UNSTABLE dle_on_superseded",
                [dict(r) for r in superseded_hits],
            )
            raise SystemExit(2)

        prog = db.session.execute(
            text(
                """
                SELECT COALESCE(MAX(id), 0) AS max_id,
                       COALESCE(MAX(sequence_id), 0) AS max_seq,
                       COUNT(*) FILTER (WHERE id > :min_id) AS new_n
                FROM driver_location_events
                WHERE driver_id=:d AND tracking_session_id=:s
                """
            ),
            {"d": DRIVER_ID, "s": sid, "min_id": int(baseline["max_id"])},
        ).mappings().first()
        last_max_id = int(prog["max_id"])
        print(
            f"SAMPLE {samples} elapsed={time.monotonic()-t0:.0f}s",
            f"max_id={prog['max_id']} max_seq={prog['max_seq']} new_n={prog['new_n']}",
        )

    final_new = db.session.execute(
        text(
            """
            SELECT COUNT(*) AS new_n,
                   COALESCE(MAX(sequence_id), 0) AS max_seq,
                   COUNT(DISTINCT location_event_id) AS distinct_eid,
                   COUNT(DISTINCT capture_id) AS distinct_capture
            FROM driver_location_events
            WHERE driver_id=:d AND tracking_session_id=:s AND id > :min_id
            """
        ),
        {"d": DRIVER_ID, "s": sid, "min_id": int(baseline["max_id"])},
    ).mappings().first()
    new_n = int(final_new["new_n"] or 0)
    print(
        f"FINAL new_n={new_n} max_seq={final_new['max_seq']} "
        f"distinct_eid={final_new['distinct_eid']} "
        f"distinct_capture={final_new['distinct_capture']} "
        f"last_max_id={last_max_id}"
    )
    if new_n < MIN_NEW_DLE:
        print("VERDICT UNSTABLE insufficient_new_dle_on_active")
        raise SystemExit(2)
    if int(final_new["distinct_eid"] or 0) != new_n:
        print("VERDICT UNSTABLE duplicate_location_event_id")
        raise SystemExit(2)
    if int(final_new["distinct_capture"] or 0) != new_n:
        print("VERDICT UNSTABLE duplicate_capture_id")
        raise SystemExit(2)

    print("VERDICT STABLE_Q3_PASS")
    print(f"  sid={sid} gen={gen} new_dle={new_n} window_sec={STABLE_SEC}")
    print("  PG_FIRST remains OFF — re-GO P5-B only after explicit GO")
    raise SystemExit(0)
