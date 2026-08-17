"""FG PRE-GATE canary 135 #3 — prouver de NOUVELLES captures avant HOME.

Attend ≥ min_new event_id / seq sur la session active, DLE+canonical qui avancent,
REST live/recent, conflict=0. Sortie PASS/FAIL explicite.
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from sqlalchemy import text

from app import create_app
from models import db

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
EXPECT_SESS = os.getenv("P0E_EXPECT_SESS", "trk_sess_1786985556979_ypmkdr5z")
MIN_NEW = int(os.getenv("P0E_FG_MIN_NEW", "3"))
WINDOW = int(os.getenv("P0E_FG_SEC", "90"))
POLL = float(os.getenv("P0E_POLL_SEC", "5"))
BASELINE_SEQ = os.getenv("P0E_BASELINE_SEQ")  # optional lock; else snap at start


def dec(raw):
    out = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


def main() -> int:
    app = create_app()
    with app.app_context():
        from ext import redis_client
        from services.company_driver_locations import build_company_driver_locations_items
        from services.tracking.location_candidate import is_pg_first_canonical_enabled

        print("FG_PREGATE_START", datetime.now(UTC).isoformat())
        print(
            f"driver={DRIVER_ID} expect_sess={EXPECT_SESS} "
            f"min_new={MIN_NEW} window={WINDOW}s poll={POLL}s"
        )
        print("PG_FIRST", is_pg_first_canonical_enabled())

        active = db.session.execute(
            text(
                """
                SELECT tracking_session_id, session_generation, status, started_at
                FROM tracking_sessions
                WHERE driver_id=:d AND status='active'
                ORDER BY id DESC LIMIT 1
                """
            ),
            {"d": DRIVER_ID},
        ).mappings().first()
        print("ACTIVE", dict(active) if active else None)
        if not active:
            print("FG_PREGATE FAIL no_active_session")
            return 2
        sid = active["tracking_session_id"]
        if EXPECT_SESS and sid != EXPECT_SESS:
            print(f"FG_PREGATE FAIL session_mismatch got={sid} expect={EXPECT_SESS}")
            return 3

        company_id = db.session.execute(
            text("SELECT company_id FROM driver WHERE id=:d"), {"d": DRIVER_ID}
        ).scalar()

        snap = db.session.execute(
            text(
                """
                SELECT COALESCE(MAX(sequence_id),0) AS max_seq,
                       COUNT(1) AS n
                FROM driver_location_events
                WHERE tracking_session_id=:s
                """
            ),
            {"s": sid},
        ).mappings().first()
        baseline_seq = int(BASELINE_SEQ) if BASELINE_SEQ else int(snap["max_seq"] or 0)
        print(f"BASELINE_SEQ={baseline_seq} dle_n={snap['n']}")

        seen_eids: set[str] = set()
        # seed known eids at/below baseline so only NEW count
        for row in db.session.execute(
            text(
                """
                SELECT location_event_id, sequence_id, recorded_at
                FROM driver_location_events
                WHERE tracking_session_id=:s AND sequence_id <= :b
                ORDER BY sequence_id
                """
            ),
            {"s": sid, "b": baseline_seq},
        ).mappings():
            seen_eids.add(row["location_event_id"])

        t0 = time.time()
        n = 0
        last_seq = baseline_seq
        last_canon = -1
        rest_statuses: list[str] = []
        new_eids: list[str] = []
        last_recorded = None

        while True:
            n += 1
            elapsed = int(time.time() - t0)
            db.session.expire_all()

            row = db.session.execute(
                text(
                    """
                    SELECT id, sequence_id, location_event_id, recorded_at, created_at
                    FROM driver_location_events
                    WHERE tracking_session_id=:s
                    ORDER BY sequence_id DESC LIMIT 1
                    """
                ),
                {"s": sid},
            ).mappings().first()

            # collect new eids above baseline
            fresh = db.session.execute(
                text(
                    """
                    SELECT location_event_id, sequence_id, recorded_at
                    FROM driver_location_events
                    WHERE tracking_session_id=:s AND sequence_id > :b
                    ORDER BY sequence_id ASC
                    """
                ),
                {"s": sid, "b": baseline_seq},
            ).mappings().all()
            for f in fresh:
                eid = f["location_event_id"]
                if eid not in seen_eids:
                    seen_eids.add(eid)
                    new_eids.append(eid)
                    print(
                        f"NEW_EID seq={f['sequence_id']} eid={eid} "
                        f"recorded_at={f['recorded_at']}"
                    )

            key = f"driver:{DRIVER_ID}:loc:canonical"
            canon = dec(redis_client.hgetall(key) or {})
            ttl = redis_client.ttl(key)
            c_seq = int(canon.get("sequence_id") or -1) if canon else -1
            c_sess = canon.get("tracking_session_id") or ""
            last_canon = c_seq

            items = build_company_driver_locations_items(
                int(company_id or 1), is_demo_company=False
            )
            hit = [
                i
                for i in items
                if int(i.get("driver_id") or i.get("id") or 0) == DRIVER_ID
            ]
            rest = hit[0] if hit else {}
            st = str(rest.get("location_status") or "")
            age = rest.get("last_seen_seconds")
            rest_statuses.append(st)

            if row:
                last_seq = int(row["sequence_id"])
                last_recorded = row["recorded_at"]

            print(
                f"SAMPLE n={n} t=+{elapsed}s "
                f"dle_seq={row['sequence_id'] if row else None} "
                f"dle_eid={row['location_event_id'] if row else None} "
                f"recorded_at={row['recorded_at'] if row else None} "
                f"new_eid_count={len(new_eids)} "
                f"canon_seq={c_seq} canon_sess={c_sess} ttl={ttl} "
                f"rest_status={st} rest_age={age}"
            )

            # early PASS
            seq_delta = last_seq - baseline_seq
            if (
                len(new_eids) >= MIN_NEW
                and seq_delta >= MIN_NEW
                and c_seq >= baseline_seq + MIN_NEW
                and c_sess == sid
                and st in ("live", "recent")
            ):
                print("EARLY_PASS criteria met")
                break

            if elapsed >= WINDOW:
                break
            time.sleep(POLL)

        seq_delta = last_seq - baseline_seq
        canon_delta = last_canon - baseline_seq if last_canon >= 0 else -1
        rest_ok = any(s in ("live", "recent") for s in rest_statuses[-3:]) or (
            rest_statuses and rest_statuses[-1] in ("live", "recent")
        )
        recorded_ok = last_recorded is not None

        print("SUMMARY")
        print(f"  baseline_seq={baseline_seq} last_seq={last_seq} seq_delta={seq_delta}")
        print(f"  new_eid_count={len(new_eids)} new_eids={new_eids}")
        print(f"  canon_last={last_canon} canon_delta_vs_baseline={canon_delta}")
        print(f"  rest_statuses_tail={rest_statuses[-5:]}")
        sess_ok = (not EXPECT_SESS) or (sid == EXPECT_SESS)
        print(f"  last_recorded_at={last_recorded}")
        print(f"  session_stable={sess_ok} sid={sid}")

        ok = (
            len(new_eids) >= MIN_NEW
            and seq_delta >= MIN_NEW
            and canon_delta >= MIN_NEW
            and rest_ok
            and recorded_ok
            and sess_ok
        )
        if ok:
            print("FG_PREGATE PASS")
            return 0
        print("FG_PREGATE FAIL")
        if len(new_eids) < MIN_NEW:
            print("  reason=insufficient_new_event_ids")
        if seq_delta < MIN_NEW:
            print("  reason=seq_not_advancing")
        if canon_delta < MIN_NEW:
            print("  reason=canonical_not_advancing")
        if not rest_ok:
            print("  reason=rest_not_live_or_recent")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
