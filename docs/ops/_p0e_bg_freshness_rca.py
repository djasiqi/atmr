"""Mini-RCA background freshness — PUT/DLE/canonical/REST (séparé Q1).

Tourne côté backend (docker exec). Échantillonne toutes les POLL s pendant WINDOW s.
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from app import create_app
from sqlalchemy import text

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
WINDOW = int(os.getenv("P0E_BG_SEC", "120"))
POLL = float(os.getenv("P0E_POLL_SEC", "15"))
PHASE = os.getenv("P0E_PHASE", "UNKNOWN")


def dec(raw):
    out = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client
    from services.tracking.location_candidate import is_pg_first_canonical_enabled
    from services.company_driver_locations import build_company_driver_locations_items

    print("BG_FRESHNESS_RCA_START", datetime.now(UTC).isoformat())
    print(f"phase={PHASE} driver={DRIVER_ID} window={WINDOW}s poll={POLL}s")
    print("PG_FIRST", is_pg_first_canonical_enabled())

    company_id = db.session.execute(
        text("SELECT company_id FROM driver WHERE id=:d"), {"d": DRIVER_ID}
    ).scalar()

    t0 = time.time()
    n = 0
    first_dle = None
    last_dle = None
    first_canon_seq = None
    last_canon_seq = None
    rest_statuses = []

    while True:
        n += 1
        elapsed = int(time.time() - t0)
        db.session.expire_all()

        dle = db.session.execute(
            text(
                """
                SELECT id, sequence_id, tracking_session_id, location_event_id,
                       recorded_at, created_at, mission_id
                FROM driver_location_events
                WHERE driver_id=:d
                ORDER BY id DESC LIMIT 1
                """
            ),
            {"d": DRIVER_ID},
        ).mappings().first()

        key = f"driver:{DRIVER_ID}:loc:canonical"
        canon = dec(redis_client.hgetall(key) or {})
        ttl = redis_client.ttl(key)
        c_seq = int(canon.get("sequence_id") or -1) if canon else -1
        c_rec = canon.get("recorded_at") or canon.get("captured_at") or ""

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
        src = rest.get("position_source")
        rest_statuses.append(st)

        if dle:
            if first_dle is None:
                first_dle = dict(dle)
            last_dle = dict(dle)
        if c_seq >= 0:
            if first_canon_seq is None:
                first_canon_seq = c_seq
            last_canon_seq = c_seq

        print(
            f"SAMPLE n={n} t=+{elapsed}s phase={PHASE} "
            f"dle_id={dle['id'] if dle else None} "
            f"dle_seq={dle['sequence_id'] if dle else None} "
            f"dle_rec={dle['recorded_at'] if dle else None} "
            f"dle_sess={dle['tracking_session_id'] if dle else None} "
            f"canon_seq={c_seq} canon_rec={c_rec} ttl={ttl} "
            f"canon_sess={canon.get('tracking_session_id')} "
            f"canon_eid={canon.get('location_event_id')} "
            f"rest_status={st} rest_age={age} rest_src={src}"
        )

        if elapsed >= WINDOW:
            break
        time.sleep(POLL)

    dle_delta = (
        (last_dle["id"] - first_dle["id"])
        if first_dle and last_dle
        else -1
    )
    seq_delta = (
        (int(last_dle["sequence_id"]) - int(first_dle["sequence_id"]))
        if first_dle and last_dle
        else -1
    )
    canon_delta = (
        (last_canon_seq - first_canon_seq)
        if first_canon_seq is not None and last_canon_seq is not None
        else -1
    )

    print("SUMMARY")
    print(f"  dle_delta_id={dle_delta} dle_delta_seq={seq_delta}")
    print(f"  canon_delta_seq={canon_delta}")
    print(f"  rest_statuses={rest_statuses}")
    print(f"  first_dle={first_dle}")
    print(f"  last_dle={last_dle}")

    # Discriminants A/B/C/D (côté serveur ; PUT compté hors-process)
    stale_end = rest_statuses and rest_statuses[-1] in {
        "last_known",
        "offline",
        "stale",
    }
    live_start = rest_statuses and rest_statuses[0] in {"live", "recent"}

    if dle_delta <= 1 and canon_delta <= 0:
        print("HYPOTHESIS A_or_D low_pg_and_canonical")
    elif dle_delta > 1 and canon_delta <= 0:
        print("HYPOTHESIS B pg_advances_canonical_stale")
    elif dle_delta > 1 and canon_delta > 0 and stale_end:
        print("HYPOTHESIS C pg_canon_ok_rest_stale")
    elif dle_delta > 1 and canon_delta > 0 and not stale_end:
        print("HYPOTHESIS OK chain_fresh")
    else:
        print("HYPOTHESIS MIXED_REVIEW")

    print("BG_FRESHNESS_RCA_END", datetime.now(UTC).isoformat())
