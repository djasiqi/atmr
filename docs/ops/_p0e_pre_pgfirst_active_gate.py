"""Gate AVANT re-GO PG_FIRST — discriminant serveur strict post pm clear.

Critères :
- PG_FIRST=false
- nouvelle tracking_session_id ≠ blacklist (sessions polluées)
- status=active
- started_at / created_at APRÈS P0E_CLEAR_AFTER (UTC)
- ≥ MIN_DLE sur cette session, mission_id attendu
- seq monotone, eid/capture uniques
- aucune DLE post-clear sur ancienne session superseded

Exit 0 = READY_FOR_PGFIRST
Exit 2 = NOT_READY
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

from app import create_app
from sqlalchemy import text

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
MISSION_ID = int(os.getenv("P0E_MISSION_ID", "38243"))
MIN_DLE = int(os.getenv("P0E_MIN_DLE", "2"))
# pm clear ~ 2026-08-17T13:17:20Z
CLEAR_AFTER = os.getenv("P0E_CLEAR_AFTER", "2026-08-17T13:17:00+00:00")
BLACKLIST = {
    s.strip()
    for s in os.getenv(
        "P0E_SESSION_BLACKLIST",
        "trk_sess_1786968778000_nlh0et7f,"
        "trk_sess_1786971820868_fr3ty46h,"
        "trk_sess_1786966963875_1tbcieoy,"
        "trk_sess_1786965149557_7lkzgzna",
    ).split(",")
    if s.strip()
}


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


app = create_app()
with app.app_context():
    from models import db

    clear_after = _parse_ts(CLEAR_AFTER)
    pg_first = os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "false")
    print("GATE_PRE_PGFIRST_STRICT")
    print(f"  now={datetime.now(UTC).isoformat()}")
    print(f"  driver_id={DRIVER_ID} mission_id={MISSION_ID}")
    print(f"  clear_after={clear_after.isoformat()}")
    print(f"  PG_FIRST={pg_first}")
    print(f"  blacklist_n={len(BLACKLIST)}")

    checks: dict[str, bool] = {}
    checks["pg_first_off"] = str(pg_first).lower() in ("false", "0", "no", "")

    active = db.session.execute(
        text(
            """
            SELECT tracking_session_id, session_generation, status,
                   started_at, created_at, updated_at
            FROM tracking_sessions
            WHERE driver_id=:d AND status='active'
            ORDER BY id DESC LIMIT 1
            """
        ),
        {"d": DRIVER_ID},
    ).mappings().first()
    print("ACTIVE", dict(active) if active else None)

    checks["has_active"] = bool(active)
    if not active:
        print("VERDICT NOT_READY no_active_session")
        for k, v in checks.items():
            print(f"  {k}={v}")
        raise SystemExit(2)

    sid = str(active["tracking_session_id"])
    started = active.get("started_at") or active.get("created_at")
    if started is not None and started.tzinfo is None:
        started = started.replace(tzinfo=UTC)

    checks["session_not_blacklisted"] = sid not in BLACKLIST
    checks["session_after_clear"] = bool(started and started >= clear_after)
    checks["session_active"] = str(active.get("status")) == "active"

    rows = db.session.execute(
        text(
            """
            SELECT id, sequence_id, location_event_id, capture_id,
                   mission_id, created_at
            FROM driver_location_events
            WHERE driver_id=:d AND tracking_session_id=:s
            ORDER BY id ASC
            """
        ),
        {"d": DRIVER_ID, "s": sid},
    ).mappings().all()
    print(f"DLE_ON_ACTIVE={len(rows)} session={sid}")
    for r in rows[-8:]:
        print(dict(r))

    checks["min_dle"] = len(rows) >= MIN_DLE
    mission_ok = all(
        int(r["mission_id"] or 0) == MISSION_ID for r in rows
    ) if rows else False
    checks["mission_match"] = mission_ok and len(rows) > 0

    if len(rows) >= 2:
        seqs = [int(r["sequence_id"]) for r in rows]
        eids = [r["location_event_id"] for r in rows]
        caps = [r["capture_id"] for r in rows if r.get("capture_id")]
        checks["seq_monotone"] = all(
            seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1)
        )
        checks["eid_unique"] = len(eids) == len(set(eids))
        checks["cap_unique"] = len(caps) == len(set(caps)) and len(caps) == len(
            rows
        )
    elif len(rows) == 1:
        checks["seq_monotone"] = True
        checks["eid_unique"] = True
        checks["cap_unique"] = bool(rows[0].get("capture_id"))
    else:
        checks["seq_monotone"] = False
        checks["eid_unique"] = False
        checks["cap_unique"] = False

    # Post-clear DLE on superseded / blacklisted sessions = FAIL
    polluted = db.session.execute(
        text(
            """
            SELECT e.id, e.tracking_session_id, e.sequence_id, e.created_at,
                   s.status
            FROM driver_location_events e
            LEFT JOIN tracking_sessions s
              ON s.tracking_session_id = e.tracking_session_id
            WHERE e.driver_id=:d
              AND e.created_at >= :ca
              AND (
                    e.tracking_session_id = ANY(:bl)
                 OR COALESCE(s.status, '') = 'superseded'
              )
            ORDER BY e.id DESC
            LIMIT 10
            """
        ),
        {"d": DRIVER_ID, "ca": clear_after, "bl": list(BLACKLIST)},
    ).mappings().all()
    print(f"POST_CLEAR_POLLUTED_DLE={len(polluted)}")
    for r in polluted[:5]:
        print(dict(r))
    checks["no_post_clear_superseded_loc"] = len(polluted) == 0

    # PG advancing recently on the active session
    n_active_recent = db.session.execute(
        text(
            """
            SELECT COUNT(*) FROM driver_location_events
            WHERE driver_id=:d AND tracking_session_id=:s
              AND created_at > NOW() - INTERVAL '5 minutes'
            """
        ),
        {"d": DRIVER_ID, "s": sid},
    ).scalar()
    checks["active_pg_recent"] = int(n_active_recent or 0) >= 1

    print("CHECKS")
    for k, v in checks.items():
        print(f"  {k}={v}")

    ok = all(checks.values())
    print("VERDICT", "READY_FOR_PGFIRST" if ok else "NOT_READY")
    if ok:
        print("PRE_PGFIRST_GATE_PASS")
        print(f"ACTIVE_SESSION={sid}")
        print(f"ACTIVE_GEN={active.get('session_generation')}")
        print(f"DLE_COUNT={len(rows)}")
    else:
        raise SystemExit(2)
