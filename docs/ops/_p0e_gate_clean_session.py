"""Gate pré-Phase2 : session tracking propre chauffeur 20135.

Évalue la session la plus récente (pas tout l'historique > BASE_ID).
DLQ : préférer P0E_DLQ_COUNT_OVERRIDE (hôte).
"""
from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime

from app import create_app
from sqlalchemy import text

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
BASE_ID = int(os.getenv("P0E_BASE_DLE_ID", "5903"))
OLD_SESSION = os.getenv(
    "P0E_OLD_SESSION", "trk_sess_1786965149557_7lkzgzna"
)
MIN_NEW = int(os.getenv("P0E_MIN_NEW", "3"))
DLQ_WINDOW = os.getenv("P0E_DLQ_WINDOW", "3m")


def _dlq_count() -> int:
    override = os.getenv("P0E_DLQ_COUNT_OVERRIDE")
    if override is not None and str(override).strip() != "":
        try:
            return int(override)
        except ValueError:
            return -1
    try:
        out = subprocess.check_output(
            [
                "docker",
                "logs",
                "atmr-tracking-kafka-consumer-1",
                "--since",
                DLQ_WINDOW,
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        print(f"DLQ_LOG_ERR {type(exc).__name__}")
        return -1
    return sum(1 for line in out.splitlines() if "event_id_payload_conflict" in line)


app = create_app()
with app.app_context():
    from models import db

    print("GATE_PRE_PHASE2")
    print(f"  now={datetime.now(UTC).isoformat()}")
    print(f"  driver_id={DRIVER_ID} base_id={BASE_ID}")
    print(f"  old_session={OLD_SESSION}")

    pg_first = os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "false")
    outbox = os.getenv("TRACKING_PERSIST_WITH_OUTBOX", "unset")
    print(f"  PG_FIRST={pg_first}")
    print(f"  OUTBOX_env={outbox}")

    latest = db.session.execute(
        text(
            """
            SELECT id, sequence_id, session_generation, tracking_session_id,
                   location_event_id, capture_id, created_at
            FROM driver_location_events
            WHERE driver_id=:d ORDER BY id DESC LIMIT 1
            """
        ),
        {"d": DRIVER_ID},
    ).mappings().first()
    print("LATEST", dict(latest) if latest else None)

    if not latest:
        print("VERDICT FAIL_NO_DLE")
        raise SystemExit(2)

    cur_session = latest["tracking_session_id"]
    rows = db.session.execute(
        text(
            """
            SELECT id, sequence_id, session_generation, tracking_session_id,
                   location_event_id, capture_id, created_at
            FROM driver_location_events
            WHERE driver_id=:d AND tracking_session_id=:s
            ORDER BY id ASC
            """
        ),
        {"d": DRIVER_ID, "s": cur_session},
    ).mappings().all()
    # garder les MIN_NEW+ dernières pour l'affichage / checks récents
    recent = list(rows[-max(MIN_NEW, 8) :])
    print(f"CURRENT_SESSION={cur_session}")
    print(f"SESSION_ROW_COUNT={len(rows)}")
    for r in recent:
        print(
            f"  id={r['id']} seq={r['sequence_id']} gen={r['session_generation']} "
            f"eid={r['location_event_id']} cap={r['capture_id']} at={r['created_at']}"
        )

    drv = db.session.execute(
        text(
            "SELECT id, last_position_update, latitude, longitude "
            "FROM driver WHERE id=:d"
        ),
        {"d": DRIVER_ID},
    ).mappings().first()
    print("DRIVER", dict(drv) if drv else None)

    dlq_n = _dlq_count()
    print(f"DLQ_conflict_window={DLQ_WINDOW} count={dlq_n}")

    # Fenêtre récente = dernières MIN_NEW de la session courante
    window = list(rows[-MIN_NEW:]) if len(rows) >= MIN_NEW else list(rows)

    checks: dict[str, bool] = {}
    checks["pg_first_off"] = str(pg_first).lower() in ("false", "0", "no", "")
    checks["dle_gt_base"] = int(latest["id"]) > BASE_ID
    checks["min_new"] = len(window) >= MIN_NEW
    checks["session_new"] = bool(cur_session) and cur_session != OLD_SESSION

    eids = [r["location_event_id"] for r in window]
    caps = [r["capture_id"] for r in window if r.get("capture_id")]
    checks["eid_unique"] = len(eids) == len(set(eids)) and len(eids) >= MIN_NEW
    checks["cap_unique"] = len(caps) == len(set(caps)) and len(caps) >= MIN_NEW

    seqs = [int(r["sequence_id"]) for r in window]
    checks["seq_monotone"] = (
        all(seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1))
        if len(seqs) >= 2
        else False
    )

    checks["dlq_zero"] = dlq_n == 0
    checks["driver_position_recent"] = bool(
        drv and drv.get("last_position_update")
    )

    print("CHECKS")
    for k, v in checks.items():
        print(f"  {k}={v}")

    required = [
        "pg_first_off",
        "dle_gt_base",
        "min_new",
        "session_new",
        "eid_unique",
        "cap_unique",
        "seq_monotone",
        "dlq_zero",
    ]
    ok = all(checks[k] for k in required)
    print("VERDICT", "PASS" if ok else "WAIT_OR_FAIL")
    if not ok:
        raise SystemExit(2)
    print("GATE_PRE_PHASE2_PASS")
