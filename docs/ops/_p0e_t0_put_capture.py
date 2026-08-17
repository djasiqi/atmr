"""P0-E — capture T0 PUT live (schema-aware, lecture seule).

Usage (dès que driver 20135 tracke) :
  docker exec atmr-backend-1 python /tmp/_p0e_t0_put_capture.py

Tranche A/B/C/D sur last_raw + loc:canonical + PG.
Ne pas lancer de SELECT sur colonnes hypothétiques.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

from app import create_app

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))


def _decode_hash(raw: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


def _print_hash(label: str, key: str, client) -> dict[str, str]:
    raw = _decode_hash(client.hgetall(key) or {})
    ttl = client.ttl(key)
    print(f"{label}")
    print(f"  key={key}")
    print(f"  exists={bool(raw)}")
    print(f"  ttl_s={ttl}")
    if not raw:
        return raw
    for k in (
        "accept_status",
        "accept_reason",
        "recorded_at",
        "received_at",
        "location_mode",
        "mission_id",
        "location_event_id",
        "capture_id",
        "tracking_session_id",
        "session_generation",
        "sequence_id",
        "lat",
        "lon",
        "source",
    ):
        if k in raw:
            print(f"  {k}={raw[k]}")
    return raw


app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from ext import redis_client

    now = datetime.now(UTC)
    print("T0_CAPTURE")
    print(f"  now_utc={now.isoformat()}")
    print(f"  driver_id={DRIVER_ID}")

    last_raw = _print_hash(
        "REDIS_LAST_RAW", f"driver:{DRIVER_ID}:loc:last_raw", redis_client
    )
    canon = _print_hash(
        "REDIS_CANONICAL", f"driver:{DRIVER_ID}:loc:canonical", redis_client
    )
    _print_hash("REDIS_LEGACY", f"driver:{DRIVER_ID}:loc", redis_client)

    latest = db.session.execute(
        text(
            "SELECT id, sequence_id, recorded_at, created_at, location_mode, "
            "mission_id, location_event_id, tracking_session_id "
            "FROM driver_location_events WHERE driver_id=:d "
            "ORDER BY id DESC LIMIT 1"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("PG_LATEST_LOC")
    if latest:
        for k, v in dict(latest).items():
            print(f"  {k}={v}")
        age = int((now - latest["created_at"]).total_seconds())
        print(f"  age_created_s={age}")
        print(f"  fresh_lt_60s={age < 60}")
    else:
        print("  NONE")

    dcols = {
        r[0]
        for r in db.session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='driver'"
            )
        ).fetchall()
    }
    sel = [
        c
        for c in ("id", "last_position_update", "latitude", "longitude", "is_active")
        if c in dcols
    ]
    drv = db.session.execute(
        text(f"SELECT {', '.join(sel)} FROM driver WHERE id=:d"),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("DRIVER")
    for k, v in dict(drv or {}).items():
        print(f"  {k}={v}")
    if drv and drv.get("last_position_update"):
        lpu = drv["last_position_update"]
        if lpu.tzinfo is None:
            lpu = lpu.replace(tzinfo=UTC)
        print(f"  age_lpu_s={int((now - lpu).total_seconds())}")

    status = (last_raw.get("accept_status") or "").strip().lower()
    reason = (last_raw.get("accept_reason") or "").strip()
    print("VERDICT")
    if not last_raw and not canon:
        print("  case=D")
        print("  meaning=PUT path did not write Redis (or pre-T0 / wrong redis)")
    elif last_raw and not canon:
        if status in {
            "accepted_observability_only",
            "ignored",
            "stale",
            "rejected",
            "rejected_invalid",
        } or status.endswith("_only"):
            print("  case=A")
            print(
                "  meaning=ingest policy blocked canonical promotion "
                f"(accept_status={status!r} reason={reason!r})"
            )
        elif status in {"accepted_canonical", "accepted"}:
            print("  case=B")
            print(
                "  meaning=accepted but canonical missing immediately "
                "(writer / promote_location_candidate)"
            )
        else:
            print("  case=A_OR_B_AMBIGUOUS")
            print(f"  accept_status={status!r} accept_reason={reason!r}")
    elif canon:
        print("  case=C_OR_OK")
        print("  meaning=canonical present now — re-probe in 30s/TTL to detect Cas C wipe")
        print(f"  canonical_ttl_s={redis_client.ttl(f'driver:{DRIVER_ID}:loc:canonical')}")
        if last_raw:
            print(f"  last_raw.accept_status={status!r}")
    else:
        print("  case=UNEXPECTED")
