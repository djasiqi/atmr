from app import create_app
from datetime import datetime, timezone, timedelta

app = create_app()
with app.app_context():
    from models import db
    from sqlalchemy import text

    print("LABEL", "PRECHECK")
    print("NOW", datetime.now().astimezone().isoformat())
    since = datetime.now(timezone.utc) - timedelta(minutes=20)

    tables = db.session.execute(
        text(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='public' AND table_name ILIKE '%mission%'
            ORDER BY 1
            """
        )
    ).fetchall()
    print("MISSION_TABLES", [t[0] for t in tables])

    health = list(
        db.session.execute(
            text(
                """
                SELECT recorded_at, app_state, tracking_active, fgs_running, native_task_running,
                       last_fix_age_seconds, native_last_fix_age_seconds, constraint_reason,
                       native_start_error, trigger_reason, release_sha
                FROM driver_device_health_events
                WHERE driver_id = 19 AND recorded_at >= :since
                ORDER BY recorded_at DESC LIMIT 20
                """
            ),
            {"since": since},
        ).mappings()
    )
    print("HEALTH_N", len(health))
    fgs_t = sum(1 for h in health if h.get("fgs_running") is True)
    nat_t = sum(1 for h in health if h.get("native_task_running") is True)
    print("FGS_TRUE_RATIO", f"{fgs_t}/{len(health)}")
    print("NATIVE_TRUE_RATIO", f"{nat_t}/{len(health)}")
    print("NATIVE_ERR_N", sum(1 for h in health if h.get("native_start_error")))
    for h in health[:8]:
        print(
            "H",
            h["recorded_at"],
            "app=",
            h["app_state"],
            "fgs=",
            h["fgs_running"],
            "nat=",
            h["native_task_running"],
            "fix=",
            h["last_fix_age_seconds"],
            "nfix=",
            h["native_last_fix_age_seconds"],
            "cr=",
            h["constraint_reason"],
            "sha=",
            (h["release_sha"] or "")[:12],
            "err=",
            (h["native_start_error"] or "")[:100],
        )

    locs = list(
        db.session.execute(
            text(
                """
                SELECT created_at, mission_id, capture_id, driver_id
                FROM driver_location_events
                WHERE driver_id = 19 AND created_at >= :since
                ORDER BY created_at DESC LIMIT 12
                """
            ),
            {"since": since},
        ).fetchall()
    )
    print("LOC_N", len(locs))
    for r in locs[:8]:
        print("LOC", r[0], "mission=", r[1], "cap=", r[2], "driver=", r[3])
