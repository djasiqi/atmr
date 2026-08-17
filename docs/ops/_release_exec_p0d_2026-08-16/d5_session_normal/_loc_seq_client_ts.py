"""Lecture seule : recorded_at client pour LOC seq 29/30 autour de T_FAIL."""
from sqlalchemy import text
from app import create_app

app = create_app()
app.app_context().push()
from models import db

rows = list(
    db.session.execute(
        text(
            """
 SELECT created_at, recorded_at, sequence_id, mission_id, tracking_session_id
 FROM driver_location_events
 WHERE driver_id = 20135
   AND sequence_id IN (29, 30)
   AND created_at >= '2026-08-16 19:18:00+00'
   AND created_at <= '2026-08-16 19:19:00+00'
 ORDER BY sequence_id
"""
        )
    ).fetchall()
)
print("N", len(rows))
for r in rows:
    created, recorded, seq, mission, sess = r
    delta_ms = None
    if created is not None and recorded is not None:
        delta_ms = (created - recorded).total_seconds() * 1000.0
    print(
        f"seq={seq} created={created} recorded={recorded} "
        f"delta_ms={delta_ms} mission={mission} sess={sess}"
    )
