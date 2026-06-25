"""Diagnostic rapide persistance GPS prod (exécuter via docker compose exec backend)."""
from app import create_app
from ext import db

app = create_app()
with app.app_context():
    print("=== trip_tracking ===")
    for label, sql in [
        ("last_24h", "SELECT COUNT(*) FROM trip_tracking WHERE timestamp > NOW() - INTERVAL '24 hours'"),
        ("driver3_since_21jun", "SELECT COUNT(*) FROM trip_tracking WHERE driver_id=3 AND timestamp > '2026-06-21 06:40:00'"),
        ("driver3_max_ts", "SELECT MAX(timestamp) FROM trip_tracking WHERE driver_id=3"),
    ]:
        print(f"{label}={db.session.execute(db.text(sql)).scalar()}")

    row = db.session.execute(
        db.text(
            """
            SELECT COUNT(*) AS active,
                   COUNT(*) FILTER (WHERE last_position_update > NOW() - INTERVAL '24 hours') AS fresh_24h,
                   COUNT(*) FILTER (WHERE last_position_update > NOW() - INTERVAL '10 minutes') AS fresh_10m
            FROM driver WHERE is_active = true
            """
        )
    ).one()
    print(f"drivers active={row.active} fresh_24h={row.fresh_24h} fresh_10m={row.fresh_10m}")

    j = db.session.execute(
        db.text("SELECT id, last_position_update, latitude, longitude FROM driver WHERE id=3")
    ).one_or_none()
    if j:
        print(f"jozsef id=3 last_position_update={j.last_position_update} lat={j.latitude} lon={j.longitude}")
