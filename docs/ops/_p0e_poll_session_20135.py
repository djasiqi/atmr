"""Poll tracking session for driver 20135 (run inside backend container)."""
from __future__ import annotations

from sqlalchemy import text

from app import create_app
from models import db


def main() -> None:
    app = create_app()
    with app.app_context():
        active = db.session.execute(
            text(
                """
                SELECT tracking_session_id, session_generation, status, started_at
                FROM tracking_sessions
                WHERE driver_id = 20135 AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """
            )
        ).mappings().first()
        print("ACTIVE", dict(active) if active else None)
        if not active:
            return
        sid = active["tracking_session_id"]
        row = db.session.execute(
            text(
                """
                SELECT COUNT(1) AS n, COALESCE(MAX(sequence_id), 0) AS max_seq,
                       MIN(created_at) AS first_at, MAX(created_at) AS last_at
                FROM driver_location_events
                WHERE tracking_session_id = :s
                """
            ),
            {"s": sid},
        ).mappings().first()
        print(
            "DLE",
            f"n={row['n']}",
            f"max_seq={row['max_seq']}",
            f"first={row['first_at']}",
            f"last={row['last_at']}",
        )
        sample = db.session.execute(
            text(
                """
                SELECT event_id, sequence_id, created_at
                FROM driver_location_events
                WHERE tracking_session_id = :s
                ORDER BY sequence_id ASC
                LIMIT 3
                """
            ),
            {"s": sid},
        ).mappings().all()
        for s in sample:
            print("SAMPLE", dict(s))


if __name__ == "__main__":
    main()
