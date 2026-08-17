from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db

    cols = {
        r[0]
        for r in db.session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='driver_location_events'"
            )
        ).fetchall()
    }
    print("has_capture_id", "capture_id" in cols)
    cols2 = {
        r[0]
        for r in db.session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='tracking_ingest_events'"
            )
        ).fetchall()
    }
    print("ingest_has_capture_id", "capture_id" in cols2)
    v = db.session.execute(text("SELECT version_num FROM alembic_version")).scalar()
    print("alembic", v)
    if "capture_id" not in cols or "capture_id" not in cols2 or v != "25ce766952e2":
        raise SystemExit("A_VERIFY_FAIL")
    print("A_VERIFY_PASS")
