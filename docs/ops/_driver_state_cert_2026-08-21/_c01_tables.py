from app import create_app
from models import db
from sqlalchemy import text

app = create_app()
with app.app_context():
    tables = db.session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND ("
            "table_name ILIKE '%location%' OR table_name ILIKE '%track%' "
            "OR table_name ILIKE '%ingest%' OR table_name ILIKE '%gps%')"
            " ORDER BY 1"
        )
    ).fetchall()
    print("TABLES", [t[0] for t in tables])
