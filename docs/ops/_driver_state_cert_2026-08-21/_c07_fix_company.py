from sqlalchemy import text
from app import create_app
from models import db

BID = 51
DRIVER_ID = 20
app = create_app()
with app.app_context():
    d = db.session.execute(
        text("SELECT id, company_id FROM driver WHERE id=:did"), {"did": DRIVER_ID}
    ).mappings().first()
    co = int(d["company_id"])
    before = db.session.execute(
        text(
            "SELECT id, company_id, executing_company_id, driver_id, status::text "
            "FROM booking WHERE id=:id"
        ),
        {"id": BID},
    ).mappings().first()
    print("BEFORE", dict(before) if before else None)
    db.session.execute(
        text(
            "UPDATE booking SET company_id=:co, executing_company_id=:co "
            "WHERE id=:id"
        ),
        {"co": co, "id": BID},
    )
    db.session.commit()
    after = db.session.execute(
        text(
            "SELECT id, company_id, executing_company_id, driver_id, status::text "
            "FROM booking WHERE id=:id"
        ),
        {"id": BID},
    ).mappings().first()
    print("AFTER", dict(after) if after else None)
