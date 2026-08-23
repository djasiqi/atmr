from sqlalchemy import text
from app import create_app
from models import db

app = create_app()
with app.app_context():
    b = db.session.execute(
        text(
            "SELECT id, status::text, driver_id, company_id, executing_company_id "
            "FROM booking WHERE id=51"
        )
    ).mappings().first()
    d = db.session.execute(
        text("SELECT id, company_id FROM driver WHERE id=20")
    ).mappings().first()
    print("BOOKING", dict(b) if b else None)
    print("DRIVER", dict(d) if d else None)
    if b and d:
        exec_c = b["executing_company_id"]
        if exec_c is not None:
            ok = int(d["company_id"]) == int(exec_c)
            print("CHECK_EXECUTING", ok, "driver_co", d["company_id"], "exec", exec_c)
        else:
            ok = int(d["company_id"]) == int(b["company_id"])
            print("CHECK_OWNER", ok, "driver_co", d["company_id"], "book", b["company_id"])
        print("DRIVER_MATCH", b["driver_id"] == d["id"])
