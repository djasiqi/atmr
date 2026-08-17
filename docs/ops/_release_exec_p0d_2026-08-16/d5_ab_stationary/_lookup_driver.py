from app import create_app
app = create_app()
app.app_context().push()
from models import db
from sqlalchemy import text

r = db.session.execute(
    text(
        """
        SELECT d.id, u.email, u.phone, u.username, u.first_name, u.last_name
        FROM drivers d
        JOIN users u ON u.id = d.user_id
        WHERE d.id = 20135
        """
    )
).fetchone()
print("ID", r[0] if r else None)
print("EMAIL", r[1] if r else None)
print("PHONE", r[2] if r else None)
print("USERNAME", r[3] if r else None)
print("NAME", (r[4] or "") if r else None, (r[5] or "") if r else None)
