from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
r=db.session.execute(text("""
SELECT id, status, driver_id, updated_at
FROM booking
WHERE driver_id=20135 AND status IN ('IN_PROGRESS','EN_ROUTE','ARRIVED','ASSIGNED')
ORDER BY updated_at DESC NULLS LAST
LIMIT 5
""")).fetchall()
print("MISSIONS", len(r))
for row in r:
    print(f"B{row[0]} {row[1]} d={row[2]} u={row[3]}")