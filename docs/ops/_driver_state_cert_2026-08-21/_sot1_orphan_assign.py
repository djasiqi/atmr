from sqlalchemy import text
from app import create_app
from models import db
app=create_app()
with app.app_context():
  # Active bookings with driver, no assignment
  rows=db.session.execute(text("""
    SELECT b.id, b.status::text AS status, b.driver_id, b.company_id, b.created_via::text AS created_via,
           b.customer_name, b.created_at
    FROM booking b
    LEFT JOIN assignment a ON a.booking_id = b.id
    WHERE b.driver_id IS NOT NULL
      AND b.status::text IN ('ASSIGNED','ACCEPTED','EN_ROUTE','IN_PROGRESS')
      AND a.id IS NULL
    ORDER BY b.id DESC
    LIMIT 30
  """)).mappings().all()
  print("ORPHAN_ACTIVE_N", len(rows))
  for r in rows:
    print("ORPHAN", dict(r))

  totals=db.session.execute(text("""
    SELECT
      COUNT(*) FILTER (WHERE a.id IS NOT NULL) AS with_assign,
      COUNT(*) FILTER (WHERE a.id IS NULL) AS without_assign,
      COUNT(*) AS total
    FROM booking b
    LEFT JOIN assignment a ON a.booking_id = b.id
    WHERE b.driver_id IS NOT NULL
      AND b.status::text IN ('ASSIGNED','ACCEPTED','EN_ROUTE','IN_PROGRESS')
  """)).mappings().first()
  print("ACTIVE_DRIVER_BOOKINGS", dict(totals))

  # Sample with assignment
  sample=db.session.execute(text("""
    SELECT b.id, b.status::text, a.status::text AS a_status
    FROM booking b
    JOIN assignment a ON a.booking_id = b.id
    WHERE b.driver_id IS NOT NULL
      AND b.status::text IN ('ASSIGNED','EN_ROUTE','IN_PROGRESS')
    ORDER BY b.id DESC LIMIT 5
  """)).mappings().all()
  print("WITH_ASSIGN_SAMPLE", [dict(x) for x in sample])
