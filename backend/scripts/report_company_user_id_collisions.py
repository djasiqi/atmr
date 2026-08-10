#!/usr/bin/env python3
"""Précontrôle ops : collisions company.user_id avant UNIQUE (PR1 Partenaires).

Exécuter via Docker, ex. :
  docker compose exec atmr_api python scripts/report_company_user_id_collisions.py
"""

from __future__ import annotations

import sys

from sqlalchemy import text

from app import create_app
from ext import db


def main() -> int:
    app = create_app()
    with app.app_context():
        rows = (
            db.session.execute(
                text(
                    """
                SELECT
                    c.user_id,
                    COUNT(*) AS company_count,
                    ARRAY_AGG(c.id ORDER BY c.id) AS company_ids,
                    ARRAY_AGG(c.name ORDER BY c.id) AS company_names,
                    (
                        SELECT COUNT(*) FROM driver d
                        WHERE d.company_id = ANY(ARRAY_AGG(c.id))
                    ) AS drivers_count,
                    (
                        SELECT COUNT(*) FROM booking b
                        WHERE b.company_id = ANY(ARRAY_AGG(c.id))
                    ) AS bookings_count
                FROM company c
                GROUP BY c.user_id
                HAVING COUNT(*) > 1
                ORDER BY c.user_id
                """
                )
            )
            .mappings()
            .all()
        )

        if not rows:
            print("OK: aucune collision company.user_id")
            return 0

        print(f"COLLISIONS: {len(rows)}")
        for row in rows:
            print(
                f"  user_id={row['user_id']} companies={row['company_count']} "
                f"ids={list(row['company_ids'])} names={list(row['company_names'])} "
                f"drivers≈{row['drivers_count']} bookings≈{row['bookings_count']}"
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
