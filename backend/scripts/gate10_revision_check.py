#!/usr/bin/env python3
"""GATE 10 — vérifications SQL revision (exécuter dans le container atmr_api)."""

from __future__ import annotations

from app import create_app
from ext import db


def main() -> int:
    app = create_app()
    with app.app_context():
        null_count = db.session.execute(
            db.text("SELECT COUNT(*) FROM transport_requests WHERE revision IS NULL")
        ).scalar()
        row = db.session.execute(
            db.text(
                "SELECT MIN(revision), MAX(revision), COUNT(*) FROM transport_requests"
            )
        ).fetchone()
        min_rev, max_rev, total = row if row else (None, None, 0)
        print(f"revision_null_count={null_count}")
        print(f"min_revision={min_rev}")
        print(f"max_revision={max_rev}")
        print(f"total_rows={total}")
        ok = null_count == 0 and (min_rev is None or min_rev >= 1)
        print(f"GATE10_SQL={'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
