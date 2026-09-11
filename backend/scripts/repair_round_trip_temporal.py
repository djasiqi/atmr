"""Audit / réparation des A/R dont le retour confirmé précède l'aller.

Usage (Docker) :
  python scripts/repair_round_trip_temporal.py
  python scripts/repair_round_trip_temporal.py --repair
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("DISABLE_EVENTLET", "1")
os.environ.setdefault("SKIP_SOCKETIO", "1")

from app import create_app
from application.bookings.round_trip_temporal import (
    audit_impossible_round_trips,
)
from ext import db


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit temporel A/R")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Invalide time_confirmed des retours impossibles (sans inventer d'heure).",
    )
    args = parser.parse_args()

    cfg = os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG") or "development"
    app = create_app(cfg)
    with app.app_context():
        report = audit_impossible_round_trips(repair=args.repair)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        if args.repair:
            db.session.commit()
            print(f"Réparés : {len(report['repaired_ids'])} retour(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
