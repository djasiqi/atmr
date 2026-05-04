#!/usr/bin/env python3
"""
Recale des réservations ACCEPTÉES d’un jour vers la veille et passe le statut à COMPLETED.

Cas d’usage : courses saisies / affichées au mauvais jour (ex. 01.05.2026 → 30.04.2026)
et à marquer comme terminées pour la facturation ou l’historique.

⚠️ Contourne validate_status_transition (ACCEPTED → COMPLETED n’est pas autorisé en API).

Usage (depuis backend/, avec venv activé) :

    python scripts/shift_accepted_bookings_day_and_complete.py \\
        --company-id ID \\
        --from-date 2026-05-01 \\
        --dry-run

    python scripts/shift_accepted_bookings_day_and_complete.py \\
        --company-id ID \\
        --from-date 2026-05-01 \\
        --apply

Par défaut : dry-run (aucune écriture). --apply pour commit.

Optionnel : --shift-days -1 (défaut : -1, veille de from-date).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app  # noqa: E402
from ext import db  # noqa: E402
from models.booking import Booking  # noqa: E402
from models.enums import BookingStatus  # noqa: E402
from sqlalchemy import Date, cast  # noqa: E402


def _parse_date(s: str) -> date:
    y, m, d = (int(x) for x in s.strip().split("-", 2))
    return date(y, m, d)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Décale scheduled_time (ACCEPTED) et passe en COMPLETED."
    )
    p.add_argument(
        "--company-id",
        type=int,
        required=True,
        help="Entreprise concernée (obligatoire pour éviter les mises à jour globales).",
    )
    p.add_argument(
        "--from-date",
        type=_parse_date,
        required=True,
        help="Journée source des scheduled_time (naifs), ex. 2026-05-01.",
    )
    p.add_argument(
        "--shift-days",
        type=int,
        default=-1,
        help="Nombre de jours à ajouter à scheduled_time (défaut: -1).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Écrit en base et commit (sans ce flag : simulation uniquement).",
    )
    args = p.parse_args()
    dry_run = not args.apply

    app = create_app()
    with app.app_context():
        day = args.from_date
        q = (
            db.session.query(Booking)
            .filter(Booking.company_id == args.company_id)
            .filter(Booking.status == BookingStatus.ACCEPTED)
            .filter(cast(Booking.scheduled_time, Date) == day)
            .order_by(Booking.scheduled_time.asc(), Booking.id.asc())
        )
        rows = q.all()
        if not rows:
            print(
                f"Aucune réservation ACCEPTED pour company_id={args.company_id} "
                f"avec scheduled_time le {day}."
            )
            return 0

        print(
            f"{'[DRY-RUN] ' if dry_run else ''}"
            f"{len(rows)} réservation(s) à traiter "
            f"(company_id={args.company_id}, jour={day}, shift_days={args.shift_days})."
        )
        for b in rows:
            old = b.scheduled_time
            new_st = (
                (old + timedelta(days=args.shift_days)) if old is not None else None
            )
            print(
                f"  id={b.id} customer={b.customer_name!r} "
                f"scheduled {old} -> {new_st} | driver_id={b.driver_id}"
            )

        if dry_run:
            print("Aucune modification (relancer avec --apply pour enregistrer).")
            return 0

        now_utc = datetime.now(UTC)
        for b in rows:
            if b.scheduled_time is not None:
                b.scheduled_time = b.scheduled_time + timedelta(days=args.shift_days)
            b.status = BookingStatus.COMPLETED
            if b.completed_at is None:
                b.completed_at = now_utc

        db.session.commit()
        print(f"OK — {len(rows)} réservation(s) mises à jour et commit.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
