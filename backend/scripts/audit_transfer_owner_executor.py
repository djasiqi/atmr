"""Audit transferts owner/executor : company_id vs owner_company_id et executing_company_id.

Usage (contexte où l’app a la DB configurée, ex. .env dans backend/ ou conteneur API) :
  cd backend && python scripts/audit_transfer_owner_executor.py
  # ou dans le conteneur :
  docker exec <api_container> python scripts/audit_transfer_owner_executor.py

Vérifie :
  1) Bookings où company_id ≠ owner_company_id du transfert accepté/complété (données incohérentes)
  2) Transferts ACCEPTED/COMPLETED dont la résa n’a pas executing_company_id

Sortie : rapport texte, code de sortie 0 si tout est cohérent, 1 si anomalies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app
from ext import db
from models import Booking
from models.booking_transfer import BookingTransfer
from models.enums import TransferStatus


def run_audit() -> int:
    app = create_app()
    with app.app_context():
        statuses = [TransferStatus.ACCEPTED, TransferStatus.COMPLETED]

        # 1) Résas où company_id ≠ owner_company_id du transfert accepté/complété
        rows_wrong_owner = (
            db.session.query(Booking.id, Booking.company_id, BookingTransfer.id, BookingTransfer.owner_company_id)
            .join(BookingTransfer, Booking.id == BookingTransfer.booking_id)
            .filter(
                BookingTransfer.status.in_(statuses),
                Booking.company_id != BookingTransfer.owner_company_id,
            )
            .all()
        )

        # 2) Transferts acceptés/complétés sans executing_company_id sur la résa
        rows_missing_executor = (
            db.session.query(BookingTransfer.id, BookingTransfer.booking_id)
            .join(Booking, BookingTransfer.booking_id == Booking.id)
            .filter(
                BookingTransfer.status.in_(statuses),
                Booking.executing_company_id.is_(None),
            )
            .all()
        )

        print("=== Audit transferts owner/executor ===\n")

        print("1) Bookings où company_id ≠ owner_company_id du transfert (ACCEPTED/COMPLETED)")
        if not rows_wrong_owner:
            print("   Aucune anomalie.\n")
        else:
            for booking_id, company_id, transfer_id, owner_company_id in rows_wrong_owner:
                print(
                    f"   booking_id=%s company_id=%s transfer_id=%s owner_company_id=%s"
                    % (booking_id, company_id, transfer_id, owner_company_id)
                )
            print(f"   Total: {len(rows_wrong_owner)} anomalie(s).\n")

        print("2) Transferts ACCEPTED/COMPLETED sans executing_company_id sur la résa")
        if not rows_missing_executor:
            print("   Aucune anomalie.\n")
        else:
            for transfer_id, booking_id in rows_missing_executor:
                print(f"   transfer_id=%s booking_id=%s" % (transfer_id, booking_id))
            print(f"   Total: {len(rows_missing_executor)} anomalie(s).\n")

        has_issues = len(rows_wrong_owner) > 0 or len(rows_missing_executor) > 0
        if has_issues:
            print("Résumé: anomalies détectées (check post-déploiement).")
            return 1
        print("Résumé: cohérent.")
        return 0


def main() -> None:
    exit_code = run_audit()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
