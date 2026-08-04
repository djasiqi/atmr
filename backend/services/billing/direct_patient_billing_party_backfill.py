"""Backfill : rattacher un BillingParty aux courses patient sans destinataire V2.

Cible :
  Booking.billed_to_type = 'patient'
  Booking.billing_party_id IS NULL
  Booking.invoice_line_id IS NULL

Par couple (company_id, client_id) :
  1. tiers actif via ClientBillingParty si présent
  2. sinon BillingParty PATIENT technique (patient_client:{id})
  3. rattacher toutes les courses ouvertes du couple

Idempotent : ré-exécuter ne crée pas de doublons external_ref.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ext import db
from models import Booking, Client
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_direct_patient,
)
from services.billing.client_stay_resolver import (
    resolve_default_billing_party_for_client,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectPatientBackfillResult:
    clients_touched: int
    bookings_updated: int
    billing_parties_created_or_reused: int
    dry_run: bool


def run_backfill_direct_patient_billing_party(
    *,
    dry_run: bool = True,
    limit: int | None = None,
    company_id: int | None = None,
) -> DirectPatientBackfillResult:
    """Répare les bookings patient sans ``billing_party_id``.

    ``dry_run=True`` (défaut) : calcule et flush en session puis rollback.
    """
    q = Booking.query.filter(
        Booking.billed_to_type == "patient",
        Booking.billing_party_id.is_(None),
        Booking.invoice_line_id.is_(None),
        Booking.client_id.isnot(None),
    ).order_by(Booking.company_id.asc(), Booking.client_id.asc(), Booking.id.asc())
    if company_id is not None:
        q = q.filter(Booking.company_id == int(company_id))
    if limit is not None:
        q = q.limit(int(limit))

    bookings = q.all()
    if not bookings:
        logger.info("Aucune course patient sans billing_party_id à réparer.")
        return DirectPatientBackfillResult(
            clients_touched=0,
            bookings_updated=0,
            billing_parties_created_or_reused=0,
            dry_run=dry_run,
        )

    groups: dict[tuple[int, int], list[Booking]] = defaultdict(list)
    for booking in bookings:
        groups[(int(booking.company_id), int(booking.client_id))].append(booking)

    clients_touched = 0
    bookings_updated = 0
    bp_resolved = 0

    for (cid, client_id), group in groups.items():
        client = Client.query.filter_by(id=client_id, company_id=cid).first()
        if client is None:
            logger.warning(
                "Client introuvable pour backfill company_id=%s client_id=%s "
                "(%s courses ignorées)",
                cid,
                client_id,
                len(group),
            )
            continue

        third_party = resolve_default_billing_party_for_client(
            client_id=client_id, company_id=cid
        )
        if third_party is not None:
            bp = third_party
        else:
            bp = get_or_create_billing_party_for_direct_patient(
                company_id=cid, client=client
            )
        bp_resolved += 1
        clients_touched += 1

        for booking in group:
            booking.billing_party_id = int(bp.id)
            bookings_updated += 1

        logger.info(
            "company_id=%s client_id=%s billing_party_id=%s bookings=%s source=%s",
            cid,
            client_id,
            bp.id,
            len(group),
            "third_party" if third_party is not None else "patient_direct",
        )

    if dry_run:
        db.session.rollback()
        logger.info(
            "[dry-run] rollback. clients=%s bookings=%s bp_resolved=%s",
            clients_touched,
            bookings_updated,
            bp_resolved,
        )
    else:
        db.session.commit()
        logger.info(
            "commit. clients=%s bookings=%s bp_resolved=%s",
            clients_touched,
            bookings_updated,
            bp_resolved,
        )

    return DirectPatientBackfillResult(
        clients_touched=clients_touched,
        bookings_updated=bookings_updated,
        billing_parties_created_or_reused=bp_resolved,
        dry_run=dry_run,
    )


def backfill_summary_dict(result: DirectPatientBackfillResult) -> dict[str, Any]:
    return {
        "clients_touched": result.clients_touched,
        "bookings_updated": result.bookings_updated,
        "billing_parties_created_or_reused": result.billing_parties_created_or_reused,
        "dry_run": result.dry_run,
    }
