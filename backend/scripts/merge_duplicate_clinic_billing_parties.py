"""Fusionne les BillingParty de type clinic dupliqués.

Objectif:
- détecter les doublons par (company_id + nom normalisé)
- choisir un enregistrement canonique
- réaffecter les références (mappings, liens clients, bookings, invoices, vouchers)
- supprimer les doublons

Usage:
  python scripts/merge_duplicate_clinic_billing_parties.py --dry-run
  python scripts/merge_duplicate_clinic_billing_parties.py --company-id 12
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from ext import db
from models import (
    BillingParty,
    BillingPartyType,
    Booking,
    ClientBillingParty,
    ClinicBillingPartyMapping,
    Invoice,
    TransportVoucher,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = (
        normalized.lower()
        .replace("’", "'")
        .replace("`", "'")
        .replace("´", "'")
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _ref_counts(bp_id: int) -> dict[str, int]:
    return {
        "mappings": ClinicBillingPartyMapping.query.filter_by(billing_party_id=bp_id).count(),
        "client_links": ClientBillingParty.query.filter_by(billing_party_id=bp_id).count(),
        "bookings": Booking.query.filter_by(billing_party_id=bp_id).count(),
        "invoices": Invoice.query.filter_by(billing_party_id=bp_id).count(),
        "vouchers": TransportVoucher.query.filter_by(billing_party_id=bp_id).count(),
    }


def _pick_canonical(parties: list[BillingParty]) -> BillingParty:
    def score(bp: BillingParty):
        refs = _ref_counts(bp.id)
        total_refs = sum(refs.values())
        has_clinic_ref = int(bool(bp.external_ref and str(bp.external_ref).startswith("clinic_company:")))
        updated = bp.updated_at or datetime.min
        # Priorité: +références, +external_ref clinic, actif, plus récent, plus grand id
        return (total_refs, has_clinic_ref, int(bool(bp.is_active)), updated, bp.id)

    return max(parties, key=score)


def _merge_fields(canonical: BillingParty, duplicate: BillingParty) -> None:
    if (not canonical.billing_address) and duplicate.billing_address:
        canonical.billing_address = duplicate.billing_address
    if (not canonical.contact_email) and duplicate.contact_email:
        canonical.contact_email = duplicate.contact_email
    if (not canonical.contact_phone) and duplicate.contact_phone:
        canonical.contact_phone = duplicate.contact_phone
    if (not canonical.external_ref) and duplicate.external_ref:
        # Ne poser external_ref que s'il n'existe pas déjà sur la même company
        exists = BillingParty.query.filter(
            BillingParty.company_id == canonical.company_id,
            BillingParty.external_ref == duplicate.external_ref,
            BillingParty.id != canonical.id,
        ).first()
        if not exists:
            canonical.external_ref = duplicate.external_ref


def _reassign_duplicate(duplicate: BillingParty, canonical: BillingParty) -> dict[str, int]:
    moved = {
        "mappings": 0,
        "client_links_updated": 0,
        "client_links_deleted": 0,
        "bookings": 0,
        "invoices": 0,
        "vouchers": 0,
    }

    # 1) Clinic mappings
    mappings = ClinicBillingPartyMapping.query.filter_by(billing_party_id=duplicate.id).all()
    for m in mappings:
        m.billing_party_id = canonical.id
        moved["mappings"] += 1

    # 2) Client links (gérer unicité (client_id, billing_party_id))
    links = ClientBillingParty.query.filter_by(billing_party_id=duplicate.id).all()
    for link in links:
        existing = ClientBillingParty.query.filter_by(
            client_id=link.client_id,
            billing_party_id=canonical.id,
        ).first()
        if existing:
            db.session.delete(link)
            moved["client_links_deleted"] += 1
        else:
            link.billing_party_id = canonical.id
            moved["client_links_updated"] += 1

    # 3) Booking / Invoice / Voucher
    moved["bookings"] = Booking.query.filter_by(billing_party_id=duplicate.id).update(
        {"billing_party_id": canonical.id},
        synchronize_session=False,
    )
    moved["invoices"] = Invoice.query.filter_by(billing_party_id=duplicate.id).update(
        {"billing_party_id": canonical.id},
        synchronize_session=False,
    )
    moved["vouchers"] = TransportVoucher.query.filter_by(billing_party_id=duplicate.id).update(
        {"billing_party_id": canonical.id},
        synchronize_session=False,
    )

    _merge_fields(canonical, duplicate)
    db.session.delete(duplicate)
    return moved


def run(*, dry_run: bool, company_id: int | None = None) -> dict[str, int]:
    q = BillingParty.query.filter(BillingParty.type == BillingPartyType.CLINIC)
    if company_id is not None:
        q = q.filter(BillingParty.company_id == company_id)

    all_clinics = q.order_by(BillingParty.company_id.asc(), BillingParty.id.asc()).all()
    grouped: dict[tuple[int, str], list[BillingParty]] = defaultdict(list)
    for bp in all_clinics:
        key = (bp.company_id, _normalize_name(bp.display_name))
        grouped[key].append(bp)

    stats = {
        "groups_with_duplicates": 0,
        "billing_parties_deleted": 0,
        "mappings_reassigned": 0,
        "client_links_updated": 0,
        "client_links_deleted": 0,
        "bookings_reassigned": 0,
        "invoices_reassigned": 0,
        "vouchers_reassigned": 0,
    }

    for (comp_id, norm_name), parties in grouped.items():
        if len(parties) <= 1 or not norm_name:
            continue
        stats["groups_with_duplicates"] += 1
        canonical = _pick_canonical(parties)
        duplicates = [bp for bp in parties if bp.id != canonical.id]

        logger.info(
            "Company=%s | '%s' -> canonical BP #%s ; duplicates=%s",
            comp_id,
            norm_name,
            canonical.id,
            [d.id for d in duplicates],
        )

        for dup in duplicates:
            moved = _reassign_duplicate(dup, canonical)
            stats["billing_parties_deleted"] += 1
            stats["mappings_reassigned"] += moved["mappings"]
            stats["client_links_updated"] += moved["client_links_updated"]
            stats["client_links_deleted"] += moved["client_links_deleted"]
            stats["bookings_reassigned"] += moved["bookings"]
            stats["invoices_reassigned"] += moved["invoices"]
            stats["vouchers_reassigned"] += moved["vouchers"]

    if dry_run:
        db.session.rollback()
        logger.info("[dry-run] rollback effectué.")
    else:
        db.session.commit()
        logger.info("commit effectué.")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Analyse sans commit")
    parser.add_argument("--company-id", type=int, default=None, help="Limiter à une company")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        stats = run(dry_run=bool(args.dry_run), company_id=args.company_id)
        logger.info("Résumé: %s", stats)


if __name__ == "__main__":
    main()

