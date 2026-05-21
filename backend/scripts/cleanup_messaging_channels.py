#!/usr/bin/env python3
"""
Nettoie les messages mélangés entre canaux (dispatch / équipe / mission / DM).

Usage (depuis backend/, conteneur ou venv avec Flask app) :
  python scripts/cleanup_messaging_channels.py --dry-run
  python scripts/cleanup_messaging_channels.py --company-id 9 --purge-messages --yes
  python scripts/cleanup_messaging_channels.py --rebuild-conversations --yes

Par défaut : dry-run (aucune écriture).
"""

from __future__ import annotations

import argparse
import sys

from app import create_app
from ext import db
from models import Company, Conversation, Driver, Message, MessageRead
from services.messaging.conversation_service import ConversationService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nettoyage messagerie multi-canaux")
    parser.add_argument("--company-id", type=int, default=None, help="Limiter à une entreprise")
    parser.add_argument("--dry-run", action="store_true", help="Simulation (défaut si pas --yes)")
    parser.add_argument(
        "--purge-messages",
        action="store_true",
        help="Supprimer tous les messages (et lectures) de l'entreprise",
    )
    parser.add_argument(
        "--rebuild-conversations",
        action="store_true",
        help="Recréer dispatch partagé + équipe (sans supprimer les conversations existantes)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Exécuter réellement (sans ceci = dry-run)",
    )
    return parser.parse_args()


def _company_ids(company_id: int | None) -> list[int]:
    if company_id is not None:
        return [company_id]
    return [int(c.id) for c in Company.query.order_by(Company.id).all()]


def main() -> int:
    args = _parse_args()
    dry_run = not args.yes

    app = create_app()
    with app.app_context():
        company_ids = _company_ids(args.company_id)
        if not company_ids:
            print("Aucune entreprise trouvée.")
            return 1

        for cid in company_ids:
            msg_count = Message.query.filter_by(company_id=cid).count()
            null_thread = Message.query.filter_by(company_id=cid, thread_id=None).count()
            print(f"\n=== Entreprise {cid} ===")
            print(f"  messages total      : {msg_count}")
            print(f"  sans thread_id      : {null_thread}")

            if args.purge_messages:
                read_count = (
                    db.session.query(MessageRead)
                    .join(Message, MessageRead.message_id == Message.id)
                    .filter(Message.company_id == cid)
                    .count()
                )
                print(f"  message_read liés   : {read_count}")
                if dry_run:
                    print("  [dry-run] purge messages ignorée")
                else:
                    db.session.query(MessageRead).filter(
                        MessageRead.message_id.in_(
                            db.session.query(Message.id).filter(Message.company_id == cid)
                        )
                    ).delete(synchronize_session=False)
                    Message.query.filter_by(company_id=cid).delete(synchronize_session=False)
                    db.session.commit()
                    print("  ✓ messages supprimés")

            if args.rebuild_conversations:
                if dry_run:
                    print("  [dry-run] rebuild conversations (dispatch + équipe)")
                else:
                    ConversationService.ensure_company_dispatch_conversation(cid)
                    ConversationService.ensure_company_group_conversation(cid)
                    drivers = Driver.query.filter_by(company_id=cid, is_active=True).all()
                    for d in drivers:
                        if d.user_id:
                            ConversationService.ensure_company_driver_conversation(cid, d)
                    print("  ✓ conversations canons provisionnées")

        if dry_run and (args.purge_messages or args.rebuild_conversations):
            print("\nRelancer avec --yes pour appliquer.")
        elif not args.purge_messages and not args.rebuild_conversations:
            print("\nIndiquez --purge-messages et/ou --rebuild-conversations (avec --yes).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
