#!/usr/bin/env python3
"""Comptage read-only des CLIENT/PORTAL que AUTH-SMS-02 va promouvoir.

Aucune écriture. À exécuter dans le container **avant** et **après**
``flask db upgrade``.

Usage :
    python -m scripts.preview_auth_sms_02_promotion
    python -m scripts.preview_auth_sms_02_promotion --limit 20
"""

from __future__ import annotations

import argparse

from sqlalchemy import text

from ext import db
from services.auth.portal_account_promotion_sql import (
    COUNT_PORTAL_PROMOTION_CLIENTS_SQL,
    COUNT_PORTAL_PROMOTION_USERS_SQL,
    PORTAL_PROMOTION_USER_PREDICATE,
)

PHILIPPE_USER_ID = 192847


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    from app import create_app

    app = create_app()
    with app.app_context():
        has_phone_col = bool(
            db.session.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'user'
                      AND column_name = 'phone_verified_at'
                    """
                )
            ).scalar()
        )
        phone_select = (
            "u.phone_verified_at"
            if has_phone_col
            else "NULL::timestamptz AS phone_verified_at"
        )
        users = db.session.execute(text(COUNT_PORTAL_PROMOTION_USERS_SQL)).scalar()
        clients = db.session.execute(text(COUNT_PORTAL_PROMOTION_CLIENTS_SQL)).scalar()
        rows = db.session.execute(
            text(
                f"""
                SELECT u.id, u.email, u.account_status, {phone_select}
                FROM "user" AS u
                WHERE {PORTAL_PROMOTION_USER_PREDICATE}
                ORDER BY u.id
                LIMIT :limit
                """
            ),
            {"limit": args.limit},
        ).mappings()
        print("AUTH-SMS-02 PROMOTION PREVIEW (READ-ONLY)")
        print("========================================")
        print(f"users_a_promouvoir           : {users}")
        print(f"clients_portal_a_activer     : {clients}")
        print("predicats : pending_activation + CLIENT + PORTAL + email")
        print("            + session.email_verified_at + institution_id IS NULL")
        print("            + disabled_at IS NULL")
        print("phone_verified_at : jamais écrit par cette promotion")
        if has_phone_col:
            print("colonne phone_verified_at   : presente")
        else:
            print("colonne phone_verified_at   : absente (preview avant migration)")
        print("")
        print("echantillon (id, email, status, phone_verified_at)")
        for row in rows:
            print(
                f"  {row['id']}\t{row['email']}\t{row['account_status']}\t"
                f"{row['phone_verified_at']}"
            )
        philippe = (
            db.session.execute(
                text(
                    f"""
                SELECT u.id, u.account_status, {phone_select},
                       EXISTS (
                           SELECT 1 FROM activation_session s
                           WHERE s.user_id = u.id
                             AND s.email_verified_at IS NOT NULL
                       ) AS email_verified
                FROM "user" AS u
                WHERE u.id = :uid
                """
                ),
                {"uid": PHILIPPE_USER_ID},
            )
            .mappings()
            .first()
        )
        print("")
        if philippe:
            print(
                f"philippe user_id={philippe['id']} "
                f"status={philippe['account_status']} "
                f"phone_verified_at={philippe['phone_verified_at']} "
                f"email_verified={philippe['email_verified']}"
            )
        else:
            print(f"philippe user_id={PHILIPPE_USER_ID} : absent de cette base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
