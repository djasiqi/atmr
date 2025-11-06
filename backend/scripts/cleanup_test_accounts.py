"""Utility script to inspect and optionally purge fake test accounts.

Usage examples (inside repository root):

    docker-compose exec -T api python scripts/cleanup_test_accounts.py
    docker-compose exec -T api python scripts/cleanup_test_accounts.py --delete

The first command performs a dry-run and prints the accounts that would be
removed. The second command deletes them from the database.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime
from typing import Iterable, List, Optional

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import selectinload

from app import create_app
from db import db
from models.ab_test_result import ABTestResult
from models.autonomous_action import AutonomousAction
from models.booking import Booking
from models.client import Client
from models.company import Company
from models.driver import Driver
from models.enums import UserRole
from models.invoice import CompanyBillingSettings, Invoice, InvoiceSequence
from models.user import User

TEST_USERNAME_PREFIXES: tuple[str, ...] = (
    "testuser_",
    "admin_",
    "client_",
    "driver_",
    "target_",
    "inst_",
)

TEST_EMAIL_DOMAIN = "%@example.com"


def iter_test_users() -> Iterable[User]:
    """Return all test users that match our heuristics."""

    filters = [User.username.ilike(f"{prefix}%") for prefix in TEST_USERNAME_PREFIXES]
    criterion = and_(User.email.ilike(TEST_EMAIL_DOMAIN), or_(*filters))

    return (
        User.query.filter(criterion)
        .order_by(User.created_at.asc())
        .all()
    )


def iter_users_by_created_date(target_date: date) -> Iterable[User]:
    return (
        User.query.filter(func.date(User.created_at) == target_date)
        .order_by(User.created_at.asc())
        .all()
    )


def resolve_company_by_name(name: str) -> Optional[Company]:
    normalized = name.strip().lower()
    if not normalized:
        return None
    return Company.query.filter(func.lower(Company.name) == normalized).first()


def client_filter_outside(company_id: int):
    return or_(Client.company_id.is_(None), Client.company_id != company_id)


def normalize_user_role(value) -> Optional[UserRole]:
    if isinstance(value, UserRole):
        return value
    if value is None:
        return None
    try:
        return UserRole(value)
    except ValueError:
        return None


def summarize(users: list[User]) -> str:
    counter = Counter(user.role for user in users)
    parts = [
        f"{len(users)} utilisateurs test identifiés",
    ]
    if counter:
        details = ", ".join(
            f"{count}×{role.value}" if isinstance(role, UserRole) else f"{count}×{role}"
            for role, count in counter.most_common()
        )
        parts.append(f"Répartition par rôle: {details}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecte/Supprime les comptes de test.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Supprime réellement les comptes identifiés (sinon dry-run).",
    )
    parser.add_argument(
        "--keep-company",
        dest="keep_company",
        help="Nom exact de l'entreprise dont les clients doivent être conservés.",
    )
    parser.add_argument(
        "--created-on",
        dest="created_on",
        help="Supprime les utilisateurs créés à cette date (format AAAA-MM-JJ).",
    )
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        users = list(iter_test_users())
        print(summarize(users))

        keep_company = None
        outside_clients_count = 0
        drivers_outside_count = 0
        created_on_date = None
        users_created_on: list[User] = []
        if args.keep_company:
            keep_company = resolve_company_by_name(args.keep_company)
            if not keep_company:
                print(f"❌ Entreprise '{args.keep_company}' introuvable dans la base.")
            else:
                outside_clients_count = (
                    Client.query.filter(client_filter_outside(keep_company.id)).count()
                )
                print(
                    f"{outside_clients_count} client(s) hors '{keep_company.name}' identifiés"
                )
                drivers_outside_count = (
                    Driver.query.filter(Driver.company_id != keep_company.id).count()
                )
                print(
                    f"{drivers_outside_count} chauffeur(s) hors '{keep_company.name}' identifiés"
                )

        if args.created_on:
            try:
                created_on_date = datetime.strptime(args.created_on, "%Y-%m-%d").date()
                users_created_on = list(iter_users_by_created_date(created_on_date))
                print(
                    f"{len(users_created_on)} utilisateur(s) créé(s) le {created_on_date.isoformat()}"
                )
            except ValueError:
                print(f"❌ Date invalide pour --created-on : {args.created_on} (attendu AAAA-MM-JJ)")
                users_created_on = []
                created_on_date = None

        for user in users:
            role = user.role.value if isinstance(user.role, UserRole) else user.role
            print(f" - #{user.id} {user.username} <{user.email}> ({role}) créé le {user.created_at}")

        if not args.delete:
            print("\nMode dry-run : aucune suppression effectuée. Ajoutez --delete pour nettoyer.")
            return

        total_deleted_records = 0
        total_deleted_users = 0
        pending_user_ids = {user.id for user in users}

        if keep_company and outside_clients_count:
            clients_to_delete = (
                Client.query
                .options(selectinload(Client.user))
                .filter(client_filter_outside(keep_company.id))
                .all()
            )
            deleted_users_for_clients = 0
            for client in clients_to_delete:
                user = client.user
                role_enum = normalize_user_role(getattr(user, "role", None)) if user else None
                if user and role_enum == UserRole.CLIENT:
                    db.session.delete(user)
                    deleted_users_for_clients += 1
                else:
                    db.session.delete(client)
            total_deleted_records += len(clients_to_delete)
            total_deleted_users += deleted_users_for_clients
            print(
                f"Suppression des clients hors '{keep_company.name}': {len(clients_to_delete)} (dont {deleted_users_for_clients} utilisateur(s) supprimé(s))"
            )

        if users:
            user_ids = [user.id for user in users]
            company_ids: List[int] = [
                company_id
                for (company_id,) in db.session.query(Company.id).filter(Company.user_id.in_(user_ids)).all()
            ]

            if company_ids:
                deleted_autonomous = (
                    db.session.query(AutonomousAction)
                    .filter(AutonomousAction.company_id.in_(company_ids))
                    .delete(synchronize_session=False)
                )
                if deleted_autonomous:
                    print(f"Suppression des actions autonomes associées: {deleted_autonomous}")

                deleted_billing = (
                    db.session.query(CompanyBillingSettings)
                    .filter(CompanyBillingSettings.company_id.in_(company_ids))
                    .delete(synchronize_session=False)
                )
                if deleted_billing:
                    print(f"Suppression des paramètres de facturation: {deleted_billing}")

                deleted_sequences = (
                    db.session.query(InvoiceSequence)
                    .filter(InvoiceSequence.company_id.in_(company_ids))
                    .delete(synchronize_session=False)
                )
                if deleted_sequences:
                    print(f"Suppression des séquences de facturation: {deleted_sequences}")

                deleted_invoices = (
                    db.session.query(Invoice)
                    .filter(Invoice.company_id.in_(company_ids))
                    .delete(synchronize_session=False)
                )
                if deleted_invoices:
                    print(f"Suppression des factures: {deleted_invoices}")

            for user in users:
                db.session.delete(user)
            total_deleted_records += len(users)
            total_deleted_users += len(users)
            pending_user_ids.update(user_ids)

        if keep_company and drivers_outside_count:
            drivers_to_delete = (
                Driver.query
                .options(selectinload(Driver.user))
                .filter(Driver.company_id != keep_company.id)
                .all()
            )
            driver_ids = [driver.id for driver in drivers_to_delete]
            if driver_ids:
                deleted_ab_tests = (
                    db.session.query(ABTestResult)
                    .filter(ABTestResult.driver_id.in_(driver_ids))
                    .delete(synchronize_session=False)
                )
                if deleted_ab_tests:
                    print(f"Suppression des résultats A/B associés: {deleted_ab_tests}")

            deleted_users_for_drivers = 0
            for driver in drivers_to_delete:
                user = driver.user
                role_enum = normalize_user_role(getattr(user, "role", None)) if user else None
                if user and role_enum == UserRole.DRIVER:
                    db.session.delete(user)
                    deleted_users_for_drivers += 1
                else:
                    db.session.delete(driver)

            total_deleted_records += len(drivers_to_delete)
            total_deleted_users += deleted_users_for_drivers
            print(
                f"Suppression des chauffeurs hors '{keep_company.name}': {len(drivers_to_delete)} (dont {deleted_users_for_drivers} utilisateur(s) supprimé(s))"
            )

        if created_on_date and users_created_on:
            users_filter = [user for user in users_created_on if user.id not in pending_user_ids]

            client_ids_for_users: set[int] = set()
            driver_ids_for_users: set[int] = set()
            company_ids_for_users: set[int] = set()

            for user in users_filter:
                if hasattr(user, "clients"):
                    for client in getattr(user, "clients", []) or []:
                        if client and getattr(client, "id", None) is not None:
                            client_ids_for_users.add(client.id)
                driver = getattr(user, "driver", None)
                if driver and getattr(driver, "id", None) is not None:
                    driver_ids_for_users.add(driver.id)
                company = getattr(user, "company", None)
                if company and getattr(company, "id", None) is not None:
                    company_ids_for_users.add(company.id)

            booking_ids: set[int] = set()
            if client_ids_for_users:
                booking_ids.update(
                    booking_id
                    for (booking_id,) in db.session.query(Booking.id).filter(Booking.client_id.in_(client_ids_for_users)).all()
                )
            if driver_ids_for_users:
                booking_ids.update(
                    booking_id
                    for (booking_id,) in db.session.query(Booking.id).filter(Booking.driver_id.in_(driver_ids_for_users)).all()
                )
            if company_ids_for_users:
                booking_ids.update(
                    booking_id
                    for (booking_id,) in db.session.query(Booking.id).filter(Booking.company_id.in_(company_ids_for_users)).all()
                )

            if driver_ids_for_users:
                deleted_driver_ab = (
                    db.session.query(ABTestResult)
                    .filter(ABTestResult.driver_id.in_(driver_ids_for_users))
                    .delete(synchronize_session=False)
                )
                if deleted_driver_ab:
                    print(f"Suppression des résultats A/B (drivers): {deleted_driver_ab}")

            if booking_ids:
                deleted_booking_ab = (
                    db.session.query(ABTestResult)
                    .filter(ABTestResult.booking_id.in_(booking_ids))
                    .delete(synchronize_session=False)
                )
                if deleted_booking_ab:
                    print(f"Suppression des résultats A/B (bookings): {deleted_booking_ab}")

            if company_ids_for_users:
                deleted_autonomous_created = (
                    db.session.query(AutonomousAction)
                    .filter(AutonomousAction.company_id.in_(company_ids_for_users))
                    .delete(synchronize_session=False)
                )
                if deleted_autonomous_created:
                    print(f"Suppression des actions autonomes (entreprises créées le {created_on_date.isoformat()}): {deleted_autonomous_created}")

                deleted_billing_created = (
                    db.session.query(CompanyBillingSettings)
                    .filter(CompanyBillingSettings.company_id.in_(company_ids_for_users))
                    .delete(synchronize_session=False)
                )
                if deleted_billing_created:
                    print(f"Suppression des paramètres de facturation (entreprises créées le {created_on_date.isoformat()}): {deleted_billing_created}")

                deleted_sequences_created = (
                    db.session.query(InvoiceSequence)
                    .filter(InvoiceSequence.company_id.in_(company_ids_for_users))
                    .delete(synchronize_session=False)
                )
                if deleted_sequences_created:
                    print(f"Suppression des séquences de facturation (entreprises créées le {created_on_date.isoformat()}): {deleted_sequences_created}")

                deleted_invoices_created = (
                    db.session.query(Invoice)
                    .filter(Invoice.company_id.in_(company_ids_for_users))
                    .delete(synchronize_session=False)
                )
                if deleted_invoices_created:
                    print(f"Suppression des factures (entreprises créées le {created_on_date.isoformat()}): {deleted_invoices_created}")

            for user in users_filter:
                db.session.delete(user)

            total_deleted_records += len(users_filter)
            total_deleted_users += len(users_filter)
            pending_user_ids.update(user.id for user in users_filter)
            print(
                f"Suppression des utilisateurs créés le {created_on_date.isoformat()}: {len(users_filter)}"
            )

        if total_deleted_records == 0:
            print("Aucun enregistrement à supprimer.")
            return

        db.session.commit()
        print(
            f"\n✅ Suppression effectuée ({total_deleted_records} enregistrements, {total_deleted_users} utilisateur(s))."
        )


if __name__ == "__main__":
    main()

