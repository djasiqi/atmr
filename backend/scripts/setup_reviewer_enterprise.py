"""Provisionne un compte reviewer entreprise non-demo, idempotent.

Usage:
  python scripts/setup_reviewer_enterprise.py
  python scripts/setup_reviewer_enterprise.py --verify-only

Le script crée/valide:
- une company dédiée review
- un utilisateur reviewer entreprise (MFA désactivée pour cette company)
- un client et un chauffeur de démonstration
- 3 courses minimales (pending / assigned / completed)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from db import db  # noqa: E402
from models import (  # noqa: E402
    Booking,
    BookingStatus,
    Client,
    Company,
    Driver,
    User,
    UserRole,
)

DEFAULTS = {
    "company_name": "Liri Review Company",
    "reviewer_email": "reviewer.enterprise@liri.ch",
    "reviewer_password": "Review1234!",
    "client_email": "review.client@liri.ch",
    "driver_email": "review.driver@liri.ch",
}


def _now_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _find_user_by_email(email: str) -> User | None:
    return User.query.filter(User.email == email).first()


def _ensure_user(
    *,
    email: str,
    username: str,
    first_name: str,
    last_name: str,
    role: UserRole,
    password: str,
) -> User:
    user = _find_user_by_email(email)
    if user:
        return user
    user = User(
        email=email,
        username=username,
        first_name=first_name,
        last_name=last_name,
        role=role,
        account_status="active",
    )

    from security.password_policy import PasswordPolicyError, PasswordPolicyService

    try:
        PasswordPolicyService.validate_password(
            password, user_id=None, check_history=False
        )
    except PasswordPolicyError as e:
        raise SystemExit(f"Mot de passe invalide (script reviewer): {e}") from e

    user.set_password(password)  # nosemgrep python.django.security.audit.unvalidated-password.unvalidated-password - Flask/SQLAlchemy: PasswordPolicyService.validate_password appele juste avant (pas Django).
    db.session.add(user)
    db.session.flush()
    return user


def _ensure_company(reviewer_user: User, name: str) -> Company:
    company = (
        Company.query.filter(Company.user_id == reviewer_user.id).first()
        or Company.query.filter(Company.name == name).first()
    )
    if not company:
        company = Company(
            name=name,
            user_id=reviewer_user.id,
            is_approved=True,
            dispatch_enabled=True,
            contact_email=reviewer_user.email,
        )
        company.approve()
        db.session.add(company)
        db.session.flush()
    config = company.get_autonomous_config()
    security = dict(config.get("security") or {})
    mobile_mfa = dict(security.get("mobile_mfa") or {})
    mobile_mfa["required"] = False
    security["mobile_mfa"] = mobile_mfa
    config["security"] = security
    company.set_autonomous_config(config)
    return company


def _ensure_client(company: Company, email: str) -> Client:
    user = _ensure_user(
        email=email,
        username="review_client",
        first_name="Review",
        last_name="Client",
        role=UserRole.CLIENT,
        password="Client1234!",
    )
    client = Client.query.filter(
        Client.user_id == user.id, Client.company_id == company.id
    ).first()
    if client:
        return client
    client = Client(
        user_id=user.id,
        company_id=company.id,
        contact_phone="+41790000001",
        domicile_address="Rue de la Review 1, Genève",
    )
    db.session.add(client)
    db.session.flush()
    return client


def _ensure_driver(company: Company, email: str) -> Driver:
    user = _ensure_user(
        email=email,
        username="review_driver",
        first_name="Review",
        last_name="Driver",
        role=UserRole.DRIVER,
        password="Driver1234!",
    )
    driver = Driver.query.filter(Driver.user_id == user.id).first()
    if driver:
        return driver
    driver = Driver(
        user_id=user.id,
        company_id=company.id,
        is_active=True,
        is_available=True,
        vehicle_assigned="Véhicule Review",
        brand="Mercedes",
        license_plate="REVIEW-001",
    )
    db.session.add(driver)
    db.session.flush()
    return driver


def _ensure_booking(
    *,
    company: Company,
    creator: User,
    client: Client,
    customer_name: str,
    pickup: str,
    dropoff: str,
    status: BookingStatus,
    scheduled_time: datetime,
    amount: float,
    driver: Driver | None = None,
) -> Booking:
    booking = (
        Booking.query.filter(
            Booking.company_id == company.id,
            Booking.customer_name == customer_name,
            Booking.pickup_location == pickup,
            Booking.dropoff_location == dropoff,
        )
        .order_by(Booking.id.desc())
        .first()
    )
    if booking:
        return booking

    booking = Booking(
        customer_name=customer_name,
        pickup_location=pickup,
        dropoff_location=dropoff,
        scheduled_time=scheduled_time,
        amount=amount,
        status=status,
        user_id=creator.id,
        client_id=client.id,
        company_id=company.id,
        driver_id=driver.id if driver else None,
    )
    if status == BookingStatus.COMPLETED:
        booking.completed_at = datetime.now(UTC)
    db.session.add(booking)
    db.session.flush()
    return booking


def _print_summary(company: Company, reviewer_user: User) -> None:
    bookings_count = Booking.query.filter(Booking.company_id == company.id).count()
    clients_count = Client.query.filter(Client.company_id == company.id).count()
    drivers_count = Driver.query.filter(Driver.company_id == company.id).count()
    print("Reviewer setup prêt:")
    print(f"- Company: {company.name} (id={company.id})")
    print(f"- Reviewer: {reviewer_user.email}")
    print("- MFA mobile company: disabled")
    print(
        f"- Dataset: {clients_count} client(s), {drivers_count} driver(s), {bookings_count} booking(s)"
    )
    print("Vérification J+7 recommandée: J0/J+1/J+3/J+7 (login + rides + action).")


def setup_reviewer_dataset() -> None:
    reviewer_user = _ensure_user(
        email=DEFAULTS["reviewer_email"],
        username="review_enterprise",
        first_name="Store",
        last_name="Reviewer",
        role=UserRole.COMPANY,
        password=DEFAULTS["reviewer_password"],
    )
    company = _ensure_company(reviewer_user, DEFAULTS["company_name"])
    client = _ensure_client(company, DEFAULTS["client_email"])
    driver = _ensure_driver(company, DEFAULTS["driver_email"])

    now = _now_naive()
    _ensure_booking(
        company=company,
        creator=reviewer_user,
        client=client,
        customer_name="Review Pending",
        pickup="Rue de la Gare 10, Genève",
        dropoff="Hôpital Cantonal, Genève",
        status=BookingStatus.PENDING,
        scheduled_time=now + timedelta(hours=1),
        amount=42.0,
    )
    _ensure_booking(
        company=company,
        creator=reviewer_user,
        client=client,
        customer_name="Review Assigned",
        pickup="Avenue de Frontenex 15, Genève",
        dropoff="Clinique des Grangettes, Chêne-Bougeries",
        status=BookingStatus.ASSIGNED,
        scheduled_time=now + timedelta(hours=2),
        amount=55.0,
        driver=driver,
    )
    _ensure_booking(
        company=company,
        creator=reviewer_user,
        client=client,
        customer_name="Review Completed",
        pickup="Route de Malagnou 35, Genève",
        dropoff="Aéroport de Genève",
        status=BookingStatus.COMPLETED,
        scheduled_time=now - timedelta(hours=3),
        amount=60.0,
        driver=driver,
    )
    db.session.commit()
    _print_summary(company, reviewer_user)


def verify_only() -> int:
    reviewer_user = _find_user_by_email(DEFAULTS["reviewer_email"])
    if not reviewer_user:
        print("Reviewer user introuvable.")
        return 1
    company = Company.query.filter(Company.user_id == reviewer_user.id).first()
    if not company:
        print("Reviewer company introuvable.")
        return 1
    security = company.get_autonomous_config().get("security", {})
    mobile_mfa_required = bool(
        (security.get("mobile_mfa") or {}).get("required", False)
    )
    if mobile_mfa_required:
        print("Reviewer company invalide: mobile_mfa.required=true.")
        return 1
    _print_summary(company, reviewer_user)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision reviewer company + dataset")
    parser.add_argument(
        "--verify-only", action="store_true", help="Vérifie la configuration existante"
    )
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        if args.verify_only:
            raise SystemExit(verify_only())
        setup_reviewer_dataset()


if __name__ == "__main__":
    main()
