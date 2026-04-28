from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import func, or_, text

from ext import db
from models import (
    Booking,
    BookingStatus,
    Client,
    ClientType,
    Company,
    CompanyBillingSettings,
    DispatchMode,
    Driver,
    Institution,
    InstitutionPatient,
    InstitutionRole,
    Invoice,
    InvoiceStatus,
    RequestStatus,
    TransportRequest,
    User,
    UserRole,
    Vehicle,
)
from models.enums import ManagementMode
from services.demo.seed_spec import PROFILES, build_relative_transport_slots
from services.demo.utils import get_demo_default_password

DEMO_EMAIL_DOMAIN = "demo.lirie.ch"
logger = logging.getLogger(__name__)


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _database_name_from_url(database_url: str) -> str:
    parsed = urlparse(database_url)
    return parsed.path.lstrip("/").strip().lower()


def _assert_demo_seed_environment() -> None:
    """Empêche le seed démo en base officielle."""
    if _is_truthy(os.getenv("ALLOW_NON_DEMO_SEED")):
        return

    database_url = (
        os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI") or ""
    ).strip()
    db_name = _database_name_from_url(database_url)
    if "demo" in db_name:
        return

    raise RuntimeError("Seed demo bloque: base non-demo detectee. Utilisez une base *_demo ou ALLOW_NON_DEMO_SEED=true.")

DEMO_PATIENT_IDENTITIES: list[tuple[str, str]] = [
    ("Aline", "Morel"),
    ("Karim", "Haddad"),
    ("Sophie", "Vuille"),
    ("Nicolas", "Bernasconi"),
    ("Lea", "Rochat"),
    ("Omar", "Bensaid"),
    ("Camille", "Fournier"),
    ("Mathis", "Perrin"),
    ("Nadia", "Benali"),
    ("Luc", "Grosjean"),
    ("Ines", "Muller"),
    ("Julien", "Meyer"),
]

DEMO_DRIVER_IDENTITIES: list[tuple[str, str]] = [
    ("Yanis", "Dubois"),
    ("Maya", "Schmidt"),
    ("Romain", "Favre"),
    ("Sarah", "Aubert"),
    ("Loic", "Gauthier"),
    ("Nora", "Borel"),
]

# Coordonnées chauffeurs démo sur voirie (Genève), pour éviter les points en zone lac.
DEMO_DRIVER_GPS_POINTS: list[tuple[str, str]] = [
    ("46.2049", "6.1437"),  # Plainpalais
    ("46.2106", "6.1289"),  # Saint-Jean
    ("46.1959", "6.1396"),  # Carouge
    ("46.2181", "6.1117"),  # Meyrin route
    ("46.1815", "6.1305"),  # La Praille
    ("46.2142", "6.1465"),  # Pâquis / rive droite
]

DEMO_GENEVA_ADDRESSES: list[dict[str, str]] = [
    {"street": "Rue de Carouge 58", "zip": "1205", "city": "Geneve", "lat": "46.1937", "lon": "6.1450"},
    {"street": "Avenue Wendt 5", "zip": "1203", "city": "Geneve", "lat": "46.2145", "lon": "6.1269"},
    {"street": "Rue de Lyon 93", "zip": "1203", "city": "Geneve", "lat": "46.2104", "lon": "6.1212"},
    {"street": "Rue de Lausanne 71", "zip": "1202", "city": "Geneve", "lat": "46.2162", "lon": "6.1478"},
    {"street": "Chemin des Palettes 24", "zip": "1212", "city": "Grand-Lancy", "lat": "46.1769", "lon": "6.1102"},
    {"street": "Route de Meyrin 33", "zip": "1203", "city": "Geneve", "lat": "46.2178", "lon": "6.1115"},
    {"street": "Avenue de la Praille 35", "zip": "1227", "city": "Carouge", "lat": "46.1810", "lon": "6.1300"},
    {"street": "Rue de Montchoisy 46", "zip": "1207", "city": "Geneve", "lat": "46.2025", "lon": "6.1607"},
]

DEMO_DROPOFF_LOCATIONS: list[str] = [
    "Hopital cantonal, Rue Gabrielle-Perret-Gentil 4, 1205 Geneve",
    "Clinique de Carouge, Avenue Cardinal-Mermillod 1, 1227 Carouge",
    "Centre de soins de Chene, Route de Chene 100, 1224 Chene-Bougeries",
    "Policlinique des Acacias, Rue des Epinettes 19, 1227 Les Acacias",
]


def _identity_for_index(items: list[tuple[str, str]], idx: int) -> tuple[str, str]:
    return items[idx % len(items)]


def _address_for_index(idx: int) -> tuple[str, str, str, Decimal, Decimal]:
    row = DEMO_GENEVA_ADDRESSES[idx % len(DEMO_GENEVA_ADDRESSES)]
    return (
        row["street"],
        row["city"],
        row["zip"],
        Decimal(row["lat"]),
        Decimal(row["lon"]),
    )


def _driver_coords_for_index(idx: int) -> tuple[Decimal, Decimal]:
    lat, lon = DEMO_DRIVER_GPS_POINTS[idx % len(DEMO_DRIVER_GPS_POINTS)]
    return Decimal(lat), Decimal(lon)


def _is_placeholder_name(first_name: str | None, last_name: str | None) -> bool:
    first = str(first_name or "").strip().lower()
    last = str(last_name or "").strip().lower()
    return (
        first in {"patient", "chauffeur"}
        or "demo" in first
        or "demo" in last
        or "de mo" in last
        or "de mo" in first
    )


def _is_placeholder_pickup(value: str | None) -> bool:
    text_value = str(value or "").strip().lower()
    return (
        text_value.startswith("adresse pickup demo")
        or text_value.startswith("ems demo")
        or text_value.startswith("ems démo")
        or "rue demo" in text_value
    )


def _is_placeholder_dropoff(value: str | None) -> bool:
    text_value = str(value or "").strip().lower()
    return text_value.startswith("hug - service") or text_value.startswith("hug service")


def _is_placeholder_customer_name(value: str | None) -> bool:
    text_value = str(value or "").strip().lower()
    return text_value.startswith("patient demo") or " demo " in text_value


def _demo_email(local_part: str) -> str:
    return f"{local_part}@{DEMO_EMAIL_DOMAIN}"


def _demo_email_filter_expr(column: Any) -> Any:
    lowered = func.lower(func.coalesce(column, ""))
    return or_(
        lowered.like(f"%@{DEMO_EMAIL_DOMAIN}"),
        lowered.like("%@demo.local"),
        lowered.like("demo-%@%"),
    )


def _upsert_user(
    *,
    username: str,
    email: str,
    role: UserRole,
    first_name: str,
    last_name: str,
    institution_id: int | None = None,
    institution_role: str | None = None,
) -> User:
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User()
        user.email = email
    user.username = username
    user.role = role
    user.first_name = first_name
    user.last_name = last_name
    user.institution_id = institution_id
    user.institution_role = institution_role
    user.account_status = "active"
    user.set_password(get_demo_default_password())
    db.session.add(user)
    db.session.flush()
    return user


def _ensure_demo_enum_compatibility() -> None:
    """Garantit les valeurs d'enum nécessaires au seed démo."""
    db.session.execute(
        text("ALTER TYPE user_role ADD VALUE IF NOT EXISTS 'INSTITUTION'")
    )
    db.session.commit()


def _reset_existing_demo_data() -> None:
    demo_users = User.query.filter(
        _demo_email_filter_expr(User.email)
    ).all()
    demo_user_ids = [user.id for user in demo_users]
    demo_company_ids = [
        company_id
        for (company_id,) in db.session.query(Company.id)
        .filter(Company.user_id.in_(demo_user_ids))
        .all()
    ]

    if demo_user_ids:
        # Respecter les contraintes FK: supprimer d'abord les entités qui référencent
        # clients/companies avant de supprimer ces dernières.
        Booking.query.filter(
            Booking.user_id.in_(demo_user_ids) | Booking.company_id.in_(demo_company_ids)
        ).delete(synchronize_session=False)
        TransportRequest.query.filter(
            TransportRequest.created_by_user_id.in_(demo_user_ids)
            | TransportRequest.accepted_by_company_id.in_(demo_company_ids)
        ).delete(synchronize_session=False)
        Invoice.query.filter(Invoice.company_id.in_(demo_company_ids)).delete(
            synchronize_session=False
        )
        CompanyBillingSettings.query.filter(
            CompanyBillingSettings.company_id.in_(demo_company_ids)
        ).delete(synchronize_session=False)
        Driver.query.filter(Driver.user_id.in_(demo_user_ids)).delete(
            synchronize_session=False
        )
        Client.query.filter(Client.user_id.in_(demo_user_ids)).delete(
            synchronize_session=False
        )
        Company.query.filter(Company.user_id.in_(demo_user_ids)).delete(
            synchronize_session=False
        )
        TransportRequest.query.filter(
            TransportRequest.created_by_user_id.in_(demo_user_ids)
        ).delete(synchronize_session=False)
        User.query.filter(User.id.in_(demo_user_ids)).delete(synchronize_session=False)

    Institution.query.filter(
        _demo_email_filter_expr(Institution.contact_email)
    ).delete(synchronize_session=False)

    db.session.commit()


def reset_and_seed_demo_dataset(
    *,
    profile_name: str = "sales",
    reference_day: date | None = None,
    reset: bool = True,
) -> dict[str, int]:
    _assert_demo_seed_environment()
    if profile_name not in PROFILES:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"Profil inconnu `{profile_name}` (attendus: {supported})")

    profile = PROFILES[profile_name]
    reference_day = reference_day or datetime.now(UTC).date()
    _ensure_demo_enum_compatibility()

    if reset:
        _reset_existing_demo_data()

    companies: list[Company] = []
    for idx in range(profile.companies):
        owner = _upsert_user(
            username=f"demo_company_{idx + 1}",
            email=_demo_email(f"company{idx + 1}"),
            role=UserRole.COMPANY,
            first_name="Compte",
            last_name=f"Transport {idx + 1}",
        )
        company = Company.query.filter_by(user_id=owner.id).first()
        if not company:
            company = Company()
            company.user_id = owner.id
        company.name = f"LIRIE Demo Transport {idx + 1}"
        company.address = "Genève, Suisse"
        company.contact_email = _demo_email(f"ops{idx + 1}")
        company.contact_phone = "+41 22 000 00 00"
        company.dispatch_enabled = True
        company.dispatch_mode = DispatchMode.MANUAL
        company.is_approved = True
        db.session.add(company)
        db.session.flush()
        companies.append(company)

    vehicles: list[Vehicle] = []
    for idx in range(profile.vehicles):
        company = companies[idx % len(companies)]
        plate = f"DEMO-{idx + 1:03d}"
        vehicle = Vehicle.query.filter_by(
            company_id=company.id, license_plate=plate
        ).first()
        if not vehicle:
            vehicle = Vehicle()
            vehicle.company_id = company.id
            vehicle.license_plate = plate
        vehicle.model = ["Mercedes Vito", "VW Caddy", "Ford Transit"][idx % 3]
        vehicle.year = 2022 + (idx % 3)
        vehicle.seats = 5 + (idx % 3)
        vehicle.wheelchair_accessible = idx % 2 == 0
        db.session.add(vehicle)
        db.session.flush()
        vehicles.append(vehicle)

    drivers: list[Driver] = []
    for idx in range(profile.drivers):
        company = companies[idx % len(companies)]
        first_name, last_name = _identity_for_index(DEMO_DRIVER_IDENTITIES, idx)
        user = _upsert_user(
            username=f"demo_driver_{idx + 1}",
            email=_demo_email(f"driver{idx + 1}"),
            role=UserRole.DRIVER,
            first_name=first_name,
            last_name=last_name,
        )
        driver = Driver.query.filter_by(user_id=user.id).first()
        if not driver:
            driver = Driver()
            driver.user_id = user.id
        driver.company_id = company.id
        driver.vehicle_id = vehicles[idx % len(vehicles)].id
        driver.vehicle_assigned = vehicles[idx % len(vehicles)].model
        driver.brand = "LIRIE Demo"
        driver.license_plate = vehicles[idx % len(vehicles)].license_plate
        driver.is_active = True
        driver.is_available = True
        driver_lat, driver_lon = _driver_coords_for_index(idx)
        driver.latitude = driver_lat
        driver.longitude = driver_lon
        db.session.add(driver)
        db.session.flush()
        drivers.append(driver)

    institutions: list[Institution] = []
    for idx in range(profile.institutions):
        email = _demo_email(f"institution{idx + 1}")
        institution = Institution.query.filter_by(contact_email=email).first()
        if not institution:
            institution = Institution()
        institution.name = f"Institution Démo {idx + 1}"
        institution.institution_type = ["ems", "clinic", "curatelle"][idx % 3]
        institution.address = "Genève, Suisse"
        institution.contact_email = email
        institution.contact_phone = "+41 22 100 10 10"
        db.session.add(institution)
        db.session.flush()
        institutions.append(institution)

        _upsert_user(
            username=f"demo_institution_{idx + 1}",
            email=_demo_email(f"institution.user{idx + 1}"),
            role=UserRole.INSTITUTION,
            first_name="Référent",
            last_name=f"Institution {idx + 1}",
            institution_id=institution.id,
            institution_role=InstitutionRole.REQUESTER.value,
        )

    clients: list[Client] = []
    for idx in range(profile.patients):
        first_name, last_name = _identity_for_index(DEMO_PATIENT_IDENTITIES, idx)
        user = _upsert_user(
            username=f"demo_patient_{idx + 1}",
            email=_demo_email(f"patient{idx + 1}"),
            role=UserRole.CLIENT,
            first_name=first_name,
            last_name=last_name,
        )
        company = companies[idx % len(companies)]
        client = Client.query.filter_by(user_id=user.id, company_id=company.id).first()
        if not client:
            client = Client()
            client.user_id = user.id
            client.company_id = company.id
        domicile_address, domicile_city, domicile_zip, domicile_lat, domicile_lon = (
            _address_for_index(idx)
        )
        client.contact_email = user.email
        client.contact_phone = f"+41 79 500 0{idx:02d}"
        client.client_type = ClientType.TRANSPORT
        client.management_mode = ManagementMode.MANAGED
        client.is_active = True
        client.domicile_address = domicile_address
        client.domicile_city = domicile_city
        client.domicile_zip = domicile_zip
        client.domicile_lat = domicile_lat
        client.domicile_lon = domicile_lon
        client.billing_address = domicile_address
        client.billing_lat = domicile_lat
        client.billing_lon = domicile_lon
        client.door_code = f"D{idx + 1:02d}"
        client.floor = str((idx % 6) + 1)
        client.access_notes = "Interphone principal, acces PMR disponible."
        client.gp_name = f"Dr {last_name}"
        client.gp_phone = "+41 22 500 00 00"
        client.default_billed_to_type = "patient"
        client.default_billed_to_contact = (
            f"{user.first_name or 'Patient'} {user.last_name or ''}".strip()
        )
        client.residence_facility = f"Residence Les Tilleuls {((idx % 4) + 1)}"
        db.session.add(client)
        db.session.flush()
        clients.append(client)

        inst = institutions[idx % len(institutions)]
        patient = InstitutionPatient.query.filter_by(
            institution_id=inst.id, external_reference=f"DEMO-PAT-{idx + 1:03d}"
        ).first()
        if not patient:
            patient = InstitutionPatient()
            patient.institution_id = inst.id
            patient.external_reference = f"DEMO-PAT-{idx + 1:03d}"
        patient.first_name = user.first_name or "Patient"
        patient.last_name = user.last_name or "Démo"
        patient.city = "Genève"
        patient.postal_code = "1200"
        patient.address = domicile_address
        db.session.add(patient)

    db.session.flush()

    slots = build_relative_transport_slots(reference_day, profile)
    bookings_created = 0
    requests_created = 0
    for idx, (scheduled_time, status) in enumerate(slots):
        company = companies[idx % len(companies)]
        client = clients[idx % len(clients)]
        driver = drivers[idx % len(drivers)] if drivers else None

        booking = Booking()
        booking.user_id = client.user_id
        booking.client_id = client.id
        booking.company_id = company.id
        booking.customer_name = (
            f"{client.user.first_name} {client.user.last_name}".strip()
        )
        booking.scheduled_time = scheduled_time.replace(tzinfo=None)
        booking.pickup_location = client.domicile_address or _address_for_index(idx)[0]
        booking.dropoff_location = DEMO_DROPOFF_LOCATIONS[
            idx % len(DEMO_DROPOFF_LOCATIONS)
        ]
        booking.amount = float(Decimal("55.00") + Decimal(idx % 12))
        if status in {BookingStatus.ASSIGNED, BookingStatus.ACCEPTED} and driver:
            booking.driver_id = driver.id
        if status == BookingStatus.COMPLETED:
            booking.completed_at = scheduled_time + timedelta(minutes=35)
            if driver:
                booking.driver_id = driver.id
        booking.status = status
        db.session.add(booking)
        bookings_created += 1

        institution = institutions[idx % len(institutions)]
        ext_ref = f"DEMO-REQ-{idx + 1:04d}"
        request = TransportRequest.find_by_external_reference(
            institution.id, ext_ref
        )
        if not request:
            request = TransportRequest()
            request.institution_id = institution.id
            request.created_by_user_id = (
                institution.users[0].id if institution.users else None
            )
            request.external_reference = ext_ref
            request.scheduled_time = scheduled_time
            request.pickup_location = booking.pickup_location
            request.dropoff_location = booking.dropoff_location
            if status == BookingStatus.COMPLETED:
                request.status = RequestStatus.CONVERTED.value
                request.converted_at = datetime.now(UTC)
            elif status == BookingStatus.PENDING:
                request.status = RequestStatus.SENT.value
                request.sent_at = datetime.now(UTC)
            else:
                request.status = RequestStatus.ACCEPTED.value
                request.accepted_at = datetime.now(UTC)
            request.accepted_by_company_id = company.id
            request.is_round_trip = idx % 5 == 0
            db.session.add(request)
            requests_created += 1

    db.session.flush()

    invoices_total = (
        profile.invoices_draft + profile.invoices_sent + profile.invoices_paid
    )
    invoice_statuses = (
        [InvoiceStatus.DRAFT] * profile.invoices_draft
        + [InvoiceStatus.SENT] * profile.invoices_sent
        + [InvoiceStatus.PAID] * profile.invoices_paid
    )
    for idx in range(invoices_total):
        company = companies[idx % len(companies)]
        client = clients[idx % len(clients)]
        issued_at = datetime.combine(reference_day, datetime.min.time()).replace(
            hour=9 + (idx % 4), minute=0, tzinfo=UTC
        )
        invoice_number = f"DEMO-{reference_day.strftime('%Y%m')}-{idx + 1:04d}"
        invoice = Invoice.query.filter_by(
            company_id=company.id, invoice_number=invoice_number
        ).first()
        if not invoice:
            invoice = Invoice()
            invoice.company_id = company.id
            invoice.invoice_number = invoice_number
        invoice.company_id = company.id
        invoice.client_id = client.id
        invoice.period_month = reference_day.month
        invoice.period_year = reference_day.year
        invoice.invoice_number = invoice_number
        invoice.currency = "CHF"
        amount = Decimal("120.00") + Decimal(idx * 10)
        invoice.subtotal_amount = amount
        invoice.total_amount = amount
        invoice.balance_due = (
            Decimal("0.00") if invoice_statuses[idx] == InvoiceStatus.PAID else amount
        )
        invoice.amount_paid = (
            amount if invoice_statuses[idx] == InvoiceStatus.PAID else Decimal("0.00")
        )
        invoice.issued_at = issued_at
        invoice.due_date = issued_at + timedelta(days=30)
        invoice.status = invoice_statuses[idx]
        db.session.add(invoice)

    db.session.commit()

    return {
        "companies": len(companies),
        "institutions": len(institutions),
        "drivers": len(drivers),
        "vehicles": len(vehicles),
        "patients": len(clients),
        "bookings": bookings_created,
        "transport_requests": requests_created,
        "invoices": invoices_total,
    }


def ensure_demo_reference_dataset(
    *,
    profile_name: str = "sales",
) -> dict[str, int]:
    """Garantit un socle de démo partagé, sans reset destructif.

    Si le dataset est incomplet, on complète via le seed déterministe.
    """
    _assert_demo_seed_environment()
    if profile_name not in PROFILES:
        supported = ", ".join(sorted(PROFILES))
        raise ValueError(f"Profil inconnu `{profile_name}` (attendus: {supported})")

    profile = PROFILES[profile_name]
    companies_count = (
        Company.query.join(User, Company.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .count()
    )
    drivers_count = (
        Driver.query.join(User, Driver.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .count()
    )
    clients_count = (
        Client.query.join(User, Client.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .count()
    )
    bookings_count = (
        Booking.query.join(Company, Booking.company_id == Company.id)
        .join(User, Company.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .count()
    )
    invoices_count = (
        Invoice.query.join(Company, Invoice.company_id == Company.id)
        .join(User, Company.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .count()
    )

    needs_reseed = (
        companies_count < profile.companies
        or drivers_count < profile.drivers
        or clients_count < profile.patients
        or bookings_count
        < (
            profile.transports_completed
            + profile.transports_today
            + profile.transports_tomorrow
        )
        or invoices_count
        < (profile.invoices_draft + profile.invoices_sent + profile.invoices_paid)
    )

    if needs_reseed:
        logger.info(
            "[demo_seed] dataset incomplet, completion automatique (profile=%s)",
            profile_name,
        )
        return reset_and_seed_demo_dataset(profile_name=profile_name, reset=False)

    # Backfill défensif: forcer des champs domicile complets pour les clients démo.
    demo_clients = (
        Client.query.join(User, Client.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .all()
    )
    touched = 0
    for idx, client in enumerate(demo_clients):
        user = getattr(client, "user", None)
        if client.client_type is None:
            client.client_type = ClientType.TRANSPORT
            client.management_mode = ManagementMode.MANAGED
            touched += 1
        if not bool(getattr(client, "is_active", True)):
            client.is_active = True
            touched += 1
        if user and _is_placeholder_name(user.first_name, user.last_name):
            first_name, last_name = _identity_for_index(DEMO_PATIENT_IDENTITIES, idx)
            user.first_name = first_name
            user.last_name = last_name
            touched += 1

        has_full_domicile = (
            bool(client.domicile_address)
            and bool(client.domicile_city)
            and bool(client.domicile_zip)
            and client.domicile_lat is not None
            and client.domicile_lon is not None
        )
        address, city, postal_code, lat, lon = _address_for_index(idx)
        if not has_full_domicile:
            client.domicile_address = client.domicile_address or address
            client.domicile_city = client.domicile_city or city
            client.domicile_zip = client.domicile_zip or postal_code
            client.domicile_lat = client.domicile_lat or lat
            client.domicile_lon = client.domicile_lon or lon
            touched += 1
        elif _is_placeholder_pickup(client.domicile_address):
            client.domicile_address = address
            client.domicile_city = city
            client.domicile_zip = postal_code
            client.domicile_lat = lat
            client.domicile_lon = lon
            touched += 1
        client.billing_address = client.billing_address or client.domicile_address

        if user and _is_placeholder_name(
            getattr(user, "first_name", ""), getattr(user, "last_name", "")
        ):
            client.default_billed_to_contact = (
                f"{user.first_name or 'Patient'} {user.last_name or ''}".strip()
            )
            touched += 1

    demo_drivers = (
        Driver.query.join(User, Driver.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .order_by(Driver.id.asc())
        .all()
    )
    for idx, driver in enumerate(demo_drivers):
        target_lat, target_lon = _driver_coords_for_index(idx)
        if driver.latitude != target_lat:
            driver.latitude = target_lat
            touched += 1
        if driver.longitude != target_lon:
            driver.longitude = target_lon
            touched += 1

    demo_bookings = (
        Booking.query.join(Company, Booking.company_id == Company.id)
        .join(User, Company.user_id == User.id)
        .filter(_demo_email_filter_expr(User.email))
        .all()
    )
    for idx, booking in enumerate(demo_bookings):
        if _is_placeholder_customer_name(booking.customer_name):
            client_user = getattr(getattr(booking, "client", None), "user", None)
            first_name = getattr(client_user, "first_name", "") or ""
            last_name = getattr(client_user, "last_name", "") or ""
            full_name = f"{first_name} {last_name}".strip()
            if full_name:
                booking.customer_name = full_name
                touched += 1
        if _is_placeholder_pickup(booking.pickup_location):
            booking.pickup_location = _address_for_index(idx)[0]
            touched += 1
        if _is_placeholder_dropoff(booking.dropoff_location):
            booking.dropoff_location = DEMO_DROPOFF_LOCATIONS[
                idx % len(DEMO_DROPOFF_LOCATIONS)
            ]
            touched += 1

    if touched:
        db.session.commit()
        logger.info("[demo_seed] clients domicile backfill=%s", touched)

    return {
        "companies": companies_count,
        "drivers": drivers_count,
        "patients": clients_count,
        "bookings": bookings_count,
        "invoices": invoices_count,
    }
