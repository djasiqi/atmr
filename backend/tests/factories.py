# ruff: noqa: DTZ001, DTZ003, DTZ005, DTZ011, N815
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false
"""
Factories pour générer des données de test avec factory_boy.
DTZ003/DTZ011 = datetime.utcnow()/date.today() OK dans tests
reportMissingImports = factory/faker installés dans Docker, pas localement
"""
import uuid
from datetime import date, datetime, timedelta
from typing import Any

import factory  # type: ignore[import-untyped]
from factory import fuzzy  # type: ignore[import-untyped]
from faker import Faker  # type: ignore[import-untyped]

from db import db
from models.ab_test_result import ABTestResult
from models.booking import Booking
from models.client import Client
from models.company import Company
from models.dispatch import Assignment, DispatchRun
from models.driver import Driver
from models.enums import (
    AssignmentStatus,
    BookingStatus,
    DispatchMode,
    DispatchStatus,
    DriverType,
)
from models.invoice import Invoice
from models.ml_prediction import MLPrediction
from models.user import User
from models.vehicle import Vehicle

fake = Faker('fr_FR')


class SQLAlchemyModelFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Base factory pour modèles SQLAlchemy."""

    class Meta:
        abstract = True
        sqlalchemy_session = db.session
        sqlalchemy_session_persistence = "commit"


# ========== USER & COMPANY ==========

class UserFactory(SQLAlchemyModelFactory):
    """Factory pour User."""

    class Meta:
        model = User

    username = factory.LazyAttribute(lambda _: f"user_{uuid.uuid4().hex[:8]}")
    email = factory.LazyAttribute(lambda _: f"test_{uuid.uuid4().hex[:12]}@atmr-test.ch")
    password = factory.LazyFunction(
        lambda: "$2b$12$KIXabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"  # Hash bcrypt fake
    )
    first_name = factory.LazyAttribute(lambda _: fake.first_name())
    last_name = factory.LazyAttribute(lambda _: fake.last_name())
    phone = factory.LazyAttribute(lambda _: f"+41{fake.numerify('##########')}")  # Format suisse valide
    role = fuzzy.FuzzyChoice(['driver', 'dispatcher', 'admin', 'client'])
    created_at = factory.LazyFunction(datetime.utcnow)


class CompanyFactory(SQLAlchemyModelFactory):
    """Factory pour Company."""

    class Meta:
        model = Company

    name = factory.LazyAttribute(lambda _: fake.company())
    address = factory.LazyAttribute(lambda _: fake.address().replace('\n', ', ')[:200])
    latitude = factory.LazyAttribute(lambda _: fake.latitude())
    longitude = factory.LazyAttribute(lambda _: fake.longitude())

    contact_email = factory.LazyAttribute(lambda _: fake.company_email())
    contact_phone = factory.LazyAttribute(lambda _: fake.phone_number()[:20])

    iban = factory.LazyAttribute(lambda _: fake.iban())
    uid_ide = factory.LazyAttribute(lambda _: f"CHE-{fake.random_number(digits=9, fix_len=True)}")
    billing_email = factory.LazyAttribute(lambda _: fake.email())

    user = factory.SubFactory(UserFactory, role='admin')
    is_approved = True
    created_at = factory.LazyFunction(datetime.utcnow)

    dispatch_mode = DispatchMode.SEMI_AUTO
    max_daily_bookings = 50


# ========== CLIENT ==========

class ClientFactory(SQLAlchemyModelFactory):
    """Factory pour Client."""

    class Meta:
        model = Client

    user = factory.SubFactory(UserFactory, role='client')
    company = factory.SubFactory(CompanyFactory)

    billing_address = factory.LazyAttribute(lambda _: fake.address().replace('\n', ', ')[:255])
    contact_email = factory.LazyAttribute(lambda _: fake.email())
    contact_phone = factory.LazyAttribute(lambda _: f"+41{fake.numerify('##########')}")


# ========== DRIVER & VEHICLE ==========

class DriverFactory(SQLAlchemyModelFactory):
    """Factory pour Driver."""

    class Meta:
        model = Driver

    user = factory.SubFactory(UserFactory, role='driver')
    company = factory.SubFactory(CompanyFactory)

    vehicle_assigned = factory.LazyAttribute(lambda _: fake.word().capitalize())
    brand = factory.LazyAttribute(lambda _: fake.company()[:100])
    license_plate = factory.LazyAttribute(lambda _: f"{fake.lexify('??')}-{fake.numerify('####')}")

    is_active = True
    is_available = True
    driver_type = DriverType.REGULAR

    latitude = factory.LazyAttribute(lambda _: 46.2 + (fake.random.random() - 0.5) * 0.2)  # Genève area
    longitude = factory.LazyAttribute(lambda _: 6.1 + (fake.random.random() - 0.5) * 0.2)
    last_position_update = factory.LazyFunction(datetime.utcnow)

    contract_type = "CDI"
    weekly_hours = 40
    hourly_rate_cents = 5000  # 50 CHF/h


class VehicleFactory(SQLAlchemyModelFactory):
    """Factory pour Vehicle."""

    class Meta:
        model = Vehicle

    company = factory.SubFactory(CompanyFactory)
    driver = factory.SubFactory(DriverFactory)

    brand = factory.LazyAttribute(lambda _: fake.company()[:50])
    model = factory.LazyAttribute(lambda _: fake.word().capitalize()[:50])
    license_plate = factory.LazyAttribute(lambda _: f"{fake.lexify('??')}-{fake.numerify('####')}")
    year = factory.LazyAttribute(lambda _: fake.random_int(min=2010, max=2025))

    capacity_passengers = fuzzy.FuzzyInteger(4, 8)
    capacity_wheelchairs = fuzzy.FuzzyInteger(0, 2)
    capacity_beds = fuzzy.FuzzyInteger(0, 1)

    is_active = True
    has_medical_equipment = False


# ========== BOOKING ==========

class BookingFactory(SQLAlchemyModelFactory):
    """Factory pour Booking."""

    class Meta:
        model = Booking

    customer_name = factory.LazyAttribute(lambda _: fake.name())

    pickup_location = factory.LazyAttribute(lambda _: fake.address().replace('\n', ', ')[:200])
    pickup_lat = factory.LazyAttribute(lambda _: 46.2 + (fake.random.random() - 0.5) * 0.2)
    pickup_lon = factory.LazyAttribute(lambda _: 6.1 + (fake.random.random() - 0.5) * 0.2)

    dropoff_location = factory.LazyAttribute(lambda _: fake.address().replace('\n', ', ')[:200])
    dropoff_lat = factory.LazyAttribute(lambda _: 46.2 + (fake.random.random() - 0.5) * 0.2)
    dropoff_lon = factory.LazyAttribute(lambda _: 6.1 + (fake.random.random() - 0.5) * 0.2)

    booking_type = "standard"
    scheduled_time = factory.LazyFunction(lambda: datetime.utcnow() + timedelta(hours=2))

    amount = factory.LazyAttribute(lambda _: round(fake.random.uniform(50.0, 200.0), 2))
    status = BookingStatus.PENDING

    client = factory.SubFactory(ClientFactory)
    company = factory.SubFactory(CompanyFactory)
    user_id = factory.LazyAttribute(lambda obj: obj.client.user.id if obj.client else None)

    duration_seconds = factory.LazyAttribute(lambda _: fake.random_int(min=600, max=3600))
    distance_meters = factory.LazyAttribute(lambda _: fake.random_int(min=1000, max=50000))

    wheelchair_client_has = False
    wheelchair_need = False
    is_round_trip = False

    created_at = factory.LazyFunction(datetime.utcnow)


# ========== ASSIGNMENT & DISPATCH ==========

class AssignmentFactory(SQLAlchemyModelFactory):
    """Factory pour Assignment."""

    class Meta:
        model = Assignment

    booking = factory.SubFactory(BookingFactory)
    driver = factory.SubFactory(DriverFactory)

    status = AssignmentStatus.SCHEDULED

    planned_pickup_at = factory.LazyAttribute(
        lambda obj: obj.booking.scheduled_time if obj.booking else datetime.utcnow() + timedelta(hours=1)
    )
    planned_dropoff_at = factory.LazyAttribute(
        lambda obj: obj.planned_pickup_at + timedelta(minutes=30) if obj.planned_pickup_at else None
    )

    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)


class DispatchRunFactory(SQLAlchemyModelFactory):
    """Factory pour DispatchRun."""

    class Meta:
        model = DispatchRun

    company = factory.SubFactory(CompanyFactory)
    day = factory.LazyFunction(date.today)

    status = DispatchStatus.PENDING

    started_at = None
    completed_at = None
    created_at = factory.LazyFunction(datetime.utcnow)


# ========== INVOICE & PAYMENT ==========

class InvoiceFactory(SQLAlchemyModelFactory):
    """Factory pour Invoice."""

    class Meta:
        model = Invoice

    company = factory.SubFactory(CompanyFactory)
    client = factory.SubFactory(ClientFactory)

    invoice_number = factory.Sequence(lambda n: f"INV-{date.today().year}-{n:05d}")
    invoice_date = factory.LazyFunction(date.today)
    due_date = factory.LazyAttribute(lambda obj: obj.invoice_date + timedelta(days=30))

    total_amount = factory.LazyAttribute(lambda _: round(fake.random.uniform(100.0, 1000.0), 2))
    tax_amount = factory.LazyAttribute(lambda obj: round(obj.total_amount * 0.077, 2))  # TVA 7.7%

    status = "pending"
    currency = "CHF"

    created_at = factory.LazyFunction(datetime.utcnow)


# ========== ML MODELS ==========

class MLPredictionFactory(SQLAlchemyModelFactory):
    """Factory pour MLPrediction."""

    class Meta:
        model = MLPrediction

    booking = factory.SubFactory(BookingFactory)
    driver = factory.SubFactory(DriverFactory)
    request_id = factory.LazyAttribute(lambda _: f"req_{fake.uuid4()[:8]}")

    predicted_delay_minutes = factory.LazyAttribute(lambda _: round(fake.random.uniform(0.0, 15.0), 2))
    confidence = factory.LazyAttribute(lambda _: round(fake.random.uniform(0.5, 0.95), 3))
    risk_level = fuzzy.FuzzyChoice(['low', 'medium', 'high'])
    contributing_factors = factory.LazyAttribute(
        lambda _: '{"distance_x_weather": 0.42, "traffic_x_weather": 0.35}'
    )

    model_version = "v1.0"
    prediction_time_ms = factory.LazyAttribute(lambda _: round(fake.random.uniform(50.0, 500.0), 1))
    feature_flag_enabled = True
    traffic_percentage = 10

    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)


class ABTestResultFactory(SQLAlchemyModelFactory):
    """Factory pour ABTestResult."""

    class Meta:
        model = ABTestResult

    booking = factory.SubFactory(BookingFactory)
    driver = factory.SubFactory(DriverFactory)
    test_timestamp = factory.LazyFunction(datetime.utcnow)

    ml_delay_minutes = factory.LazyAttribute(lambda _: round(fake.random.uniform(3.0, 10.0), 2))
    ml_confidence = factory.LazyAttribute(lambda _: round(fake.random.uniform(0.5, 0.9), 3))
    ml_risk_level = fuzzy.FuzzyChoice(['low', 'medium', 'high'])
    ml_prediction_time_ms = factory.LazyAttribute(lambda _: round(fake.random.uniform(100.0, 900.0), 1))
    ml_weather_factor = factory.LazyAttribute(lambda _: round(fake.random.uniform(0.0, 1.0), 2))

    heuristic_delay_minutes = factory.LazyAttribute(lambda _: round(fake.random.uniform(5.0, 12.0), 2))
    heuristic_prediction_time_ms = factory.LazyAttribute(lambda _: round(fake.random.uniform(0.0, 1.0), 1))

    difference_minutes = factory.LazyAttribute(
        lambda obj: abs(obj.ml_delay_minutes - obj.heuristic_delay_minutes)
    )
    ml_faster = factory.LazyAttribute(lambda obj: obj.ml_prediction_time_ms < obj.heuristic_prediction_time_ms)
    speed_advantage_ms = factory.LazyAttribute(
        lambda obj: obj.heuristic_prediction_time_ms - obj.ml_prediction_time_ms
    )

    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)


# ========== HELPERS POUR TESTS ==========

def create_booking_with_coordinates(
    company: Any | None = None,
    pickup_lat: float = 46.2044,
    pickup_lon: float = 6.1432,
    dropoff_lat: float = 46.2100,
    dropoff_lon: float = 6.1500,
    **kwargs: Any
) -> Booking:
    """Crée un booking avec coordonnées spécifiques."""
    if company is None:
        company = CompanyFactory()

    client = ClientFactory(company=company)
    return BookingFactory(
        company=company,
        client=client,
        user_id=client.user.id,
        pickup_lat=pickup_lat,
        pickup_lon=pickup_lon,
        dropoff_lat=dropoff_lat,
        dropoff_lon=dropoff_lon,
        **kwargs
    )


def create_driver_with_position(
    company: Any | None = None,
    latitude: float = 46.2044,
    longitude: float = 6.1432,
    is_available: bool = True,
    **kwargs: Any
) -> Driver:
    """Crée un driver avec position spécifique."""
    if company is None:
        company = CompanyFactory()

    return DriverFactory(
        company=company,
        user=UserFactory(role='driver'),
        latitude=latitude,
        longitude=longitude,
        is_available=is_available,
        **kwargs
    )


def create_assignment_with_booking_driver(
    booking: Any | None = None,
    driver: Any | None = None,
    company: Any | None = None,
    **kwargs: Any
) -> Assignment:
    """Crée un assignment avec booking et driver spécifiques."""
    if company is None:
        company = CompanyFactory()

    if booking is None:
        booking = create_booking_with_coordinates(company=company)

    if driver is None:
        driver = create_driver_with_position(company=company)

    return AssignmentFactory(
        booking=booking,
        driver=driver,
        **kwargs
    )


def create_dispatch_scenario(
    company: Any | None = None,
    num_bookings: int = 5,
    num_drivers: int = 3,
    dispatch_day: date | None = None
) -> dict[str, Any]:
    """
    Crée un scénario de dispatch complet pour tests.
    Returns:
        dict avec company, bookings, drivers, dispatch_run
    """
    if company is None:
        company = CompanyFactory()

    if dispatch_day is None:
        dispatch_day = date.today()

    # Créer drivers
    drivers = [
        create_driver_with_position(company=company, is_available=True)
        for _ in range(num_drivers)
    ]

    # Créer bookings
    bookings = [
        create_booking_with_coordinates(
            company=company,
            scheduled_time=datetime.combine(dispatch_day, datetime.min.time()) + timedelta(hours=8 + i),
            status=BookingStatus.PENDING
        )
        for i in range(num_bookings)
    ]

    # Créer dispatch run
    dispatch_run = DispatchRunFactory(
        company=company,
        day=dispatch_day,
        status=DispatchStatus.PENDING
    )

    db.session.commit()

    return {
        "company": company,
        "drivers": drivers,
        "bookings": bookings,
        "dispatch_run": dispatch_run,
    }

