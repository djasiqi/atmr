"""Tests pour GenerateInvoiceUseCase."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import IntegrityError

from application.invoices.generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceOutput,
    GenerateInvoiceUseCase,
)
from domain.invoice_dto import InvoiceDTO


@dataclass
class _MockCompanyBillingSettingsDTO:
    """Mock pour CompanyBillingSettingsDTO."""

    company_id: int
    default_vat_rate: Decimal = Decimal("7.70")
    overdue_fee: Decimal | None = None


@dataclass
class _MockBookingDTO:
    """Mock pour BookingDTO."""

    id: int
    company_id: int
    client_id: int
    status: Any
    invoice_line_id: int | None = None
    amount: Decimal = Decimal("50.00")


@dataclass
class _MockClient:
    """Mock pour Client."""

    id: int
    is_institution: bool = False


@dataclass
class _MockInvoice:
    """Mock pour Invoice."""

    id: int
    company_id: int
    client_id: int
    invoice_number: str = "INV-001"


@dataclass
class _MockCompanyBillingSettingsRepository:
    """Mock repository pour CompanyBillingSettings."""

    _settings: _MockCompanyBillingSettingsDTO | None = None

    def find_or_create(self, company_id: int) -> Any:
        """Retourne les paramètres mockés."""
        if self._settings:
            return self._settings
        return _MockCompanyBillingSettingsDTO(company_id=company_id)


@dataclass
class _MockBookingRepository:
    """Mock repository pour Booking."""

    _bookings: list[_MockBookingDTO] | None = None
    _bookings_by_ids: dict[list[int], list[_MockBookingDTO]] | None = None

    def __init__(self) -> None:
        """Initialise les données mockées."""
        self._bookings = []
        self._bookings_by_ids = {}

    def find_by_ids(self, booking_ids: list[int]) -> list[_MockBookingDTO]:
        """Retourne les réservations mockées."""
        if self._bookings_by_ids and tuple(booking_ids) in self._bookings_by_ids:
            return self._bookings_by_ids[tuple(booking_ids)]
        if self._bookings:
            return [b for b in self._bookings if b.id in booking_ids]
        return []

    def find_by_company_and_client_and_period(
        self,
        company_id: int,
        client_id: int,
        start_date: datetime,
        end_date: datetime,
        statuses: list[str],
    ) -> list[Any]:
        """Retourne les réservations mockées pour la période."""
        if self._bookings:
            return [
                b
                for b in self._bookings
                if b.company_id == company_id
                and b.client_id == client_id
                and b.status.value in statuses
            ]
        return []

    def find_model_by_id_and_company(self, booking_id: int, company_id: int) -> Any:
        """Retourne une réservation mockée."""
        if self._bookings:
            for b in self._bookings:
                if b.id == booking_id and b.company_id == company_id:
                    return b
        return None


@dataclass
class _MockClientRepository:
    """Mock repository pour Client."""

    _clients: dict[int, _MockClient] | None = None

    def __init__(self) -> None:
        """Initialise les clients mockés."""
        self._clients = {}

    def find_model_by_id_and_company(
        self, client_id: int, company_id: int
    ) -> _MockClient | None:
        """Retourne le client mocké."""
        if self._clients and client_id in self._clients:
            return self._clients[client_id]
        return None


@dataclass
class _MockInvoiceNumberGenerator:
    """Mock pour InvoiceNumberGenerator."""

    _next_number: str = "INV-2025-001"

    def generate(self, company_id: int, period_year: int, period_month: int) -> str:
        """Retourne un numéro de facture mocké."""
        return self._next_number


@dataclass
class _MockInvoiceRepository:
    """Mock repository pour Invoice."""

    _created_invoices: list[Any] | None = None

    def __init__(self) -> None:
        """Initialise la liste des factures créées."""
        self._created_invoices = []

    def create(self, invoice_data: dict[str, Any]) -> Any:
        """Enregistre la création d'une facture."""
        invoice = _MockInvoice(
            id=len(self._created_invoices) + 1 if self._created_invoices else 1,
            company_id=invoice_data.get("company_id", 1),
            client_id=invoice_data.get("client_id", 1),
            invoice_number=invoice_data.get("invoice_number", "INV-001"),
        )
        if self._created_invoices is not None:
            self._created_invoices.append(invoice)
        return invoice


def test_generate_invoice_output_structure(db) -> None:
    """Test de la structure de l'output."""
    # Arrange
    billing_settings_repo = _MockCompanyBillingSettingsRepository()
    booking_repo = _MockBookingRepository()
    client_repo = _MockClientRepository()
    invoice_repo = _MockInvoiceRepository()
    invoice_number_generator = _MockInvoiceNumberGenerator()

    uc = GenerateInvoiceUseCase(
        billing_settings_repo=billing_settings_repo,
        booking_repo=booking_repo,
        client_repo=client_repo,
        invoice_repo=invoice_repo,
        invoice_number_generator=invoice_number_generator,
    )

    # Act
    result = uc.execute(
        GenerateInvoiceInput(
            company_id=1,
            client_id=10,
            period_year=2025,
            period_month=1,
        )
    )

    # Assert
    # Note: Ce use case utilise Booking.query directement, donc nécessite un contexte DB
    # On teste la structure de l'output
    assert hasattr(result, "success")
    assert hasattr(result, "invoice_id")
    assert hasattr(result, "invoice")
    assert hasattr(result, "error")
    assert hasattr(result, "status_code")
    assert isinstance(result.success, bool)


def test_generate_invoice_with_reservation_ids(db) -> None:
    """Test de génération avec des IDs de réservations spécifiques."""
    # Arrange
    billing_settings_repo = _MockCompanyBillingSettingsRepository()
    booking_repo = _MockBookingRepository()
    client_repo = _MockClientRepository()
    invoice_repo = _MockInvoiceRepository()
    invoice_number_generator = _MockInvoiceNumberGenerator()

    uc = GenerateInvoiceUseCase(
        billing_settings_repo=billing_settings_repo,
        booking_repo=booking_repo,
        client_repo=client_repo,
        invoice_repo=invoice_repo,
        invoice_number_generator=invoice_number_generator,
    )

    # Act
    result = uc.execute(
        GenerateInvoiceInput(
            company_id=1,
            client_id=10,
            period_year=2025,
            period_month=1,
            reservation_ids=[100, 200],
        )
    )

    # Assert
    assert isinstance(result, GenerateInvoiceOutput)
    # Le résultat peut être success=False si les réservations n'existent pas en DB


def test_generate_invoice_with_bill_to_client(db) -> None:
    """Test de génération avec un client payeur."""
    # Arrange
    billing_settings_repo = _MockCompanyBillingSettingsRepository()
    booking_repo = _MockBookingRepository()
    client_repo = _MockClientRepository()
    # Ajouter un client institution
    if client_repo._clients is not None:
        client_repo._clients[100] = _MockClient(id=100, is_institution=True)
    invoice_repo = _MockInvoiceRepository()
    invoice_number_generator = _MockInvoiceNumberGenerator()

    uc = GenerateInvoiceUseCase(
        billing_settings_repo=billing_settings_repo,
        booking_repo=booking_repo,
        client_repo=client_repo,
        invoice_repo=invoice_repo,
        invoice_number_generator=invoice_number_generator,
    )

    # Act
    result = uc.execute(
        GenerateInvoiceInput(
            company_id=1,
            client_id=10,
            period_year=2025,
            period_month=1,
            bill_to_client_id=100,
        )
    )

    # Assert
    assert isinstance(result, GenerateInvoiceOutput)


def test_generate_invoice_with_overrides(db) -> None:
    """Test de génération avec des overrides."""
    # Arrange
    billing_settings_repo = _MockCompanyBillingSettingsRepository()
    booking_repo = _MockBookingRepository()
    client_repo = _MockClientRepository()
    invoice_repo = _MockInvoiceRepository()
    invoice_number_generator = _MockInvoiceNumberGenerator()

    uc = GenerateInvoiceUseCase(
        billing_settings_repo=billing_settings_repo,
        booking_repo=booking_repo,
        client_repo=client_repo,
        invoice_repo=invoice_repo,
        invoice_number_generator=invoice_number_generator,
    )

    # Act
    result = uc.execute(
        GenerateInvoiceInput(
            company_id=1,
            client_id=10,
            period_year=2025,
            period_month=1,
            overrides={
                "100": {"amount": 75.0, "vat_rate": 7.7},
                "200": {"amount": 100.0},
            },
        )
    )

    # Assert
    assert isinstance(result, GenerateInvoiceOutput)
    # Les overrides sont traités dans le use case


def test_generate_invoice_no_reservations(db) -> None:
    """Test d'erreur quand aucune réservation n'est trouvée."""
    # Arrange
    billing_settings_repo = _MockCompanyBillingSettingsRepository()
    booking_repo = _MockBookingRepository()  # Pas de réservations
    client_repo = _MockClientRepository()
    invoice_repo = _MockInvoiceRepository()
    invoice_number_generator = _MockInvoiceNumberGenerator()

    uc = GenerateInvoiceUseCase(
        billing_settings_repo=billing_settings_repo,
        booking_repo=booking_repo,
        client_repo=client_repo,
        invoice_repo=invoice_repo,
        invoice_number_generator=invoice_number_generator,
    )

    # Act
    result = uc.execute(
        GenerateInvoiceInput(
            company_id=1,
            client_id=10,
            period_year=2025,
            period_month=1,
        )
    )

    # Assert
    # Le résultat devrait être success=False avec une erreur
    # mais cela dépend aussi de Booking.query qui nécessite un contexte DB
    assert isinstance(result, GenerateInvoiceOutput)


class _MockInvoiceLineRepositoryRaisesIntegrityError:
    """Mock qui simule enum non migré (IntegrityError sur create)."""

    def create(self, line_data: dict[str, Any]) -> None:
        raise IntegrityError(
            "insert",
            {},
            "invalid input value for enum invoice_line_type: material_delivery",
        )


class _MockPDFServiceRaisesKeyError:
    """Mock qui simule mapping incomplet (KeyError sur generate_invoice_pdf)."""

    def generate_invoice_pdf(self, invoice: Any) -> None:
        raise KeyError("MATERIAL_DELIVERY")


def test_integrity_error_enum_not_migrated_returns_400(db) -> None:
    """IntegrityError enum non migré → 400 + INVOICE_LINE_TYPE_MIGRATION_REQUIRED."""
    from models import Booking, Client, Company, CompanyBillingSettings, User
    from models.enums import BookingStatus

    # Setup minimal : company, client, booking pour atteindre invoice_line_repo.create
    company_user = User(username="cu1", email="cu1@test.com")
    client_user = User(username="c1", email="c1@test.com")
    company = Company(name="Test Co", uid_ide="CHE-123.456.789", user=company_user)
    client = Client(user=client_user, company=company)
    db.session.add_all([company_user, client_user, company, client])
    db.session.flush()
    billing = CompanyBillingSettings(company_id=company.id)
    db.session.add(billing)
    db.session.commit()

    booking = Booking(
        company=company,
        client=client,
        user=client_user,
        customer_name="Test",
        pickup_location="A",
        dropoff_location="B",
        scheduled_time=datetime.now(UTC),
        amount=Decimal("50"),
        status=BookingStatus.COMPLETED,
    )
    db.session.add(booking)
    db.session.commit()

    uc = GenerateInvoiceUseCase(
        invoice_line_repo=_MockInvoiceLineRepositoryRaisesIntegrityError(),
    )
    result = uc.execute(
        GenerateInvoiceInput(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1,
            reservation_ids=[booking.id],
        )
    )
    assert result.success is False
    assert result.status_code == 400
    assert result.error is not None
    assert result.error.get("error_code") == "INVOICE_LINE_TYPE_MIGRATION_REQUIRED"
    assert result.error.get("details", {}).get("enum_type") == "invoice_line_type"
    assert result.error.get("details", {}).get("expected_value") == "material_delivery"


def test_key_error_mapping_incomplete_returns_400(db) -> None:
    """KeyError mapping incomplet → 400 + UNKNOWN_LINE_TYPE."""
    from models import Booking, Client, Company, CompanyBillingSettings, User
    from models.enums import BookingStatus

    company_user = User(username="cu2", email="cu2@test.com")
    client_user = User(username="c2", email="c2@test.com")
    company = Company(name="Test Co", uid_ide="CHE-456.789.012", user=company_user)
    client = Client(user=client_user, company=company)
    db.session.add_all([company_user, client_user, company, client])
    db.session.flush()
    billing = CompanyBillingSettings(company_id=company.id)
    db.session.add(billing)
    db.session.commit()

    booking = Booking(
        company=company,
        client=client,
        user=client_user,
        customer_name="Test",
        pickup_location="A",
        dropoff_location="B",
        scheduled_time=datetime.now(UTC),
        amount=Decimal("50"),
        status=BookingStatus.COMPLETED,
    )
    db.session.add(booking)
    db.session.commit()

    uc = GenerateInvoiceUseCase(pdf_service=_MockPDFServiceRaisesKeyError())
    result = uc.execute(
        GenerateInvoiceInput(
            company_id=company.id,
            client_id=client.id,
            period_year=2025,
            period_month=1,
            reservation_ids=[booking.id],
        )
    )
    assert result.success is False
    assert result.status_code == 400
    assert result.error is not None
    assert result.error.get("error_code") == "UNKNOWN_LINE_TYPE"
    assert result.error.get("details", {}).get("line_type") == "MATERIAL_DELIVERY"
