"""Tests pour CancelInvoiceUseCase."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

from application.invoices.cancel_invoice import (
    CancelInvoiceInput,
    CancelInvoiceOutput,
    CancelInvoiceUseCase,
)
from models.enums import InvoiceStatus


@dataclass
class _MockBooking:
    """Mock pour une réservation."""

    id: int
    invoice_line_id: int | None = None


@dataclass
class _MockInvoiceLine:
    """Mock pour une ligne de facture."""

    id: int
    reservation_id: int | None = None


@dataclass
class _MockInvoice:
    """Mock pour une facture."""

    id: int
    status: InvoiceStatus | str
    cancelled_at: datetime | None = None
    updated_at: datetime | None = None
    balance_due: Decimal = Decimal("0.00")
    lines: list[_MockInvoiceLine] | None = None

    def __post_init__(self) -> None:
        """Initialise les lignes si None."""
        if self.lines is None:
            self.lines = []


@dataclass
class _MockBookingRepository:
    """Mock repository pour Booking."""

    _bookings: dict[int, _MockBooking] | None = None

    def find_by_id(self, booking_id: int) -> Any | None:
        """Retourne la réservation mockée si elle existe."""
        if self._bookings and booking_id in self._bookings:
            return self._bookings[booking_id]
        return None


@contextmanager
def _patch_booking_query_for_cancel(
    bookings: dict[int, _MockBooking],
) -> Generator[MagicMock, None, None]:
    """Évite un vrai accès DB pour Booking.query (PostgreSQL optionnel en CI locale)."""
    mock_cls = MagicMock()

    def _filter(*_args: Any, **_kwargs: Any) -> MagicMock:
        inner = MagicMock()
        inner.all.return_value = list(bookings.values())
        return inner

    mock_cls.query.filter.side_effect = _filter
    mock_cls.query.get.side_effect = lambda bid: (
        bookings.get(int(bid)) if bid is not None else None
    )

    # ``from models import Booking`` lit ``models.Booking`` à l'appel d'execute.
    with patch("models.Booking", mock_cls):
        yield mock_cls


def test_cancel_invoice_draft_success(db) -> None:
    """Test d'annulation réussie d'une facture en brouillon."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.DRAFT,
        balance_due=Decimal("100.00"),
        lines=[
            _MockInvoiceLine(id=1, reservation_id=10),
            _MockInvoiceLine(id=2, reservation_id=20),
        ],
    )
    bookings = {
        10: _MockBooking(id=10, invoice_line_id=1),
        20: _MockBooking(id=20, invoice_line_id=2),
    }
    booking_repo = _MockBookingRepository(_bookings=bookings)
    uc = CancelInvoiceUseCase(booking_repo=booking_repo)

    # Act
    with _patch_booking_query_for_cancel(bookings):
        result = uc.execute(CancelInvoiceInput(invoice=invoice, force=False))

    # Assert
    assert result.success is True
    assert result.error is None
    assert result.status_code is None
    assert invoice.status == InvoiceStatus.CANCELLED
    assert invoice.cancelled_at is not None
    assert invoice.balance_due == Decimal("0.00")
    assert bookings[10].invoice_line_id is None
    assert bookings[20].invoice_line_id is None


def test_cancel_invoice_already_cancelled() -> None:
    """Test d'annulation d'une facture déjà annulée."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.CANCELLED,
        cancelled_at=datetime.now(UTC),
        balance_due=Decimal("0.00"),
    )
    booking_repo = _MockBookingRepository()
    uc = CancelInvoiceUseCase(booking_repo=booking_repo)

    # Act
    result = uc.execute(CancelInvoiceInput(invoice=invoice, force=False))

    # Assert
    assert result.success is True
    assert result.error is None
    assert result.status_code is None


def test_cancel_invoice_invalid_status() -> None:
    """Test d'annulation d'une facture avec statut invalide (sans force)."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.SENT,
        balance_due=Decimal("100.00"),
    )
    booking_repo = _MockBookingRepository()
    uc = CancelInvoiceUseCase(booking_repo=booking_repo)

    # Act
    result = uc.execute(CancelInvoiceInput(invoice=invoice, force=False))

    # Assert
    assert result.success is False
    assert result.error is not None
    assert "draft" in result.error["error"].lower()
    assert result.status_code == 400


def test_cancel_invoice_force(db) -> None:
    """Test d'annulation forcée d'une facture avec n'importe quel statut."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.SENT,
        balance_due=Decimal("100.00"),
        lines=[_MockInvoiceLine(id=1, reservation_id=10)],
    )
    bookings = {10: _MockBooking(id=10, invoice_line_id=1)}
    booking_repo = _MockBookingRepository(_bookings=bookings)
    uc = CancelInvoiceUseCase(booking_repo=booking_repo)

    # Act
    with _patch_booking_query_for_cancel(bookings):
        result = uc.execute(CancelInvoiceInput(invoice=invoice, force=True))

    # Assert
    assert result.success is True
    assert result.error is None
    assert result.status_code is None
    assert bookings[10].invoice_line_id is None


def test_cancel_invoice_invalid_status_string() -> None:
    """Test d'annulation avec un statut invalide (string)."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status="INVALID_STATUS",
        balance_due=Decimal("100.00"),
    )
    booking_repo = _MockBookingRepository()
    uc = CancelInvoiceUseCase(booking_repo=booking_repo)

    # Act
    result = uc.execute(CancelInvoiceInput(invoice=invoice, force=False))

    # Assert
    assert result.success is False
    assert result.error is not None
    assert "invalide" in result.error["error"].lower()
    assert result.status_code == 400


def test_cancel_invoice_releases_bookings(db) -> None:
    """Test que l'annulation libère les réservations associées."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.DRAFT,
        balance_due=Decimal("100.00"),
        lines=[
            _MockInvoiceLine(id=1, reservation_id=10),
            _MockInvoiceLine(id=2, reservation_id=None),  # Pas de réservation
        ],
    )
    bookings = {
        10: _MockBooking(id=10, invoice_line_id=1),
    }
    booking_repo = _MockBookingRepository(_bookings=bookings)
    uc = CancelInvoiceUseCase(booking_repo=booking_repo)

    # Act
    with _patch_booking_query_for_cancel(bookings):
        result = uc.execute(CancelInvoiceInput(invoice=invoice, force=False))

    # Assert
    assert result.success is True
    assert bookings[10].invoice_line_id is None


def test_is_direct_client_invoice() -> None:
    """Test _is_direct_client_invoice : détection facture client directe."""
    from application.invoices.cancel_invoice import _is_direct_client_invoice

    class _Inv:
        billed_to_company_id = None
        bill_to_client_id = None
        billing_strategy = "s1_patient"

    assert _is_direct_client_invoice(_Inv()) is True

    class _InvClinic:
        billed_to_company_id = 1
        bill_to_client_id = None
        billing_strategy = "s1_patient"

    assert _is_direct_client_invoice(_InvClinic()) is False

    class _InvTierce:
        billed_to_company_id = None
        bill_to_client_id = 2
        billing_strategy = "s1_patient"

    assert _is_direct_client_invoice(_InvTierce()) is False
