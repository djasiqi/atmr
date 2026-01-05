"""Tests pour GetInvoiceUseCase."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from application.invoices.get_invoice import (
    GetInvoiceInput,
    GetInvoiceOutput,
    GetInvoiceUseCase,
)
from domain.invoice_dto import InvoiceDTO, InvoiceLineDTO
from models.enums import InvoiceStatus


@dataclass
class _MockInvoiceRepository:
    """Mock repository pour Invoice."""

    _invoice: InvoiceDTO | None = None

    def find_by_id_with_lines(
        self, invoice_id: int, company_id: int
    ) -> InvoiceDTO | None:
        """Retourne la facture mockée si elle correspond."""
        if (
            self._invoice
            and self._invoice.id == invoice_id
            and self._invoice.company_id == company_id
        ):
            return self._invoice
        return None


def test_get_invoice_success() -> None:
    """Test de récupération réussie d'une facture."""
    # Arrange
    invoice_dto = InvoiceDTO(
        id=1,
        company_id=10,
        client_id=20,
        invoice_number="INV-2025-001",
        status=InvoiceStatus.DRAFT,
        lines=[
            InvoiceLineDTO(
                id=1,
                invoice_id=1,
                line_type="booking",
                description="Course 1",
                quantity=Decimal("1.00"),
                unit_price=Decimal("50.00"),
                line_total=Decimal("50.00"),
            )
        ],
    )
    mock_repo = _MockInvoiceRepository(_invoice=invoice_dto)
    uc = GetInvoiceUseCase(invoice_repo=mock_repo)

    # Act
    result = uc.execute(GetInvoiceInput(invoice_id=1, company_id=10))

    # Assert
    assert result.found is True
    assert result.invoice is not None
    assert result.invoice.id == 1
    assert result.invoice.company_id == 10
    assert result.invoice.invoice_number == "INV-2025-001"
    assert result.invoice.status == InvoiceStatus.DRAFT
    assert result.invoice.lines is not None
    assert len(result.invoice.lines) == 1
    assert result.error is None
    assert result.status_code is None


def test_get_invoice_not_found() -> None:
    """Test de récupération d'une facture inexistante."""
    # Arrange
    mock_repo = _MockInvoiceRepository(_invoice=None)
    uc = GetInvoiceUseCase(invoice_repo=mock_repo)

    # Act
    result = uc.execute(GetInvoiceInput(invoice_id=999, company_id=10))

    # Assert
    assert result.found is False
    assert result.invoice is None
    assert result.error is not None
    assert result.error["error"] == "Facture non trouvée"
    assert result.status_code == 404


def test_get_invoice_wrong_company() -> None:
    """Test de récupération d'une facture d'une autre entreprise."""
    # Arrange
    invoice_dto = InvoiceDTO(
        id=1,
        company_id=10,
        client_id=20,
        invoice_number="INV-2025-001",
        status=InvoiceStatus.DRAFT,
    )
    mock_repo = _MockInvoiceRepository(_invoice=invoice_dto)
    uc = GetInvoiceUseCase(invoice_repo=mock_repo)

    # Act
    result = uc.execute(GetInvoiceInput(invoice_id=1, company_id=99))

    # Assert
    assert result.found is False
    assert result.invoice is None
    assert result.error is not None
    assert result.error["error"] == "Facture non trouvée"
    assert result.status_code == 404


def test_get_invoice_with_lines() -> None:
    """Test de récupération d'une facture avec ses lignes."""
    # Arrange
    invoice_dto = InvoiceDTO(
        id=2,
        company_id=10,
        client_id=20,
        invoice_number="INV-2025-002",
        status=InvoiceStatus.SENT,
        lines=[
            InvoiceLineDTO(
                id=1,
                invoice_id=2,
                line_type="booking",
                description="Course 1",
                quantity=Decimal("1.00"),
                unit_price=Decimal("50.00"),
                line_total=Decimal("50.00"),
            ),
            InvoiceLineDTO(
                id=2,
                invoice_id=2,
                line_type="booking",
                description="Course 2",
                quantity=Decimal("1.00"),
                unit_price=Decimal("75.00"),
                line_total=Decimal("75.00"),
            ),
        ],
    )
    mock_repo = _MockInvoiceRepository(_invoice=invoice_dto)
    uc = GetInvoiceUseCase(invoice_repo=mock_repo)

    # Act
    result = uc.execute(GetInvoiceInput(invoice_id=2, company_id=10))

    # Assert
    assert result.found is True
    assert result.invoice is not None
    assert result.invoice.lines is not None
    assert len(result.invoice.lines) == 2
    assert result.invoice.lines[0].description == "Course 1"
    assert result.invoice.lines[1].description == "Course 2"
