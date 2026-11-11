"""Tests pour DuplicateInvoiceUseCase."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from application.invoices.cancel_invoice import (
    CancelInvoiceOutput,
    CancelInvoiceUseCase,
)
from application.invoices.duplicate_invoice import (
    DuplicateInvoiceInput,
    DuplicateInvoiceOutput,
    DuplicateInvoiceUseCase,
)
from models.enums import InvoiceStatus


@dataclass
class _MockInvoiceLine:
    """Mock pour une ligne de facture."""

    id: int
    reservation_id: int | None = None
    line_total: Decimal = Decimal("0.00")
    vat_rate: Decimal | None = None
    adjustment_note: str | None = None


@dataclass
class _MockClient:
    """Mock pour un client."""

    id: int
    user: Any | None = None


@dataclass
class _MockUser:
    """Mock pour un utilisateur."""

    first_name: str | None = None
    last_name: str | None = None
    username: str | None = None


@dataclass
class _MockInvoice:
    """Mock pour une facture."""

    id: int
    status: InvoiceStatus
    client_id: int
    bill_to_client_id: int | None = None
    period_year: int = 2025
    period_month: int = 1
    lines: list[_MockInvoiceLine] | None = None
    client: _MockClient | None = None

    def __post_init__(self) -> None:
        """Initialise les lignes si None."""
        if self.lines is None:
            self.lines = []


@dataclass
class _MockCancelInvoiceUseCase:
    """Mock pour CancelInvoiceUseCase. Le duplicata ne doit pas l'appeler."""

    _should_succeed: bool = True
    calls: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def execute(self, input_data: Any) -> CancelInvoiceOutput:
        """Retourne un résultat mocké."""
        assert self.calls is not None
        self.calls.append(input_data)
        if self._should_succeed:
            return CancelInvoiceOutput(success=True)
        return CancelInvoiceOutput(
            success=False,
            error={"error": "Erreur d'annulation"},
            status_code=500,
        )


def test_duplicate_invoice_success() -> None:
    """Duplicata fidèle : brouillon de saisie, facture émise inchangée."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.SENT,
        client_id=10,
        bill_to_client_id=None,
        period_year=2025,
        period_month=1,
        lines=[
            _MockInvoiceLine(
                id=1,
                reservation_id=100,
                line_total=Decimal("50.00"),
                vat_rate=Decimal("7.70"),
            ),
            _MockInvoiceLine(
                id=2,
                reservation_id=200,
                line_total=Decimal("75.00"),
                vat_rate=None,
            ),
        ],
        client=_MockClient(
            id=10,
            user=_MockUser(first_name="John", last_name="Doe", username="jdoe"),
        ),
    )
    cancel_uc = _MockCancelInvoiceUseCase(_should_succeed=True)
    uc = DuplicateInvoiceUseCase(cancel_invoice_use_case=cancel_uc)

    # Act
    result = uc.execute(DuplicateInvoiceInput(invoice=invoice))

    # Assert
    assert result.success is True
    assert result.draft_context is not None
    assert result.draft_context["client_id"] == 10
    assert result.draft_context["period_year"] == 2025
    assert result.draft_context["period_month"] == 1
    assert result.draft_context["reservation_ids"] == [100, 200]
    assert result.draft_context["client"]["display_name"] == "John Doe"
    assert invoice.status == InvoiceStatus.SENT
    assert cancel_uc.calls == []


def test_duplicate_invoice_already_draft() -> None:
    """Test d'erreur si la facture est déjà un brouillon."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.DRAFT,
        client_id=10,
        lines=[_MockInvoiceLine(id=1, reservation_id=100)],
    )
    cancel_uc = _MockCancelInvoiceUseCase()
    uc = DuplicateInvoiceUseCase(cancel_invoice_use_case=cancel_uc)

    # Act
    result = uc.execute(DuplicateInvoiceInput(invoice=invoice))

    # Assert
    assert result.success is False
    assert result.draft_context is None
    assert result.error is not None
    assert "brouillon" in result.error["error"].lower()
    assert result.status_code == 400


def test_duplicate_invoice_no_reservations() -> None:
    """Test d'erreur si aucune course liée."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.SENT,
        client_id=10,
        lines=[
            _MockInvoiceLine(id=1, reservation_id=None),  # Pas de réservation
        ],
    )
    cancel_uc = _MockCancelInvoiceUseCase()
    uc = DuplicateInvoiceUseCase(cancel_invoice_use_case=cancel_uc)

    # Act
    result = uc.execute(DuplicateInvoiceInput(invoice=invoice))

    # Assert
    assert result.success is False
    assert result.status_code == 400
    assert result.error is not None
    assert "error_code" not in result.error
    assert "course" in result.error["error"].lower()
    assert cancel_uc.calls == []


def test_duplicate_invoice_does_not_cancel_issued() -> None:
    """Le duplicata n'annule pas l'original, même si l'annulation échouerait."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.SENT,
        client_id=10,
        lines=[_MockInvoiceLine(id=1, reservation_id=100)],
    )
    cancel_uc = _MockCancelInvoiceUseCase(_should_succeed=False)
    uc = DuplicateInvoiceUseCase(cancel_invoice_use_case=cancel_uc)

    # Act
    result = uc.execute(DuplicateInvoiceInput(invoice=invoice))

    # Assert
    assert result.success is True
    assert result.draft_context is not None
    assert result.draft_context["reservation_ids"] == [100]
    assert invoice.status == InvoiceStatus.SENT
    assert cancel_uc.calls == []


def test_duplicate_invoice_third_party_billing() -> None:
    """Test de duplication avec facturation tierce."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.SENT,
        client_id=10,
        bill_to_client_id=20,  # Facturation tierce
        period_year=2025,
        period_month=2,
        lines=[
            _MockInvoiceLine(id=1, reservation_id=100, line_total=Decimal("100.00"))
        ],
    )
    cancel_uc = _MockCancelInvoiceUseCase(_should_succeed=True)
    uc = DuplicateInvoiceUseCase(cancel_invoice_use_case=cancel_uc)

    # Act
    result = uc.execute(DuplicateInvoiceInput(invoice=invoice))

    # Assert
    assert result.success is True
    assert result.draft_context is not None
    assert result.draft_context["bill_to_client_id"] == 20
    assert result.draft_context["period_month"] == 2
    assert result.draft_context["reservation_ids"] == [100]
    assert invoice.status == InvoiceStatus.SENT
    assert cancel_uc.calls == []


def test_duplicate_invoice_builds_overrides() -> None:
    """Test que les overrides sont correctement construits."""
    # Arrange
    invoice = _MockInvoice(
        id=1,
        status=InvoiceStatus.SENT,
        client_id=10,
        lines=[
            _MockInvoiceLine(
                id=1,
                reservation_id=100,
                line_total=Decimal("50.00"),
                vat_rate=Decimal("7.70"),
                adjustment_note="Note spéciale",
            ),
        ],
    )
    cancel_uc = _MockCancelInvoiceUseCase(_should_succeed=True)
    uc = DuplicateInvoiceUseCase(cancel_invoice_use_case=cancel_uc)

    # Act
    result = uc.execute(DuplicateInvoiceInput(invoice=invoice))

    # Assert
    assert result.success is True
    assert result.draft_context is not None
    assert result.draft_context["overrides"]["100"] == {
        "line_total": "50.00",
        "vat_rate": "7.70",
        "adjustment_note": "Note spéciale",
    }
    assert invoice.status == InvoiceStatus.SENT
    assert cancel_uc.calls == []
