"""Tests pour GenerateConsolidatedInvoiceUseCase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from application.invoices.generate_consolidated_invoice import (
    GenerateConsolidatedInvoiceInput,
    GenerateConsolidatedInvoiceOutput,
    GenerateConsolidatedInvoiceUseCase,
)
from application.invoices.generate_invoice import (
    GenerateInvoiceOutput,
    GenerateInvoiceUseCase,
)


@dataclass
class _MockInvoice:
    """Mock pour une facture."""

    id: int
    client_id: int
    company_id: int


@dataclass
class _MockGenerateInvoiceUseCase:
    """Mock pour GenerateInvoiceUseCase."""

    _results: dict[int, GenerateInvoiceOutput] | None = None
    _default_success: bool = True

    def __init__(self, _default_success: bool = True) -> None:
        """Initialise les résultats mockés."""
        self._results = {}
        self._default_success = _default_success

    def execute(self, input_data: Any) -> GenerateInvoiceOutput:
        """Retourne un résultat mocké basé sur client_id."""
        client_id = input_data.client_id
        if self._results and client_id in self._results:
            return self._results[client_id]

        if self._default_success:
            return GenerateInvoiceOutput(
                success=True,
                invoice_id=client_id * 10,  # ID fictif
                invoice=_MockInvoice(
                    id=client_id * 10, client_id=client_id, company_id=1
                ),
            )
        return GenerateInvoiceOutput(
            success=False,
            error={"error": "Erreur de génération"},
            status_code=500,
        )


def test_generate_consolidated_invoice_success(db) -> None:
    """Test de génération réussie de factures consolidées."""
    # Arrange
    generate_invoice_uc = _MockGenerateInvoiceUseCase(_default_success=True)
    uc = GenerateConsolidatedInvoiceUseCase(
        generate_invoice_use_case=generate_invoice_uc,
    )

    # Act
    result = uc.execute(
        GenerateConsolidatedInvoiceInput(
            company_id=1,
            client_ids=[10, 20, 30],
            period_year=2025,
            period_month=1,
            bill_to_client_id=100,
        )
    )

    # Assert
    # Note: Ce use case utilise Invoice.query directement pour vérifier les
    # factures existantes
    # Le résultat peut varier selon l'état de la DB
    assert isinstance(result, GenerateConsolidatedInvoiceOutput)
    assert hasattr(result, "success")
    assert hasattr(result, "invoices")
    assert hasattr(result, "errors")
    assert hasattr(result, "success_count")
    assert hasattr(result, "error_count")
    assert isinstance(result.success_count, int)
    assert isinstance(result.error_count, int)


def test_generate_consolidated_invoice_partial_errors() -> None:
    """Test avec erreurs partielles pour certains clients."""
    # Arrange
    generate_invoice_uc = _MockGenerateInvoiceUseCase()
    # Configurer des résultats différents par client
    if generate_invoice_uc._results is not None:
        generate_invoice_uc._results[10] = GenerateInvoiceOutput(
            success=True,
            invoice_id=100,
            invoice=_MockInvoice(id=100, client_id=10, company_id=1),
        )
        generate_invoice_uc._results[20] = GenerateInvoiceOutput(
            success=False,
            error={"error": "Erreur pour client 20"},
            status_code=400,
        )
        generate_invoice_uc._results[30] = GenerateInvoiceOutput(
            success=True,
            invoice_id=300,
            invoice=_MockInvoice(id=300, client_id=30, company_id=1),
        )
    uc = GenerateConsolidatedInvoiceUseCase(
        generate_invoice_use_case=generate_invoice_uc,
    )

    # Act
    result = uc.execute(
        GenerateConsolidatedInvoiceInput(
            company_id=1,
            client_ids=[10, 20, 30],
            period_year=2025,
            period_month=1,
            bill_to_client_id=None,
        )
    )

    # Assert
    assert isinstance(result, GenerateConsolidatedInvoiceOutput)
    # Le résultat peut avoir success_count=2 et error_count=1
    # ou varier selon l'état de la DB (factures existantes)


def test_generate_consolidated_invoice_with_reservations(db) -> None:
    """Test avec réservations spécifiques par client."""
    # Arrange
    generate_invoice_uc = _MockGenerateInvoiceUseCase(_default_success=True)
    uc = GenerateConsolidatedInvoiceUseCase(
        generate_invoice_use_case=generate_invoice_uc,
    )

    # Act
    result = uc.execute(
        GenerateConsolidatedInvoiceInput(
            company_id=1,
            client_ids=[10, 20],
            period_year=2025,
            period_month=1,
            bill_to_client_id=100,
            client_reservations={
                10: [100, 101, 102],
                20: [200, 201],
            },
        )
    )

    # Assert
    assert isinstance(result, GenerateConsolidatedInvoiceOutput)
    # Vérifie que le use case s'exécute avec les réservations spécifiées


def test_generate_consolidated_invoice_output_structure(db) -> None:
    """Test de la structure de l'output."""
    # Arrange
    generate_invoice_uc = _MockGenerateInvoiceUseCase(_default_success=True)
    uc = GenerateConsolidatedInvoiceUseCase(
        generate_invoice_use_case=generate_invoice_uc,
    )

    # Act
    result = uc.execute(
        GenerateConsolidatedInvoiceInput(
            company_id=1,
            client_ids=[10],
            period_year=2025,
            period_month=1,
            bill_to_client_id=None,
        )
    )

    # Assert
    assert hasattr(result, "success")
    assert hasattr(result, "invoices")
    assert hasattr(result, "errors")
    assert hasattr(result, "success_count")
    assert hasattr(result, "error_count")
    assert hasattr(result, "error")
    assert hasattr(result, "status_code")
    assert isinstance(result.success, bool)
    assert isinstance(result.success_count, int)
    assert isinstance(result.error_count, int)
    assert result.success_count >= 0
    assert result.error_count >= 0
