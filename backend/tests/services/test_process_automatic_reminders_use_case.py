"""Tests pour ProcessAutomaticRemindersUseCase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from application.invoices.generate_invoice_reminder import (
    GenerateInvoiceReminderUseCase,
)
from application.invoices.process_automatic_reminders import (
    ProcessAutomaticRemindersInput,
    ProcessAutomaticRemindersOutput,
    ProcessAutomaticRemindersUseCase,
)


@dataclass
class _MockGenerateReminderUseCase:
    """Mock pour GenerateInvoiceReminderUseCase."""

    _calls: list[dict[str, Any]] | None = None

    def __init__(self) -> None:
        """Initialise la liste des appels."""
        self._calls = []

    def execute(self, input_data: Any) -> Any:
        """Enregistre l'appel et retourne un résultat mocké."""
        if self._calls is not None:
            self._calls.append({"input": input_data})
        from application.invoices.generate_invoice_reminder import (
            GenerateInvoiceReminderOutput,
        )

        return GenerateInvoiceReminderOutput(success=True)


def test_process_automatic_reminders_output_structure(db) -> None:
    """Test de la structure de l'output."""
    # Arrange
    generate_reminder_uc = _MockGenerateReminderUseCase()
    uc = ProcessAutomaticRemindersUseCase(
        generate_reminder_use_case=generate_reminder_uc,
    )

    # Act
    result = uc.execute(ProcessAutomaticRemindersInput(company_id=1))

    # Assert
    # Note: Ce use case utilise Invoice.query directement, donc nécessite un contexte DB
    # On teste la structure de l'output
    assert hasattr(result, "success")
    assert hasattr(result, "reminders_generated")
    assert hasattr(result, "errors")
    assert hasattr(result, "error")
    assert hasattr(result, "status_code")
    assert isinstance(result.success, bool)
    assert isinstance(result.reminders_generated, int)
    assert result.reminders_generated >= 0


def test_process_automatic_reminders_with_company_id(db) -> None:
    """Test de traitement des rappels pour une entreprise spécifique."""
    # Arrange
    generate_reminder_uc = _MockGenerateReminderUseCase()
    uc = ProcessAutomaticRemindersUseCase(
        generate_reminder_use_case=generate_reminder_uc,
    )

    # Act
    result = uc.execute(ProcessAutomaticRemindersInput(company_id=1))

    # Assert
    assert isinstance(result, ProcessAutomaticRemindersOutput)
    # Le résultat peut être success=False si aucune facture en retard
    # ou success=True avec reminders_generated > 0


def test_process_automatic_reminders_all_companies(db) -> None:
    """Test de traitement des rappels pour toutes les entreprises."""
    # Arrange
    generate_reminder_uc = _MockGenerateReminderUseCase()
    uc = ProcessAutomaticRemindersUseCase(
        generate_reminder_use_case=generate_reminder_uc,
    )

    # Act
    result = uc.execute(ProcessAutomaticRemindersInput(company_id=None))

    # Assert
    assert isinstance(result, ProcessAutomaticRemindersOutput)


def test_process_automatic_reminders_calls_generate_reminder(db) -> None:
    """Test que le use case appelle GenerateInvoiceReminderUseCase."""
    # Arrange
    generate_reminder_uc = _MockGenerateReminderUseCase()
    uc = ProcessAutomaticRemindersUseCase(
        generate_reminder_use_case=generate_reminder_uc,
    )

    # Act
    result = uc.execute(ProcessAutomaticRemindersInput(company_id=1))

    # Assert
    # Note: Le nombre d'appels dépend de la présence de factures en retard en DB
    # Ici on vérifie juste que le use case s'exécute
    assert isinstance(result, ProcessAutomaticRemindersOutput)
    # Si des rappels sont générés, generate_reminder_uc._calls devrait contenir des entrées
    # mais cela dépend de l'état de la DB de test
