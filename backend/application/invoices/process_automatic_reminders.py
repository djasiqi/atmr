"""Use-case: traiter les rappels automatiques.

Ce use case migre la logique métier de InvoiceService.process_automatic_reminders()
vers l'architecture DDD.
"""

from __future__ import annotations  # noqa: I001

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_
from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from ext import db
from models import Invoice, InvoiceStatus
from application.invoices.generate_invoice_reminder import (
    GenerateInvoiceReminderUseCase,
    GenerateInvoiceReminderInput,
)
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_repository import InvoiceRepository

logger = logging.getLogger(__name__)

REMINDER_LEVEL_ONE = 1
REMINDER_LEVEL_THRESHOLD = 2
REMINDER_LEVEL_THREE = 3


@dataclass(frozen=True, slots=True)
class ProcessAutomaticRemindersInput:
    """Input pour traiter les rappels automatiques.

    Attributes:
        company_id: ID de l'entreprise (optionnel, si None traite toutes
            les entreprises)
    """

    company_id: int | None = None


@dataclass(frozen=True, slots=True)
class ProcessAutomaticRemindersOutput:
    """Output pour traiter les rappels automatiques.

    Attributes:
        success: True si l'opération a réussi
        reminders_generated: Nombre de rappels générés
        errors: Liste des erreurs par facture
        error: Dictionnaire d'erreurs globales (si échec total)
        status_code: Code HTTP (si échec total)
    """

    success: bool
    reminders_generated: int = 0
    errors: list[dict[str, Any]] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class ProcessAutomaticRemindersUseCase:
    """Use-case Application: traiter les rappels automatiques.

    Ce use case migre la logique métier de InvoiceService.process_automatic_reminders()
    vers l'architecture DDD.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        generate_reminder_use_case: GenerateInvoiceReminderUseCase | None = None,
        invoice_repo: InvoiceRepository | None = None,
        billing_settings_repo: CompanyBillingSettingsRepository | None = None,
    ):
        """Initialise le use case avec injection de dépendances.

        Args:
            generate_reminder_use_case: Use case pour générer un rappel
            invoice_repo: Repository pour Invoice
            billing_settings_repo: Repository pour CompanyBillingSettings
        """
        self.generate_reminder_use_case = (
            generate_reminder_use_case or GenerateInvoiceReminderUseCase()
        )
        self.invoice_repo = invoice_repo or InvoiceRepository()
        self.billing_settings_repo = (
            billing_settings_repo or CompanyBillingSettingsRepository()
        )

    def execute(
        self, input_data: ProcessAutomaticRemindersInput
    ) -> ProcessAutomaticRemindersOutput:
        """Traite les rappels automatiques pour les factures en retard.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            ProcessAutomaticRemindersOutput avec le nombre de rappels générés
        """
        try:
            now = datetime.now(UTC)
            reminders_generated = 0
            errors: list[dict[str, Any]] = []

            # Récupérer les paramètres de facturation pour chaque entreprise
            # ou pour l'entreprise spécifiée
            if input_data.company_id:
                companies_to_process = [input_data.company_id]
            else:
                # Récupérer toutes les entreprises avec des factures en retard
                companies_with_overdue = (
                    db.session.query(Invoice.company_id)
                    .filter(Invoice.status == InvoiceStatus.OVERDUE)
                    .distinct()
                    .all()
                )
                companies_to_process = [c[0] for c in companies_with_overdue]

            for company_id in companies_to_process:
                try:
                    # Vérifier si les rappels automatiques sont activés
                    from models import CompanyBillingSettings

                    billing_settings = CompanyBillingSettings.query.filter_by(
                        company_id=company_id
                    ).first()
                    if (
                        not billing_settings
                        or not billing_settings.auto_reminders_enabled
                    ):
                        continue

                    # Récupérer le planning des rappels
                    reminder_schedule = billing_settings.reminder_schedule_days or {}
                    schedule_days_1 = reminder_schedule.get("1", 10)
                    schedule_days_2 = reminder_schedule.get("2", 5)
                    schedule_days_3 = reminder_schedule.get("3", 5)

                    # Traiter les rappels niveau 1
                    cutoff_date_1 = now - timedelta(days=schedule_days_1)
                    invoices_level_1 = (
                        Invoice.query.filter(
                            and_(
                                Invoice.company_id == company_id,
                                Invoice.status == InvoiceStatus.OVERDUE,
                                Invoice.reminder_level == 0,
                                Invoice.due_date <= cutoff_date_1,
                            )
                        )
                        .filter(
                            (Invoice.last_reminder_at.is_(None))
                            | (Invoice.last_reminder_at <= cutoff_date_1)
                        )
                        .all()
                    )

                    for invoice in invoices_level_1:
                        try:
                            reminder_input = GenerateInvoiceReminderInput(
                                invoice_id=invoice.id, level=REMINDER_LEVEL_ONE
                            )
                            reminder_result = self.generate_reminder_use_case.execute(
                                reminder_input
                            )
                            if reminder_result.success:
                                reminders_generated += 1
                            else:
                                errors.append(
                                    {
                                        "invoice_id": invoice.id,
                                        "error": reminder_result.error
                                        or {"error": "Erreur inconnue"},
                                    }
                                )
                        except Exception as e:
                            logger.error(
                                "Erreur lors de la génération du rappel "
                                "niveau 1 pour facture %s: %s",
                                invoice.id,
                                e,
                            )
                            errors.append({"invoice_id": invoice.id, "error": str(e)})
                            continue

                    # Traiter les rappels niveau 2
                    cutoff_date_2 = now - timedelta(days=schedule_days_2)
                    invoices_level_2 = Invoice.query.filter(
                        and_(
                            Invoice.company_id == company_id,
                            Invoice.status == InvoiceStatus.OVERDUE,
                            Invoice.reminder_level == REMINDER_LEVEL_ONE,
                            Invoice.last_reminder_at <= cutoff_date_2,
                        )
                    ).all()

                    for invoice in invoices_level_2:
                        try:
                            reminder_input = GenerateInvoiceReminderInput(
                                invoice_id=invoice.id, level=REMINDER_LEVEL_THRESHOLD
                            )
                            reminder_result = self.generate_reminder_use_case.execute(
                                reminder_input
                            )
                            if reminder_result.success:
                                reminders_generated += 1
                            else:
                                errors.append(
                                    {
                                        "invoice_id": invoice.id,
                                        "error": reminder_result.error
                                        or {"error": "Erreur inconnue"},
                                    }
                                )
                        except Exception as e:
                            logger.error(
                                "Erreur lors de la génération du rappel "
                                "niveau 2 pour facture %s: %s",
                                invoice.id,
                                e,
                            )
                            errors.append({"invoice_id": invoice.id, "error": str(e)})
                            continue

                    # Traiter les rappels niveau 3
                    cutoff_date_3 = now - timedelta(days=schedule_days_3)
                    invoices_level_3 = Invoice.query.filter(
                        and_(
                            Invoice.company_id == company_id,
                            Invoice.status == InvoiceStatus.OVERDUE,
                            Invoice.reminder_level == REMINDER_LEVEL_THRESHOLD,
                            Invoice.last_reminder_at <= cutoff_date_3,
                        )
                    ).all()

                    for invoice in invoices_level_3:
                        try:
                            reminder_input = GenerateInvoiceReminderInput(
                                invoice_id=invoice.id, level=REMINDER_LEVEL_THREE
                            )
                            reminder_result = self.generate_reminder_use_case.execute(
                                reminder_input
                            )
                            if reminder_result.success:
                                reminders_generated += 1
                            else:
                                errors.append(
                                    {
                                        "invoice_id": invoice.id,
                                        "error": reminder_result.error
                                        or {"error": "Erreur inconnue"},
                                    }
                                )
                        except Exception as e:
                            logger.error(
                                "Erreur lors de la génération du rappel "
                                "niveau 3 pour facture %s: %s",
                                invoice.id,
                                e,
                            )
                            errors.append({"invoice_id": invoice.id, "error": str(e)})
                            continue

                except Exception as e:
                    logger.error(
                        "Erreur lors du traitement des rappels pour "
                        "l'entreprise %s: %s",
                        company_id,
                        e,
                    )
                    errors.append({"company_id": company_id, "error": str(e)})
                    continue

            logger.info(
                "Traitement des rappels automatiques terminé: %s rappels "
                "générés, %s erreurs",
                reminders_generated,
                len(errors),
            )

            return ProcessAutomaticRemindersOutput(
                success=True,
                reminders_generated=reminders_generated,
                errors=errors if errors else None,
            )

        except (OperationalError, DBAPIError, IntegrityError) as e:
            db.session.rollback()
            logger.error(
                "Erreur DB lors du traitement des rappels automatiques "
                "(DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return ProcessAutomaticRemindersOutput(
                success=False,
                error={"error": "Erreur de base de données"},
                status_code=500,
            )
        except Exception:
            db.session.rollback()
            logger.exception(
                "Erreur inattendue lors du traitement des rappels automatiques"
            )
            return ProcessAutomaticRemindersOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )
