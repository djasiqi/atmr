"""Use-case: générer un rappel de facture.

Ce use case migre la logique métier de InvoiceService.generate_reminder()
vers l'architecture DDD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.exc import DBAPIError, IntegrityError, OperationalError

from ext import db
from models import Invoice, InvoiceLineType, InvoiceReminder
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_line_repository import InvoiceLineRepository
from repositories.invoice_repository import InvoiceRepository
from services.documents.pdf import PDFService

logger = logging.getLogger(__name__)

LEVEL_ONE = 1
LEVEL_THRESHOLD = 2
LEVEL_THREE = 3
FEE_AMOUNT_ZERO = 0


@dataclass(frozen=True, slots=True)
class GenerateInvoiceReminderInput:
    """Input pour générer un rappel de facture.

    Attributes:
        invoice_id: ID de la facture
        level: Niveau du rappel (1, 2, 3)
    """

    invoice_id: int
    level: int


@dataclass(frozen=True, slots=True)
class GenerateInvoiceReminderOutput:
    """Output pour générer un rappel de facture.

    Attributes:
        success: True si l'opération a réussi
        reminder: Le rappel créé (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    success: bool
    reminder: InvoiceReminder | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class GenerateInvoiceReminderUseCase:
    """Use-case Application: générer un rappel de facture.

    Ce use case migre la logique métier de InvoiceService.generate_reminder()
    vers l'architecture DDD.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        invoice_repo: InvoiceRepository | None = None,
        invoice_line_repo: InvoiceLineRepository | None = None,
        billing_settings_repo: CompanyBillingSettingsRepository | None = None,
        pdf_service: PDFService | None = None,
    ):
        """Initialise le use case avec injection de dépendances.

        Args:
            invoice_repo: Repository pour Invoice
            invoice_line_repo: Repository pour InvoiceLine
            billing_settings_repo: Repository pour CompanyBillingSettings
            pdf_service: Service de génération PDF
        """
        self.invoice_repo = invoice_repo or InvoiceRepository()
        self.invoice_line_repo = invoice_line_repo or InvoiceLineRepository()
        self.billing_settings_repo = (
            billing_settings_repo or CompanyBillingSettingsRepository()
        )
        self.pdf_service = pdf_service or PDFService()

    def execute(
        self, input_data: GenerateInvoiceReminderInput
    ) -> GenerateInvoiceReminderOutput:
        """Génère un rappel pour une facture en retard.

        Args:
            input_data: Données d'entrée du use case

        Returns:
            GenerateInvoiceReminderOutput avec le rappel créé
        """
        try:
            # 1. Récupérer la facture
            invoice = Invoice.query.filter_by(id=input_data.invoice_id).first()
            if not invoice:
                msg = "Facture non trouvée"
                raise ValueError(msg)

            # 2. Vérifier que le rappel n'a pas déjà été généré
            if input_data.level <= invoice.reminder_level:
                msg = f"Le rappel niveau {input_data.level} a déjà été généré"
                raise ValueError(msg)

            # 3. Récupérer les paramètres de facturation
            # Récupérer le modèle pour accéder aux champs reminder1_fee, etc.
            from models import CompanyBillingSettings

            billing_settings = CompanyBillingSettings.query.filter_by(
                company_id=invoice.company_id
            ).first()
            if not billing_settings:
                msg = "Paramètres de facturation non trouvés"
                raise ValueError(msg)

            # 4. Calculer les frais selon le niveau
            fee_amount = Decimal("0.00")
            if input_data.level == LEVEL_ONE:
                fee_amount = Decimal(str(billing_settings.reminder1_fee or 0))
            elif input_data.level == LEVEL_THRESHOLD:
                fee_amount = Decimal(str(billing_settings.reminder2_fee or 0))
            elif input_data.level == LEVEL_THREE:
                fee_amount = Decimal(str(billing_settings.reminder3_fee or 0))

            # 5. Créer le rappel
            reminder = InvoiceReminder()
            reminder.invoice_id = input_data.invoice_id
            reminder.level = input_data.level
            reminder.added_fee = fee_amount
            reminder.generated_at = datetime.now(UTC)

            db.session.add(reminder)
            db.session.flush()  # Pour obtenir l'ID

            # 6. Ajouter les frais à la facture si nécessaire
            if fee_amount > Decimal(str(FEE_AMOUNT_ZERO)):
                # Créer une ligne de frais
                fee_line_data = {
                    "invoice_id": input_data.invoice_id,
                    "type": InvoiceLineType.REMINDER_FEE,
                    "description": f"Frais de rappel niveau {input_data.level}",
                    "qty": Decimal("1"),
                    "unit_price": fee_amount,
                    "line_total": fee_amount,
                    "vat_rate": None,
                    "vat_amount": Decimal("0.00"),
                    "total_with_vat": fee_amount,
                }
                self.invoice_line_repo.create(fee_line_data)

                # Mettre à jour les montants de la facture
                invoice.reminder_fee_amount += fee_amount
                invoice.total_amount += fee_amount
                invoice.balance_due += fee_amount

            # 7. Mettre à jour le niveau de rappel
            invoice.reminder_level = input_data.level
            invoice.last_reminder_at = datetime.now(UTC)

            # 8. Générer le PDF du rappel
            pdf_url = self.pdf_service.generate_reminder_pdf(invoice, input_data.level)
            reminder.pdf_url = pdf_url

            # 9. Commit de la transaction
            db.session.commit()

            logger.info(
                "Rappel niveau %s généré pour facture %s",
                input_data.level,
                invoice.invoice_number,
            )

            return GenerateInvoiceReminderOutput(success=True, reminder=reminder)

        except (OperationalError, DBAPIError, IntegrityError) as e:
            db.session.rollback()
            logger.error(
                "Erreur DB lors de la génération du rappel (DB error: %s): %s",
                type(e).__name__,
                str(e),
            )
            return GenerateInvoiceReminderOutput(
                success=False,
                error={"error": "Erreur de base de données"},
                status_code=500,
            )
        except ValueError as e:
            db.session.rollback()
            logger.warning(
                "Erreur de validation lors de la génération du rappel: %s", e
            )
            return GenerateInvoiceReminderOutput(
                success=False,
                error={"error": str(e)},
                status_code=400,
            )
        except Exception:
            db.session.rollback()
            logger.exception("Erreur inattendue lors de la génération du rappel")
            return GenerateInvoiceReminderOutput(
                success=False,
                error={"error": "Erreur interne"},
                status_code=500,
            )

