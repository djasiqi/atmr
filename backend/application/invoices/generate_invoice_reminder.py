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
from infrastructure.invoices.invoice_calculator import round_to_5_cents
from models import Invoice, InvoiceReminder
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

            # 4. Calculer les frais selon le niveau et arrondir à 5 centimes
            fee_amount = Decimal("0.00")
            if input_data.level == LEVEL_ONE:
                fee_amount = round_to_5_cents(
                    Decimal(str(billing_settings.reminder1_fee or 0))
                )
            elif input_data.level == LEVEL_THRESHOLD:
                fee_amount = round_to_5_cents(
                    Decimal(str(billing_settings.reminder2_fee or 0))
                )
            elif input_data.level == LEVEL_THREE:
                fee_amount = round_to_5_cents(
                    Decimal(str(billing_settings.reminder3_fee or 0))
                )

            # 5. Calculer les montants consolidés (SANS modifier la facture principale)
            # Montant principal = solde dû de la facture initiale (sans frais de rappel)
            principal_amount = invoice.balance_due

            # Total à payer = principal + frais de rappel
            total_due = round_to_5_cents(principal_amount + fee_amount)

            # 6. Créer le rappel consolidé
            reminder = InvoiceReminder()
            reminder.invoice_id = input_data.invoice_id
            reminder.level = input_data.level
            reminder.added_fee = fee_amount
            reminder.principal_amount = principal_amount
            reminder.reminder_fee_amount = fee_amount
            reminder.total_due = total_due
            reminder.status = "OPEN"
            reminder.generated_at = datetime.now(UTC)

            db.session.add(reminder)
            db.session.flush()  # Pour obtenir l'ID (nécessaire pour le filename unique)

            # 7. Générer la référence QR-bill pour le rappel consolidé
            # On utilise le service QR-bill avec le montant total
            try:
                from services.billing import BillingProfileService
                from services.documents.qrbill import QRBillService

                qr_service = QRBillService()
                profile = BillingProfileService.get_by_company_id(invoice.company_id)

                if profile and profile.payment_reference_mode == "QRR":
                    # Générer une référence QRR pour le rappel
                    # On utilise un numéro de facture dérivé : {invoice_number}-R{level}
                    # Créer une facture "virtuelle" pour la génération
                    class VirtualInvoice:
                        def __init__(self, invoice_number, invoice_id):  # pyright: ignore[reportMissingSuperCall]
                            self.invoice_number = invoice_number
                            self.id = invoice_id

                    virtual_invoice = VirtualInvoice(
                        invoice_number=f"{invoice.invoice_number}-R{input_data.level}",
                        invoice_id=reminder.id,
                    )
                    reminder.qr_reference = qr_service._generate_qrr_reference(
                        virtual_invoice, profile
                    )
                    logger.info(
                        "QR-bill généré pour rappel consolidé: %s (montant: %s CHF)",
                        reminder.qr_reference,
                        float(total_due),
                    )
                else:
                    logger.info(
                        "Mode de référence non-QRR ou profil non trouvé, pas de QR-bill pour le rappel"
                    )
            except Exception as e:
                logger.warning(
                    "Échec de la génération du QR-bill pour le rappel: %s", str(e)
                )
                # Ne pas bloquer si le QR-bill échoue (peut être généré plus tard)

            # 8. Mettre à jour SEULEMENT le niveau de rappel (pas les montants)
            invoice.reminder_level = input_data.level
            invoice.last_reminder_at = datetime.now(UTC)
            # ✅ IMPORTANT : On ne modifie PAS total_amount, balance_due, reminder_fee_amount
            # La facture principale reste INTACTE

            # 9. Générer le PDF du rappel consolidé (PDF séparé, distinct de invoice.pdf_url)
            import os

            REMINDER_DEBUG = os.getenv("REMINDER_DEBUG", "0") == "1"

            if REMINDER_DEBUG:
                logger.info(
                    (
                        "[REMINDER_DEBUG] Avant génération PDF rappel: invoice_id=%s, level=%s, "
                        "invoice.pdf_url=%s, invoice.total=%s, invoice.due_date=%s"
                    ),
                    invoice.id,
                    input_data.level,
                    invoice.pdf_url,
                    float(invoice.total_amount) if invoice.total_amount else 0,
                    invoice.due_date.isoformat() if invoice.due_date else None,
                )

            pdf_url = self.pdf_service.generate_reminder_pdf(
                invoice, input_data.level, reminder
            )
            reminder.pdf_url = pdf_url

            # ✅ IMPORTANT : On ne régénère PAS le PDF de la facture principale
            # La facture principale reste INTACTE (principe clé du rappel consolidé)
            # - invoice.pdf_url reste inchangé
            # - invoice.total_amount reste inchangé
            # - invoice.lines reste inchangé
            # - invoice.due_date reste inchangé
            # Le PDF du rappel est stocké dans reminder.pdf_url (fichier reminder_*.pdf)

            if REMINDER_DEBUG:
                logger.info(
                    (
                        "[REMINDER_DEBUG] Après génération PDF rappel: invoice_id=%s, level=%s, "
                        "reminder.pdf_url=%s, invoice.pdf_url (INCHANGÉ)=%s, invoice.total (INCHANGÉ)=%s"
                    ),
                    invoice.id,
                    input_data.level,
                    reminder.pdf_url,
                    invoice.pdf_url,
                    float(invoice.total_amount) if invoice.total_amount else 0,
                )

            # 10. Commit de la transaction
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
