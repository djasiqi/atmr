# services/invoice_transfer_service.py
"""Service pour créer les factures liées aux transferts de courses."""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ext import db
from infrastructure.invoices.invoice_calculator import InvoiceCalculator
from infrastructure.invoices.invoice_number_generator import InvoiceNumberGenerator
from models.booking_transfer import BookingTransfer
from models.client import Client
from models.company import Company
from models.enums import (
    ClientType,
    InvoiceLineType,
    InvoiceStatus,
    TransferModel,
    UserRole,
)
from models.invoice import Invoice
from models.user import User
from repositories.client_repository import ClientRepository
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_line_repository import InvoiceLineRepository
from repositories.invoice_repository import InvoiceRepository
from repositories.invoice_sequence_repository import InvoiceSequenceRepository
from services.documents.pdf import PDFService

logger = logging.getLogger(__name__)


class InvoiceTransferService:
    """Service pour la création de factures pour les transferts."""

    def __init__(
        self,
        invoice_repo: InvoiceRepository | None = None,
        invoice_line_repo: InvoiceLineRepository | None = None,
        invoice_sequence_repo: InvoiceSequenceRepository | None = None,
        billing_settings_repo: CompanyBillingSettingsRepository | None = None,
        client_repo: ClientRepository | None = None,
        invoice_number_generator: InvoiceNumberGenerator | None = None,
        invoice_calculator: InvoiceCalculator | None = None,
        pdf_service: PDFService | None = None,
    ):
        """Initialise le service avec les dépendances."""
        super().__init__()
        self.invoice_repo = invoice_repo or InvoiceRepository()
        self.invoice_line_repo = invoice_line_repo or InvoiceLineRepository()
        self.invoice_sequence_repo = (
            invoice_sequence_repo or InvoiceSequenceRepository()
        )
        self.billing_settings_repo = (
            billing_settings_repo or CompanyBillingSettingsRepository()
        )
        self.client_repo = client_repo or ClientRepository()
        self.invoice_number_generator = (
            invoice_number_generator or InvoiceNumberGenerator()
        )
        self.invoice_calculator = invoice_calculator or InvoiceCalculator()
        self.pdf_service = pdf_service or PDFService()

    def create_invoices_for_transfer(self, transfer: BookingTransfer):
        """Créer les factures selon le modèle de transfert.

        Args:
            transfer: BookingTransfer validé

        Raises:
            ValueError: Si le modèle de transfert n'est pas supporté
            Exception: En cas d'erreur lors de la création des factures
        """
        if transfer.transfer_model == TransferModel.SUBCONTRACT:
            self._create_subcontract_invoices(transfer)
        elif transfer.transfer_model == TransferModel.ASSIGN_TO_PARTNER:
            self._create_assign_to_partner_invoices(transfer)
        elif transfer.transfer_model == TransferModel.MARKETPLACE:
            logger.warning("Modèle MARKETPLACE non encore implémenté")
            raise NotImplementedError("Modèle MARKETPLACE non encore implémenté")
        else:
            raise ValueError(f"Modèle de transfert inconnu: {transfer.transfer_model}")

    def _create_subcontract_invoices(self, transfer: BookingTransfer):
        """Créer les factures pour le modèle SUBCONTRACT.

        Modèle A : Sous-traitance classique
        - A facture le client final (via le système normal, pas ici)
        - B facture A (sous-traitance) - facture B2B
        """
        if not transfer.partner_cost:
            logger.warning(
                "Pas de coût partenaire défini, aucune facture sous-traitance créée"
            )
            return

        try:
            # 1. Récupérer ou créer un client "institution" pour l'entreprise
            # propriétaire. Ce client représente l'entreprise A dans le système de B
            owner_company = Company.query.get(transfer.owner_company_id)
            if not owner_company:
                raise ValueError(
                    f"Entreprise propriétaire {transfer.owner_company_id} introuvable"
                )

            # Chercher un client existant pour cette entreprise (B2B)
            # Un client avec is_institution=True et company_id=executing_company_id
            # qui représente owner_company
            b2b_client = (
                Client.query.filter_by(
                    company_id=transfer.executing_company_id,
                    is_institution=True,
                )
                .filter(Client.institution_name.ilike(f"%{owner_company.name}%"))
                .first()
            )

            # Si pas de client B2B, en créer un minimal
            if not b2b_client:
                # Créer un User minimal pour le client B2B
                b2b_user = User()
                b2b_user.username = f"company_{owner_company.id}_{uuid.uuid4().hex[:8]}"
                b2b_user.email = (
                    owner_company.billing_email or owner_company.contact_email
                )
                b2b_user.role = UserRole.COMPANY
                b2b_user.password = (
                    "!dummy_password_for_b2b_client!"  # Ne sera jamais utilisé
                )
                db.session.add(b2b_user)
                db.session.flush()

                b2b_client = Client()
                b2b_client.user_id = b2b_user.id
                b2b_client.company_id = transfer.executing_company_id
                b2b_client.is_institution = True
                b2b_client.institution_name = owner_company.name
                b2b_client.client_type = ClientType.CORPORATE
                b2b_client.billing_address = owner_company.address
                b2b_client.contact_email = (
                    owner_company.billing_email or owner_company.contact_email
                )
                b2b_client.contact_phone = owner_company.contact_phone
                db.session.add(b2b_client)
                db.session.flush()
                logger.info(
                    "Client B2B créé pour entreprise %s: client_id=%s",
                    owner_company.id,
                    b2b_client.id,
                )

            # 2. Récupérer les paramètres de facturation de l'entreprise exécutante
            billing_settings = self.billing_settings_repo.find_or_create(
                transfer.executing_company_id
            )

            # 3. Générer le numéro de facture
            now = datetime.now(UTC)
            sequence = self.invoice_sequence_repo.find_or_create(
                transfer.executing_company_id,
                now.year,
                now.month,
            )
            sequence = self.invoice_sequence_repo.increment_sequence(sequence.id)
            invoice_number = self.invoice_number_generator.generate(
                company_id=transfer.executing_company_id,
                period_year=now.year,
                period_month=now.month,
                billing_settings=billing_settings,
                sequence=sequence,
            )

            # 4. Calculer la TVA
            vat_rate = Decimal(str(transfer.vat_rate))
            vat_applicable = billing_settings.vat_applicable and vat_rate > Decimal("0")
            if not vat_applicable:
                vat_rate = Decimal("0")

            base_amount = transfer.partner_cost
            vat_amount, total_with_vat = self.invoice_calculator.calculate_vat(
                base_amount, vat_rate
            )

            # 5. Créer la facture
            invoice_data = {
                "company_id": transfer.executing_company_id,
                "client_id": b2b_client.id,
                "bill_to_client_id": None,
                "period_month": now.month,
                "period_year": now.year,
                "invoice_number": invoice_number,
                "currency": transfer.currency,
                "issued_at": now,
                "due_date": now
                + timedelta(days=transfer.partnership.payment_terms_days),
                "status": InvoiceStatus.DRAFT,
                "subtotal_amount": base_amount,
                "vat_total_amount": vat_amount,
                "total_amount": total_with_vat,
                "balance_due": total_with_vat,
            }
            invoice_dto = self.invoice_repo.create(invoice_data)
            invoice = Invoice.query.get(invoice_dto.id)
            if invoice is None:
                raise RuntimeError("Invoice not found after creation")

            # 6. Créer la ligne de facture
            booking = transfer.booking
            description = (
                f"Sous-traitance course #{booking.id} - "
                f"{booking.pickup_location} → {booking.dropoff_location}"
            )

            line_data = {
                "invoice_id": invoice.id,
                "type": InvoiceLineType.CUSTOM,  # Type custom pour sous-traitance
                "description": description,
                "qty": Decimal("1"),
                "unit_price": base_amount,
                "line_total": base_amount,
                "vat_rate": vat_rate if vat_applicable else None,
                "vat_amount": vat_amount,
                "total_with_vat": total_with_vat,
                "reservation_id": None,  # Pas de réservation directe
            }
            self.invoice_line_repo.create(line_data)

            # 7. Mettre à jour les métadonnées de la facture
            invoice.meta = {
                "transfer_id": transfer.id,
                "booking_id": booking.id,
                "partnership_id": transfer.partnership_id,
                "transfer_model": transfer.transfer_model.value,
                "source": "partnership_transfer",
            }

            # 8. Générer le PDF
            try:
                pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
                invoice.pdf_url = pdf_url
            except Exception as e:
                logger.warning(
                    "Erreur lors de la génération du PDF pour la facture %s: %s",
                    invoice_number,
                    e,
                )

            # 9. Commit
            db.session.commit()

            msg = (
                "Facture sous-traitance créée: %s - Entreprise %s facture %s "
                "pour %s %s"
            )
            logger.info(
                msg,
                invoice_number,
                transfer.executing_company_id,
                transfer.owner_company_id,
                total_with_vat,
                transfer.currency,
            )

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "Erreur lors de la création de la facture sous-traitance: %s", e
            )
            raise

    def _create_assign_to_partner_invoices(self, transfer: BookingTransfer):
        """Créer les factures pour le modèle ASSIGN_TO_PARTNER.

        Modèle B : Cession au partenaire
        - B facture directement le client
        - Optionnel: A facture une commission à B (non implémenté pour l'instant)
        """
        # Note: Pour ASSIGN_TO_PARTNER, la facture client sera créée
        # via le système de facturation normal lors de la génération mensuelle
        # car le booking appartient maintenant à l'entreprise exécutante

        msg = (
            "Modèle ASSIGN_TO_PARTNER: La facture client sera créée lors de "
            "la génération mensuelle pour l'entreprise %s"
        )
        logger.info(msg, transfer.executing_company_id)

        # Optionnel: Commission A → B (à implémenter si nécessaire)
        # Pour l'instant, on ne crée pas de facture de commission automatiquement
