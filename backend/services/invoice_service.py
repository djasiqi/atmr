import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_

from models import (
    Booking,
    CompanyBillingSettings,
    Invoice,
    InvoiceLine,
    InvoiceLineType,
    InvoiceReminder,
    InvoiceSequence,
    InvoiceStatus,
    db,
)
from services.pdf_service import PDFService

app_logger = logging.getLogger("invoice_service")


class InvoiceService:
    """Service pour la gestion des factures"""

    def __init__(self):
        self.pdf_service = PDFService()

    def generate_invoice(self, company_id, client_id, period_year, period_month, bill_to_client_id=None, reservation_ids=None):
        """
        Génère une nouvelle facture pour un client et une période
        
        Args:
            company_id: ID de l'entreprise
            client_id: ID du bénéficiaire du service (patient)
            period_year: Année de facturation
            period_month: Mois de facturation
            bill_to_client_id: ID du payeur (clinique/institution). Si None, client_id paie.
            reservation_ids: Liste d'IDs de réservations spécifiques. Si None, prend toutes les réservations non facturées.
        
        Returns:
            Invoice: La facture créée
        """
        try:
            # Récupérer les paramètres de facturation
            billing_settings = self._get_billing_settings(company_id)

            # Si bill_to_client_id est fourni, vérifier que c'est une institution
            if bill_to_client_id:
                from models import Client
                bill_to_client = Client.query.filter_by(
                    id=bill_to_client_id,
                    company_id=company_id
                ).first()

                if not bill_to_client:
                    raise ValueError("Client payeur non trouvé")

                if not bill_to_client.is_institution:
                    app_logger.warning(
                        f"Le client {bill_to_client_id} n'est pas marqué comme institution"
                    )

            # Récupérer les réservations
            if reservation_ids:
                # Mode sélection manuelle : récupérer seulement les réservations spécifiées
                reservations = Booking.query.filter(
                    Booking.id.in_(reservation_ids),
                    Booking.company_id == company_id,
                    Booking.client_id == client_id,
                    Booking.status.in_(['COMPLETED', 'RETURN_COMPLETED', 'ACCEPTED']),
                    Booking.invoice_line_id.is_(None)  # Pas déjà facturé
                ).all()

                if len(reservations) != len(reservation_ids):
                    app_logger.warning(
                        f"Certaines réservations ne sont pas valides ou déjà facturées. "
                        f"Demandé: {len(reservation_ids)}, Trouvé: {len(reservations)}"
                    )

                if not reservations:
                    raise ValueError("Aucune réservation valide dans la sélection")
            else:
                # Mode automatique : récupérer toutes les réservations de la période
                reservations = self._get_reservations_for_period(
                    company_id, client_id, period_year, period_month
                )

            if not reservations:
                raise ValueError("Aucune réservation trouvée pour cette période")

            # Générer le numéro de facture
            invoice_number = self._generate_invoice_number(company_id, period_year, period_month)

            # Calculer les montants
            subtotal = sum(reservation.amount or 0 for reservation in reservations)

            # Récupérer les infos du client pour les descriptions
            from models import Client
            client = Client.query.get(client_id)
            patient_name = ""
            if client and client.user:
                patient_name = f"{client.user.first_name} {client.user.last_name}".strip()
            if not patient_name:
                patient_name = f"Client #{client_id}"

            # Créer la facture
            invoice = Invoice(
                company_id=company_id,
                client_id=client_id,
                bill_to_client_id=bill_to_client_id,  # NOUVEAU: Support facturation tierce
                period_month=period_month,
                period_year=period_year,
                invoice_number=invoice_number,
                currency="CHF",
                subtotal_amount=subtotal,
                total_amount=subtotal,
                balance_due=subtotal,
                issued_at=datetime.now(UTC),
                due_date=datetime.now(UTC) + timedelta(days=billing_settings.payment_terms_days or 30),
                status=InvoiceStatus.DRAFT
            )

            db.session.add(invoice)
            db.session.flush()  # Pour obtenir l'ID

            # Créer les lignes de facture avec le nom du patient
            for reservation in reservations:
                # Si facturation tierce, inclure le nom du patient dans la description
                if bill_to_client_id:
                    description = f"Trajet pour {patient_name}: {reservation.pickup_location} → {reservation.dropoff_location}"
                else:
                    description = f"Trajet {reservation.pickup_location} → {reservation.dropoff_location}"

                line = InvoiceLine(
                    invoice_id=invoice.id,
                    type=InvoiceLineType.RIDE,
                    description=description,
                    qty=1,
                    unit_price=reservation.amount or 0,
                    line_total=reservation.amount or 0,
                    reservation_id=reservation.id
                )
                db.session.add(line)
                db.session.flush()  # Pour obtenir l'ID de la ligne

                # NOUVEAU: Lier la réservation à la ligne de facture pour éviter double facturation
                reservation.invoice_line_id = line.id

            # Générer le PDF
            pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
            invoice.pdf_url = pdf_url

            db.session.commit()

            log_msg = f"Facture générée: {invoice_number} pour client {client_id}"
            if bill_to_client_id:
                log_msg += f" (facturée à institution {bill_to_client_id})"
            app_logger.info(log_msg)

            return invoice

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"Erreur lors de la génération de la facture: {str(e)}")
            raise

    def generate_consolidated_invoice(self, company_id, client_ids, period_year, period_month, bill_to_client_id, client_reservations=None):
        """
        Génère plusieurs factures pour différents clients mais toutes adressées à une institution
        
        Args:
            company_id: ID de l'entreprise
            client_ids: Liste des IDs des patients
            period_year: Année de facturation
            period_month: Mois de facturation
            bill_to_client_id: ID de l'institution payeuse (clinique)
            client_reservations: Dict {client_id: [reservation_ids]} pour sélection manuelle. Si None, mode auto.
        
        Returns:
            Dict avec invoices créées et erreurs
        """
        invoices = []
        errors = []

        for client_id in client_ids:
            try:
                # Vérifier qu'une facture n'existe pas déjà pour ce client et cette période
                existing_invoice = Invoice.query.filter_by(
                    company_id=company_id,
                    client_id=client_id,
                    period_year=period_year,
                    period_month=period_month
                ).first()

                if existing_invoice:
                    app_logger.warning(f"Facture déjà existante pour client {client_id}, période {period_month}/{period_year}")
                    errors.append({
                        'client_id': client_id,
                        'error': 'Facture déjà existante pour cette période'
                    })
                    continue

                # Récupérer les IDs de réservations pour ce client si fourni
                reservation_ids_for_client = None
                if client_reservations and client_id in client_reservations:
                    reservation_ids_for_client = client_reservations[client_id]

                invoice = self.generate_invoice(
                    company_id=company_id,
                    client_id=client_id,
                    period_year=period_year,
                    period_month=period_month,
                    bill_to_client_id=bill_to_client_id,
                    reservation_ids=reservation_ids_for_client  # NOUVEAU: Support sélection manuelle
                )
                invoices.append(invoice)

            except ValueError as e:
                app_logger.warning(f"Impossible de créer facture pour client {client_id}: {e}")
                errors.append({
                    'client_id': client_id,
                    'error': str(e)
                })
                continue
            except Exception as e:
                app_logger.error(f"Erreur inattendue pour client {client_id}: {e}")
                errors.append({
                    'client_id': client_id,
                    'error': f"Erreur interne: {str(e)}"
                })
                continue

        app_logger.info(
            f"Facturation consolidée: {len(invoices)} factures créées, "
            f"{len(errors)} erreurs pour institution {bill_to_client_id}"
        )

        return {
            'invoices': invoices,
            'errors': errors,
            'success_count': len(invoices),
            'error_count': len(errors)
        }

    def generate_reminder(self, invoice_id, level):
        """Génère un rappel pour une facture"""
        try:
            invoice = Invoice.query.filter_by(id=invoice_id).first()
            if not invoice:
                raise ValueError("Facture non trouvée")

            if level <= invoice.reminder_level:
                raise ValueError(f"Le rappel niveau {level} a déjà été généré")

            # Récupérer les paramètres de facturation
            billing_settings = self._get_billing_settings(invoice.company_id)

            # Calculer les frais selon le niveau
            fee_amount = 0
            if level == 1:
                fee_amount = billing_settings.reminder1_fee
            elif level == 2:
                fee_amount = billing_settings.reminder2_fee
            elif level == 3:
                fee_amount = billing_settings.reminder3_fee

            # Créer le rappel
            reminder = InvoiceReminder(
                invoice_id=invoice_id,
                level=level,
                added_fee=fee_amount,
                generated_at=datetime.now(UTC)
            )

            db.session.add(reminder)

            # Ajouter les frais à la facture si nécessaire
            if fee_amount > 0:
                # Ajouter une ligne de frais
                fee_line = InvoiceLine(
                    invoice_id=invoice_id,
                    type=InvoiceLineType.REMINDER_FEE,
                    description=f"Frais de rappel niveau {level}",
                    qty=1,
                    unit_price=fee_amount,
                    line_total=fee_amount
                )
                db.session.add(fee_line)

                # Mettre à jour les montants de la facture
                invoice.reminder_fee_amount += fee_amount
                invoice.total_amount += fee_amount
                invoice.balance_due += fee_amount

            # Mettre à jour le niveau de rappel
            invoice.reminder_level = level
            invoice.last_reminder_at = datetime.now(UTC)

            # Générer le PDF du rappel
            pdf_url = self.pdf_service.generate_reminder_pdf(invoice, level)
            reminder.pdf_url = pdf_url

            db.session.commit()

            app_logger.info(f"Rappel niveau {level} généré pour facture {invoice.invoice_number}")
            return reminder

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"Erreur lors de la génération du rappel: {str(e)}")
            raise

    def _get_billing_settings(self, company_id):
        """Récupère les paramètres de facturation d'une entreprise"""
        settings = CompanyBillingSettings.query.filter_by(company_id=company_id).first()
        if not settings:
            # Créer des paramètres par défaut
            settings = CompanyBillingSettings(company_id=company_id)
            db.session.add(settings)
            db.session.commit()
        return settings

    def _get_reservations_for_period(self, company_id, client_id, period_year, period_month):
        """Récupère les réservations d'un client pour une période donnée"""
        start_date = datetime(period_year, period_month, 1)
        if period_month == 12:
            end_date = datetime(period_year + 1, 1, 1)
        else:
            end_date = datetime(period_year, period_month + 1, 1)

        reservations = Booking.query.filter(
            and_(
                Booking.company_id == company_id,
                Booking.client_id == client_id,
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
                Booking.status.in_(['COMPLETED', 'RETURN_COMPLETED', 'ACCEPTED'])  # Réservations terminées/confirmées
            )
        ).all()

        return reservations

    def _generate_invoice_number(self, company_id, period_year, period_month):
        """Génère un numéro de facture unique"""
        billing_settings = self._get_billing_settings(company_id)

        # Récupérer ou créer la séquence pour ce mois
        sequence = InvoiceSequence.query.filter_by(
            company_id=company_id,
            year=period_year,
            month=period_month
        ).first()

        if not sequence:
            sequence = InvoiceSequence(
                company_id=company_id,
                year=period_year,
                month=period_month,
                sequence=0
            )
            db.session.add(sequence)

        # Incrémenter la séquence
        sequence.sequence += 1

        # Générer le numéro selon le format
        invoice_number = billing_settings.invoice_number_format.format(
            PREFIX=billing_settings.invoice_prefix,
            YYYY=period_year,
            MM=f"{period_month:02d}",
            SEQ4=f"{sequence.sequence:04d}"
        )

        db.session.commit()
        return invoice_number

    def check_overdue_invoices(self):
        """Vérifie et met à jour les factures en retard"""
        try:
            overdue_invoices = Invoice.query.filter(
                and_(
                    Invoice.status.in_([InvoiceStatus.SENT, InvoiceStatus.PARTIALLY_PAID]),
                    Invoice.balance_due > 0,
                    Invoice.due_date < datetime.now(UTC)
                )
            ).all()

            for invoice in overdue_invoices:
                # Vérifier si les frais de retard ont déjà été ajoutés
                if invoice.late_fee_amount == 0:
                    billing_settings = self._get_billing_settings(invoice.company_id)

                    if billing_settings.overdue_fee > 0:
                        # Ajouter les frais de retard
                        late_fee_line = InvoiceLine(
                            invoice_id=invoice.id,
                            type=InvoiceLineType.LATE_FEE,
                            description="Frais de retard",
                            qty=1,
                            unit_price=billing_settings.overdue_fee,
                            line_total=billing_settings.overdue_fee
                        )
                        db.session.add(late_fee_line)

                        # Mettre à jour la facture
                        invoice.late_fee_amount = billing_settings.overdue_fee
                        invoice.total_amount += billing_settings.overdue_fee
                        invoice.balance_due += billing_settings.overdue_fee

                # Marquer comme en retard
                invoice.status = InvoiceStatus.OVERDUE

            db.session.commit()
            app_logger.info(f"{len(overdue_invoices)} factures marquées comme en retard")

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"Erreur lors de la vérification des factures en retard: {str(e)}")
            raise

    def process_automatic_reminders(self):
        """Traite les rappels automatiques"""
        try:
            companies_with_auto_reminders = CompanyBillingSettings.query.filter_by(
                auto_reminders_enabled=True
            ).all()

            for settings in companies_with_auto_reminders:
                self._process_company_reminders(settings.company_id)

        except Exception as e:
            app_logger.error(f"Erreur lors du traitement des rappels automatiques: {str(e)}")
            raise

    def _process_company_reminders(self, company_id):
        """Traite les rappels automatiques pour une entreprise"""
        try:
            billing_settings = self._get_billing_settings(company_id)
            schedule_days = billing_settings.reminder_schedule_days

            # Factures éligibles pour le 1er rappel
            if '1' in schedule_days:
                days_after_due = schedule_days['1']
                cutoff_date = datetime.now(UTC) - timedelta(days=days_after_due)

                invoices_for_reminder1 = Invoice.query.filter(
                    and_(
                        Invoice.company_id == company_id,
                        Invoice.status == InvoiceStatus.OVERDUE,
                        Invoice.reminder_level == 0,
                        Invoice.due_date <= cutoff_date
                    )
                ).all()

                for invoice in invoices_for_reminder1:
                    self.generate_reminder(invoice.id, 1)

            # Factures éligibles pour le 2e rappel
            if '2' in schedule_days:
                days_after_reminder1 = schedule_days['2']
                cutoff_date = datetime.now(UTC) - timedelta(days=days_after_reminder1)

                invoices_for_reminder2 = Invoice.query.filter(
                    and_(
                        Invoice.company_id == company_id,
                        Invoice.status == InvoiceStatus.OVERDUE,
                        Invoice.reminder_level == 1,
                        Invoice.last_reminder_at <= cutoff_date
                    )
                ).all()

                for invoice in invoices_for_reminder2:
                    self.generate_reminder(invoice.id, 2)

            # Factures éligibles pour le 3e rappel
            if '3' in schedule_days:
                days_after_reminder2 = schedule_days['3']
                cutoff_date = datetime.now(UTC) - timedelta(days=days_after_reminder2)

                invoices_for_reminder3 = Invoice.query.filter(
                    and_(
                        Invoice.company_id == company_id,
                        Invoice.status == InvoiceStatus.OVERDUE,
                        Invoice.reminder_level == 2,
                        Invoice.last_reminder_at <= cutoff_date
                    )
                ).all()

                for invoice in invoices_for_reminder3:
                    self.generate_reminder(invoice.id, 3)

        except Exception as e:
            app_logger.error(f"Erreur lors du traitement des rappels pour l'entreprise {company_id}: {str(e)}")
            raise
