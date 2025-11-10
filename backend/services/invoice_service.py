import logging
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, Optional, cast

from sqlalchemy import and_

from models import (
    Booking,
    Client,
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

# Constantes pour éviter les valeurs magiques
LEVEL_ONE = 1
LEVEL_THRESHOLD = 2
FEE_AMOUNT_ZERO = 0
PERIOD_MONTH_THRESHOLD = 12
BALANCE_DUE_ZERO = 0
LATE_FEE_AMOUNT_ZERO = 0
OVERDUE_FEE_ZERO = 0
REMINDER_LEVEL_ZERO = 0
REMINDER_LEVEL_ONE = 1
REMINDER_LEVEL_THRESHOLD = 2


class InvoiceService:
    """Service pour la gestion des factures."""

    def __init__(self):
        super().__init__()
        self.pdf_service = PDFService()

    def cancel_invoice(self, invoice: Invoice, *, force: bool = False) -> None:
        """Annule une facture en libérant les réservations associées."""
        current_status = cast(InvoiceStatus, invoice.status)

        if not force and current_status not in {InvoiceStatus.DRAFT, InvoiceStatus.CANCELLED}:
            msg = "Seules les factures au statut 'draft' peuvent être annulées."
            raise ValueError(msg)

        if current_status == InvoiceStatus.CANCELLED:
            # Rien à faire, déjà annulée
            return

        try:
            # Libérer les réservations liées à chaque ligne
            for line in invoice.lines:
                if line.reservation_id:
                    booking = Booking.query.filter_by(id=line.reservation_id).first()
                    if booking and booking.invoice_line_id == line.id:
                        booking.invoice_line_id = None
                        booking.updated_at = datetime.now(UTC)

            invoice.status = InvoiceStatus.CANCELLED
            invoice.cancelled_at = datetime.now(UTC)
            invoice.updated_at = datetime.now(UTC)
            invoice.balance_due = Decimal("0.00")
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            app_logger.error("Erreur lors de l'annulation de la facture %s: %s", invoice.id, exc)
            raise

    def duplicate_invoice(self, invoice: Invoice) -> Dict[str, Any]:
        """Prépare un brouillon correctif à partir d'une facture existante.

        Cette méthode annule la facture originale (en libérant les trajets) et renvoie
        le contexte nécessaire côté frontend pour pré-remplir le formulaire de création.
        """
        current_status = cast(InvoiceStatus, invoice.status)
        if current_status == InvoiceStatus.DRAFT:
            raise ValueError("La facture est déjà un brouillon et peut être modifiée directement.")

        reservation_lines = [line for line in invoice.lines if line.reservation_id]
        if not reservation_lines:
            raise ValueError("Aucune course liée à cette facture ne peut être dupliquée.")

        bill_to_client_id: Optional[int] = invoice.bill_to_client_id
        billing_type = "third_party" if bill_to_client_id else "direct"
        reservation_ids: list[int] = []
        overrides: Dict[str, Dict[str, Any]] = {}

        for line in reservation_lines:
            if not line.reservation_id:
                continue

            reservation_ids.append(line.reservation_id)
            override: Dict[str, Any] = {
                "amount": float(line.line_total),
            }
            if line.vat_rate is not None:
                override["vat_rate"] = float(line.vat_rate)
            if line.adjustment_note:
                override["note"] = line.adjustment_note
            overrides[str(line.reservation_id)] = override

        client_payload: Dict[str, Any] | None = None
        if invoice.client:
            client = invoice.client
            user = getattr(client, "user", None)
            client_payload = {
                "id": client.id,
                "first_name": getattr(user, "first_name", None) if user else None,
                "last_name": getattr(user, "last_name", None) if user else None,
                "username": getattr(user, "username", None) if user else None,
                "full_name": f"{getattr(user, 'first_name', '') or ''} {getattr(user, 'last_name', '') or ''}".strip()
                if user
                else None,
            }

        # Annule la facture d'origine pour libérer les réservations
        self.cancel_invoice(invoice, force=True)

        return {
            "billing_type": billing_type,
            "client_id": invoice.client_id,
            "bill_to_client_id": invoice.bill_to_client_id,
            "period_year": invoice.period_year,
            "period_month": invoice.period_month,
            "reservation_ids": reservation_ids,
            "overrides": overrides,
            "client": client_payload,
        }

    def generate_invoice(
        self,
        company_id,
        client_id,
        period_year,
        period_month,
        bill_to_client_id=None,
        reservation_ids=None,
        overrides=None,
    ):
        """Génère une nouvelle facture pour un client et une période.

        Args:
            company_id: ID de l'entreprise
            client_id: ID du bénéficiaire du service (patient)
            period_year: Année de facturation
            period_month: Mois de facturation
            bill_to_client_id: ID du payeur (clinique/institution). Si None, client_id paie.
            reservation_ids: Liste d'IDs de réservations spécifiques. Si None, prend toutes les réservations non facturées.
            overrides: Dict facultatif {reservation_id: {amount, vat_rate, note}}

        Returns:
            Invoice: La facture créée

        """
        try:
            # Récupérer les paramètres de facturation
            billing_settings = self._get_billing_settings(company_id)

            overrides_map: Dict[int, Dict[str, Any]] = {}
            if overrides and isinstance(overrides, dict):
                for key, value in overrides.items():
                    try:
                        reservation_id = int(key)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, dict):
                        overrides_map[reservation_id] = value

            # Si bill_to_client_id est fourni, vérifier que c'est une institution
            if bill_to_client_id:
                bill_to_client = Client.query.filter_by(
                    id=bill_to_client_id,
                    company_id=company_id
                ).first()

                if not bill_to_client:
                    msg = "Client payeur non trouvé"
                    raise ValueError(msg)

                if not bill_to_client.is_institution:
                    app_logger.warning(
                        "Le client %s n'est pas marqué comme institution", bill_to_client_id
                    )

            # Récupérer les réservations
            target_statuses = ["COMPLETED", "RETURN_COMPLETED"]
            if reservation_ids:
                # Mode sélection manuelle : récupérer seulement les réservations spécifiées
                reservations = Booking.query.filter(
                    Booking.id.in_(reservation_ids),
                    Booking.company_id == company_id,
                    Booking.client_id == client_id,
                    Booking.status.in_(target_statuses),
                    Booking.invoice_line_id.is_(None),  # Pas déjà facturé
                ).all()

                if len(reservations) != len(reservation_ids):
                    app_logger.warning(
                        "Certaines réservations ne sont pas valides ou déjà facturées. Demandé: %s, Trouvé: %s",
                        len(reservation_ids), len(reservations)
                    )

                if not reservations:
                    msg = "Aucune réservation valide dans la sélection"
                    raise ValueError(msg)
            else:
                # Mode automatique : récupérer toutes les réservations de la période
                reservations = self._get_reservations_for_period(
                    company_id, client_id, period_year, period_month
                )

            if not reservations:
                msg = "Aucune réservation trouvée pour cette période"
                raise ValueError(msg)

            # Générer le numéro de facture
            invoice_number = self._generate_invoice_number(company_id, period_year, period_month)

            # Préparer la TVA
            vat_applicable = bool(getattr(billing_settings, "vat_applicable", True))
            default_vat_rate = Decimal("0")
            if vat_applicable and getattr(billing_settings, "vat_rate", None) is not None:
                default_vat_rate = Decimal(str(billing_settings.vat_rate)).quantize(Decimal("0.01"))
            vat_label = getattr(billing_settings, "vat_label", None)
            vat_number = getattr(billing_settings, "vat_number", None)

            two_places = Decimal("0.01")
            subtotal = Decimal("0.00")
            vat_total = Decimal("0.00")
            vat_breakdown: dict[str, dict[str, Decimal]] = {}

            # Récupérer les infos du client pour les descriptions
            client = Client.query.get(client_id)
            patient_name = ""
            if client and client.user:
                patient_name = f"{client.user.first_name} {client.user.last_name}".strip()
            if not patient_name:
                patient_name = f"Client #{client_id}"

            # Créer la facture
            invoice = Invoice()
            invoice.company_id = company_id
            invoice.client_id = client_id
            invoice.bill_to_client_id = bill_to_client_id
            invoice.period_month = period_month
            invoice.period_year = period_year
            invoice.invoice_number = invoice_number
            invoice.currency = "CHF"
            invoice.issued_at = datetime.now(UTC)
            invoice.due_date = datetime.now(UTC) + timedelta(days=billing_settings.payment_terms_days or 30)
            invoice.status = InvoiceStatus.DRAFT

            db.session.add(invoice)
            db.session.flush()  # Pour obtenir l'ID

            # Créer les lignes de facture avec le nom du patient
            for reservation in reservations:
                base_amount = Decimal(str(reservation.amount or 0)).quantize(two_places)
                override = overrides_map.get(reservation.id)
                if override and "amount" in override and override["amount"] is not None:
                    try:
                        base_amount = Decimal(str(override["amount"])).quantize(two_places, rounding=ROUND_HALF_UP)
                    except (InvalidOperation, ValueError, TypeError):
                        app_logger.warning("Montant override invalide pour réservation %s", reservation.id)

                line_vat_rate = default_vat_rate
                if override and override.get("vat_rate") is not None:
                    try:
                        line_vat_rate = Decimal(str(override["vat_rate"])).quantize(Decimal("0.01"))
                    except (InvalidOperation, ValueError, TypeError):
                        app_logger.warning("TVA override invalide pour réservation %s", reservation.id)
                        line_vat_rate = default_vat_rate

                if not vat_applicable:
                    line_vat_rate = Decimal("0")

                vat_amount = (base_amount * line_vat_rate / Decimal("100")).quantize(two_places, rounding=ROUND_HALF_UP)
                total_with_vat = (base_amount + vat_amount).quantize(two_places, rounding=ROUND_HALF_UP)

                # Si facturation tierce, inclure le nom du patient dans la description
                if bill_to_client_id:
                    description = f"Trajet pour {patient_name}: {reservation.pickup_location} → {reservation.dropoff_location}"
                else:
                    description = f"Trajet {reservation.pickup_location} → {reservation.dropoff_location}"

                line = InvoiceLine()
                line.invoice_id = invoice.id
                line.type = InvoiceLineType.RIDE
                line.description = description
                line.qty = 1
                line.unit_price = base_amount
                line.line_total = base_amount
                line.vat_rate = line_vat_rate
                line.vat_amount = vat_amount
                line.total_with_vat = total_with_vat
                if override and override.get("note"):
                    line.adjustment_note = str(override["note"])[:500]
                line.reservation_id = reservation.id
                db.session.add(line)
                db.session.flush()  # Pour obtenir l'ID de la ligne

                # NOUVEAU: Lier la réservation à la ligne de facture pour éviter double facturation
                reservation.invoice_line_id = line.id

                subtotal += base_amount
                vat_total += vat_amount
                rate_key = f"{line_vat_rate.normalize()}"
                if rate_key not in vat_breakdown:
                    vat_breakdown[rate_key] = {"base": Decimal("0.00"), "vat": Decimal("0.00")}
                vat_breakdown[rate_key]["base"] += base_amount
                vat_breakdown[rate_key]["vat"] += vat_amount

            invoice.subtotal_amount = subtotal
            invoice.vat_total_amount = vat_total
            invoice.total_amount = subtotal + vat_total
            invoice.balance_due = invoice.total_amount
            vat_payload: Dict[str, Dict[str, float]] = {
                rate: {
                    "base": float(values["base"].quantize(two_places)),
                    "vat": float(values["vat"].quantize(two_places)),
                }
                for rate, values in vat_breakdown.items()
            }
            invoice.vat_breakdown = cast(Any, vat_payload)

            if isinstance(invoice.meta, dict):
                current_meta: Dict[str, Any] = dict(invoice.meta)
            else:
                current_meta = {}
            current_meta["vat"] = {
                "applicable": vat_applicable and (default_vat_rate > Decimal("0")),
                "default_rate": float(default_vat_rate),
                "label": vat_label,
                "number": vat_number,
            }
            invoice.meta = cast(Any, current_meta)

            # Générer le PDF
            pdf_url = self.pdf_service.generate_invoice_pdf(invoice)
            invoice.pdf_url = pdf_url

            db.session.commit()

            if bill_to_client_id:
                app_logger.info(
                    "Facture générée: %s pour client %s (facturée à institution %s)",
                    invoice_number, client_id, bill_to_client_id
                )
            else:
                app_logger.info(
                    "Facture générée: %s pour client %s",
                    invoice_number, client_id
                )

            return invoice

        except Exception as e:
            db.session.rollback()
            app_logger.error("Erreur lors de la génération de la facture: %s", str(e))
            raise

    def generate_consolidated_invoice(
        self,
        company_id,
        client_ids,
        period_year,
        period_month,
        bill_to_client_id,
        client_reservations=None,
        overrides=None,
    ):
        """Génère plusieurs factures pour différents clients mais toutes adressées à une institution.

        Args:
            company_id: ID de l'entreprise
            client_ids: Liste des IDs des patients
            period_year: Année de facturation
            period_month: Mois de facturation
            bill_to_client_id: ID de l'institution payeuse (clinique)
            client_reservations: Dict {client_id: [reservation_ids]} pour sélection manuelle. Si None, mode auto.
            overrides: Dict optionnel {reservation_id: {amount, vat_rate, note}}

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
                    app_logger.warning("Facture déjà existante pour client %s, période %s/%s", client_id, period_month, period_year)
                    errors.append({
                        "client_id": client_id,
                        "error": "Facture déjà existante pour cette période"
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
                    reservation_ids=reservation_ids_for_client,
                    overrides=overrides,
                )
                invoices.append(invoice)

            except ValueError as e:
                app_logger.warning("Impossible de créer facture pour client %s: %s", client_id, e)
                errors.append({
                    "client_id": client_id,
                    "error": str(e)
                })
                continue
            except Exception as e:
                app_logger.error("Erreur inattendue pour client %s: %s", client_id, e)
                errors.append({
                    "client_id": client_id,
                    "error": f"Erreur interne: {e!s}"
                })
                continue

        app_logger.info(
            "Facturation consolidée: %s factures créées, %s erreurs pour institution %s",
            len(invoices), len(errors), bill_to_client_id
        )

        return {
            "invoices": invoices,
            "errors": errors,
            "success_count": len(invoices),
            "error_count": len(errors)
        }

    def generate_reminder(self, invoice_id, level):
        """Génère un rappel pour une facture."""
        try:
            invoice = Invoice.query.filter_by(id=invoice_id).first()
            if not invoice:
                msg = "Facture non trouvée"
                raise ValueError(msg)

            if level <= invoice.reminder_level:
                msg = f"Le rappel niveau {level} a déjà été généré"
                raise ValueError(msg)

            # Récupérer les paramètres de facturation
            billing_settings = self._get_billing_settings(invoice.company_id)

            # Calculer les frais selon le niveau
            fee_amount = 0
            if level == LEVEL_ONE:
                fee_amount = billing_settings.reminder1fee
            elif level == LEVEL_THRESHOLD:
                fee_amount = billing_settings.reminder2fee
            elif level == LEVEL_THRESHOLD:
                fee_amount = billing_settings.reminder3fee

            # Créer le rappel
            reminder = InvoiceReminder()
            reminder.invoice_id = invoice_id
            reminder.level = level
            reminder.added_fee = fee_amount
            reminder.generated_at = datetime.now(UTC)

            db.session.add(reminder)

            # Ajouter les frais à la facture si nécessaire
            if fee_amount > FEE_AMOUNT_ZERO:
                # Ajouter une ligne de frais
                fee_line = InvoiceLine()
                fee_line.invoice_id = invoice_id
                fee_line.type = InvoiceLineType.REMINDER_FEE
                fee_line.description = f"Frais de rappel niveau {level}"
                fee_line.qty = 1
                fee_line.unit_price = fee_amount
                fee_line.line_total = fee_amount
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

            app_logger.info("Rappel niveau %s généré pour facture %s", level, invoice.invoice_number)
            return reminder

        except Exception as e:
            db.session.rollback()
            app_logger.error("Erreur lors de la génération du rappel: %s", str(e))
            raise

    def _get_billing_settings(self, company_id):
        """Récupère les paramètres de facturation d'une entreprise."""
        settings = CompanyBillingSettings.query.filter_by(company_id=company_id).first()
        if not settings:
            # Créer des paramètres par défaut
            settings = CompanyBillingSettings()
            settings.company_id = company_id
            db.session.add(settings)
            db.session.commit()
        return settings

    def _get_reservations_for_period(self, company_id, client_id, period_year, period_month):
        """Récupère les réservations d'un client pour une période donnée."""
        start_date = datetime(period_year, period_month, 1)
        end_date = datetime(period_year + 1, 1, 1) if period_month == PERIOD_MONTH_THRESHOLD else datetime(period_year, period_month + 1, 1)

        return Booking.query.filter(
            and_(
                Booking.company_id == company_id,
                Booking.client_id == client_id,
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
                Booking.status.in_(["COMPLETED", "RETURN_COMPLETED"]),
                Booking.invoice_line_id.is_(None),
            )
        ).all()


    def _generate_invoice_number(self, company_id, period_year, period_month):
        """Génère un numéro de facture unique."""
        billing_settings = self._get_billing_settings(company_id)

        # Récupérer ou créer la séquence pour ce mois
        sequence = InvoiceSequence.query.filter_by(
            company_id=company_id,
            year=period_year,
            month=period_month
        ).first()

        if not sequence:
            sequence = InvoiceSequence()
            sequence.company_id = company_id
            sequence.year = period_year
            sequence.month = period_month
            sequence.sequence = 0
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
        """Vérifie et met à jour les factures en retard."""
        try:
            overdue_invoices = Invoice.query.filter(
                and_(
                    Invoice.status.in_([InvoiceStatus.SENT, InvoiceStatus.PARTIALLY_PAID]),
                    Invoice.balance_due > BALANCE_DUE_ZERO,
                    Invoice.due_date < datetime.now(UTC)
                )
            ).all()

            for invoice in overdue_invoices:
                # Vérifier si les frais de retard ont déjà été ajoutés
                if invoice.late_fee_amount == LATE_FEE_AMOUNT_ZERO:
                    billing_settings = self._get_billing_settings(invoice.company_id)

                    if billing_settings.overdue_fee and billing_settings.overdue_fee > OVERDUE_FEE_ZERO:
                        # Ajouter les frais de retard
                        late_fee_line = InvoiceLine()
                        late_fee_line.invoice_id = invoice.id
                        late_fee_line.type = InvoiceLineType.LATE_FEE
                        late_fee_line.description = "Frais de retard"
                        late_fee_line.qty = 1
                        late_fee_line.unit_price = billing_settings.overdue_fee
                        late_fee_line.line_total = billing_settings.overdue_fee
                        db.session.add(late_fee_line)

                        # Mettre à jour la facture
                        invoice.late_fee_amount = billing_settings.overdue_fee
                        invoice.total_amount += billing_settings.overdue_fee
                        invoice.balance_due += billing_settings.overdue_fee

                # Marquer comme en retard
                invoice.status = InvoiceStatus.OVERDUE

            db.session.commit()
            app_logger.info("%s factures marquées comme en retard", len(overdue_invoices))

        except Exception as e:
            db.session.rollback()
            app_logger.error("Erreur lors de la vérification des factures en retard: %s", str(e))
            raise

    def process_automatic_reminders(self):
        """Traite les rappels automatiques."""
        try:
            companies_with_auto_reminders = CompanyBillingSettings.query.filter_by(
                auto_reminders_enabled=True
            ).all()

            for settings in companies_with_auto_reminders:
                self._process_company_reminders(settings.company_id)

        except Exception as e:
            app_logger.error("Erreur lors du traitement des rappels automatiques: %s", str(e))
            raise

    def _process_company_reminders(self, company_id):
        """Traite les rappels automatiques pour une entreprise."""
        try:
            billing_settings = self._get_billing_settings(company_id)
            schedule_days: dict[str, int] = dict(billing_settings.reminder_schedule_days or {})  # type: ignore[arg-type]

            # Factures éligibles pour le 1er rappel
            if "1" in schedule_days:
                days_after_due = int(schedule_days["1"])
                cutoff_date = datetime.now(UTC) - timedelta(days=days_after_due)

                invoices_for_reminder1 = Invoice.query.filter_by(
                    company_id=company_id,
                    status=InvoiceStatus.OVERDUE,
                    reminder_level=REMINDER_LEVEL_ZERO
                ).filter(
                    Invoice.due_date <= cutoff_date
                ).all()

                for invoice in invoices_for_reminder1:
                    self.generate_reminder(invoice.id, 1)

            # Factures éligibles pour le 2e rappel
            if "2" in schedule_days:
                days_after_reminder1 = int(schedule_days["2"])
                cutoff_date = datetime.now(UTC) - timedelta(days=days_after_reminder1)

                invoices_for_reminder2 = Invoice.query.filter_by(
                    company_id=company_id,
                    status=InvoiceStatus.OVERDUE,
                    reminder_level=REMINDER_LEVEL_ONE
                ).filter(
                    Invoice.last_reminder_at <= cutoff_date
                ).all()

                for invoice in invoices_for_reminder2:
                    self.generate_reminder(invoice.id, 2)

            # Factures éligibles pour le 3e rappel
            if "3" in schedule_days:
                days_after_reminder2 = int(schedule_days["3"])
                cutoff_date = datetime.now(UTC) - timedelta(days=days_after_reminder2)

                invoices_for_reminder3 = Invoice.query.filter_by(
                    company_id=company_id,
                    status=InvoiceStatus.OVERDUE,
                    reminder_level=REMINDER_LEVEL_THRESHOLD
                ).filter(
                    Invoice.last_reminder_at <= cutoff_date
                ).all()

                for invoice in invoices_for_reminder3:
                    self.generate_reminder(invoice.id, 3)

        except Exception as e:
            app_logger.error("Erreur lors du traitement des rappels pour l'entreprise %s: %s", company_id, str(e))
            raise
