# services/partner_invoice_service.py
# pyright: reportPossiblyUnboundVariable=false
"""Service pour gérer la facturation mensuelle consolidée des partenaires."""

import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import and_, func, or_, select

from ext import db
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from infrastructure.invoices.invoice_number_generator import InvoiceNumberGenerator
from models.booking import Booking
from models.booking_transfer import BookingTransfer
from models.enums import TransferStatus
from models.partner_invoice import (
    PartnerInvoice,
    PartnerInvoiceStatus,
    partner_invoice_transfers,
)
from models.partnership import Partnership
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)
from repositories.invoice_sequence_repository import InvoiceSequenceRepository
from services.documents.pdf import PDFService

logger = logging.getLogger(__name__)


class PartnerInvoiceService:
    """Service pour la gestion des factures mensuelles partenaires."""

    def __init__(
        self,
        billing_settings_repo: CompanyBillingSettingsRepository | None = None,
        invoice_sequence_repo: InvoiceSequenceRepository | None = None,
        invoice_number_generator: InvoiceNumberGenerator | None = None,
        invoice_calculator: InvoiceCalculator | None = None,
        pdf_service: PDFService | None = None,
    ):
        """Initialise le service avec les dépendances."""
        # Appel à super() pour satisfaire le linter (classe hérite implicitement de object)
        super().__init__()
        self.billing_settings_repo = (
            billing_settings_repo or CompanyBillingSettingsRepository()
        )
        self.invoice_sequence_repo = (
            invoice_sequence_repo or InvoiceSequenceRepository()
        )
        self.invoice_number_generator = (
            invoice_number_generator or InvoiceNumberGenerator()
        )
        self.invoice_calculator = invoice_calculator or InvoiceCalculator()
        self.pdf_service = pdf_service or PDFService()

    def generate_monthly_invoice(
        self,
        partnership_id: int,
        year: int,
        month: int,
        executing_company_id: int,
        transfer_ids: list[int] | None = None,
        overrides: dict[int, dict[str, object]] | None = None,
    ) -> PartnerInvoice:
        """Génère une facture mensuelle consolidée pour un partenariat.

        Args:
            partnership_id: ID du partenariat
            year: Année de la période
            month: Mois de la période (1-12)
            executing_company_id: ID de l'entreprise exécutante qui génère la facture
            transfer_ids: IDs des transferts à inclure (optionnel, sinon tous)
            overrides: {transfer_id: {amount: float, note?: str}} pour modifier les montants

        Returns:
            PartnerInvoice créée

        Raises:
            ValueError: Si le partenariat n'existe pas ou s'il n'y a pas de transferts
        """
        partnership = Partnership.query.get(partnership_id)
        if not partnership:
            raise ValueError(f"Partenariat {partnership_id} introuvable")

        # Vérifier que l'entreprise qui génère la facture est bien le partenaire (exécutante)
        # OU qu'elle est owner mais executing dans les transferts (partenariat créé dans le mauvais sens)
        is_partner = executing_company_id == partnership.partner_company_id
        is_owner_executing = executing_company_id == partnership.owner_company_id
        # Vérifier si l'entreprise est owner mais executing dans les transferts
        if is_owner_executing:
            transfers_as_executing = BookingTransfer.query.filter(
                BookingTransfer.partnership_id == partnership_id,
                BookingTransfer.executing_company_id == executing_company_id,
                BookingTransfer.status == TransferStatus.COMPLETED,
            ).count()
            if transfers_as_executing == 0:
                is_owner_executing = False
        if not is_partner and not is_owner_executing:
            raise ValueError(
                "Seule l'entreprise partenaire (ID: "
                + str(partnership.partner_company_id)
                + ") "
                + "ou l'entreprise owner exécutante (ID: "
                + str(partnership.owner_company_id)
                + ") "
                + "peut générer une facture pour ce partenariat"
            )

        # Pas de vérification de facture existante ici
        # On permet plusieurs factures pour la même période et la même entreprise exécutante
        # car ce peuvent être des transferts différents (par exemple, une facture créée en début de mois
        # et une autre créée plus tard dans le mois pour de nouveaux transferts)
        # La seule protection est que chaque transfert ne peut être facturé qu'une seule fois
        # (gérée par la requête qui exclut les transferts déjà facturés)

        # Récupérer tous les transferts validés et non facturés de la période
        # Inclure si scheduled_time OU validated_at dans la période (évite exclusions TZ)
        DECEMBER = 12
        start_naive = datetime(year, month, 1)
        end_naive = (
            datetime(year + 1, 1, 1)
            if month == DECEMBER
            else datetime(year, month + 1, 1)
        )
        start_utc = datetime(year, month, 1, tzinfo=UTC)
        end_utc = (
            datetime(year + 1, 1, 1, tzinfo=UTC)
            if month == DECEMBER
            else datetime(year, month + 1, 1, tzinfo=UTC)
        )

        all_eligible = (
            BookingTransfer.query.join(
                Booking, BookingTransfer.booking_id == Booking.id
            )
            .filter(
                BookingTransfer.partnership_id == partnership_id,
                BookingTransfer.status == TransferStatus.COMPLETED,
                BookingTransfer.is_validated == True,  # noqa: E712
                BookingTransfer.executing_company_id == executing_company_id,
                or_(
                    and_(
                        Booking.scheduled_time.isnot(None),
                        Booking.scheduled_time >= start_naive,
                        Booking.scheduled_time < end_naive,
                    ),
                    and_(
                        BookingTransfer.validated_at.isnot(None),
                        BookingTransfer.validated_at >= start_utc,
                        BookingTransfer.validated_at < end_utc,
                    ),
                ),
            )
            .filter(
                ~BookingTransfer.id.in_(
                    select(partner_invoice_transfers.c.booking_transfer_id)
                    .select_from(partner_invoice_transfers.join(PartnerInvoice))
                    .where(PartnerInvoice.status != PartnerInvoiceStatus.CANCELLED)
                )
            )
            .all()
        )

        if transfer_ids:
            eligible_ids = {t.id for t in all_eligible}
            invalid = set(transfer_ids) - eligible_ids
            if invalid:
                raise ValueError(
                    f"Transferts invalides ou déjà facturés: {sorted(invalid)}"
                )
            transfers = [t for t in all_eligible if t.id in transfer_ids]
        else:
            transfers = all_eligible

        if not transfers:
            raise ValueError(
                f"Aucun transfert validé non facturé trouvé pour la période {year}-{month:02d}"
            )

        # Calculer les totaux et les montants par ligne (avec overrides optionnels)
        overrides_map = overrides or {}
        line_amounts: dict[int, Decimal] = {}
        subtotal = Decimal("0")
        for transfer in transfers:
            ov = overrides_map.get(transfer.id, {})
            amount = ov.get("amount")
            if amount is not None:
                base = Decimal(str(amount))
            elif transfer.partner_cost:
                base = Decimal(str(transfer.partner_cost))
            else:
                base = Decimal("0")
            if base > 0:
                rounded_cost = round_to_5_cents(base)
                line_amounts[transfer.id] = rounded_cost
                subtotal += rounded_cost

        # Arrondir le subtotal total à 5 centimes
        subtotal = round_to_5_cents(subtotal)

        # ✅ Récupérer le crédit disponible depuis les factures précédentes du même partenariat
        # où l'executing_company_id est le même (même direction de facturation)
        available_credit = db.session.query(
            func.coalesce(func.sum(PartnerInvoice.credit_balance), 0)
        ).filter(
            PartnerInvoice.partnership_id == partnership_id,
            PartnerInvoice.executing_company_id == executing_company_id,
            PartnerInvoice.status.in_(
                [PartnerInvoiceStatus.PAID, PartnerInvoiceStatus.PARTIALLY_PAID]
            ),
            PartnerInvoice.credit_balance > 0,
        ).scalar() or Decimal("0")

        # Déduire le crédit disponible du subtotal
        if available_credit > 0:
            subtotal_before_credit = subtotal
            subtotal = max(Decimal("0"), subtotal - available_credit)
            credit_used = subtotal_before_credit - subtotal
            logger.info(
                "Crédit disponible: %s CHF, utilisé: %s CHF, subtotal avant: %s CHF, après: %s CHF",
                available_credit,
                credit_used,
                subtotal_before_credit,
                subtotal,
            )
            # Mettre à jour le crédit utilisé dans les factures précédentes
            # On réduit le crédit des factures les plus anciennes en premier
            remaining_credit_to_use = credit_used
            previous_invoices = (
                PartnerInvoice.query.filter(
                    PartnerInvoice.partnership_id == partnership_id,
                    PartnerInvoice.executing_company_id == executing_company_id,
                    PartnerInvoice.status.in_(
                        [
                            PartnerInvoiceStatus.PAID,
                            PartnerInvoiceStatus.PARTIALLY_PAID,
                        ]
                    ),
                    PartnerInvoice.credit_balance > 0,
                )
                .order_by(PartnerInvoice.issued_at.asc())
                .all()
            )
            for prev_invoice in previous_invoices:
                if remaining_credit_to_use <= 0:
                    break
                credit_to_deduct = min(
                    prev_invoice.credit_balance, remaining_credit_to_use
                )
                prev_invoice.credit_balance -= credit_to_deduct
                remaining_credit_to_use -= credit_to_deduct
                logger.info(
                    "Crédit déduit de la facture %s: %s CHF (reste: %s CHF)",
                    prev_invoice.invoice_number,
                    credit_to_deduct,
                    prev_invoice.credit_balance,
                )

        # ✅ Récupérer les paramètres de facturation de l'entreprise qui génère la facture
        # Utiliser executing_company_id (passé en paramètre) au lieu de partnership.partner_company_id
        # car si le partenariat a été créé dans le mauvais sens, l'entreprise peut être owner
        # mais executing dans les transferts
        billing_settings = self.billing_settings_repo.find_or_create(
            executing_company_id
        )

        # Délai de paiement: priorité aux paramètres de l'entreprise exécutante
        # (cohérent avec la section "Paramètres de paiement" côté settings).
        payment_terms_days = int(
            billing_settings.payment_terms_days
            if billing_settings and billing_settings.payment_terms_days
            else (partnership.payment_terms_days or 30)
        )

        # Calculer la TVA
        vat_rate = Decimal(str(billing_settings.vat_rate or 0))
        vat_applicable = billing_settings.vat_applicable and vat_rate > Decimal("0")
        if not vat_applicable:
            vat_rate = Decimal("0")

        vat_amount, total_with_vat = self.invoice_calculator.calculate_vat(
            subtotal, vat_rate
        )

        # Générer le numéro de facture
        # Utiliser executing_company_id au lieu de partnership.partner_company_id
        # pour gérer le cas où le partenariat a été créé dans le mauvais sens
        now = datetime.now(UTC)
        sequence = self.invoice_sequence_repo.find_or_create(
            executing_company_id,
            year,
            month,
        )
        sequence = self.invoice_sequence_repo.increment_sequence(sequence.id)
        invoice_number = self.invoice_number_generator.generate(
            company_id=executing_company_id,
            period_year=year,
            period_month=month,
            billing_settings=billing_settings,
            sequence=sequence,
        )
        # Préfixer avec "PARTNER-" pour distinguer des factures normales
        invoice_number = f"PARTNER-{invoice_number}"

        # Créer la facture
        partner_invoice = PartnerInvoice()
        partner_invoice.partnership_id = partnership_id
        partner_invoice.executing_company_id = executing_company_id
        partner_invoice.period_year = year
        partner_invoice.period_month = month
        partner_invoice.invoice_number = invoice_number
        partner_invoice.subtotal_amount = subtotal
        partner_invoice.vat_amount = vat_amount
        partner_invoice.total_amount = total_with_vat
        partner_invoice.currency = transfers[0].currency if transfers else "CHF"
        partner_invoice.status = PartnerInvoiceStatus.DRAFT
        partner_invoice.issued_at = now
        partner_invoice.due_date = now + timedelta(days=payment_terms_days)

        db.session.add(partner_invoice)
        db.session.flush()

        # Lier les transferts à la facture
        for transfer in transfers:
            db.session.execute(
                partner_invoice_transfers.insert().values(
                    partner_invoice_id=partner_invoice.id,
                    booking_transfer_id=transfer.id,
                )
            )

        # Générer le PDF (brouillon : la facture reste DRAFT jusqu'à action explicite "Envoyer")
        try:
            pdf_url = self._generate_invoice_pdf(
                partner_invoice, transfers, line_amounts=line_amounts
            )
            partner_invoice.pdf_url = pdf_url
        except Exception as e:
            logger.warning(
                "Erreur lors de la génération du PDF pour la facture partenaire %s: %s",
                invoice_number,
                e,
            )

        # Rester en DRAFT : workflow Brouillon → (action "Envoyer") → Envoyée (SENT)
        # Ne pas appeler mark_as_sent ici.

        db.session.commit()

        logger.info(
            "Facture mensuelle partenaire créée: %s - Partenariat %s - %s transferts - %s %s",
            invoice_number,
            partnership_id,
            len(transfers),
            total_with_vat,
            partner_invoice.currency,
        )

        return partner_invoice

    def _generate_invoice_pdf(
        self,
        partner_invoice: PartnerInvoice,
        transfers: list[BookingTransfer],
        *,
        line_amounts: dict[int, Decimal] | None = None,
    ) -> str:
        """Génère le PDF de la facture partenaire.

        Args:
            partner_invoice: Facture partenaire
            transfers: Liste des transferts inclus dans la facture
            line_amounts: Montants par transfer_id (après overrides), pour affichage cohérent

        Returns:
            URL du PDF généré
        """
        from pathlib import Path

        from flask import current_app

        from services.partnerships.invoices_pdf import (
            generate_partner_invoice_pdf_content,
        )

        # Générer le contenu PDF
        pdf_content = generate_partner_invoice_pdf_content(
            partner_invoice, transfers, line_amounts=line_amounts or {}
        )

        # Sauvegarder le fichier
        filename = (
            f"partner_invoice_{partner_invoice.invoice_number}_"
            f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        # ✅ Chemin correct: /app/uploads (pas /app/services/uploads)
        uploads_dir = Path(current_app.config.get("UPLOAD_FOLDER", "/app/uploads"))
        invoices_dir = Path(uploads_dir, "invoices")
        invoices_dir.mkdir(parents=True, exist_ok=True)
        filepath = Path(invoices_dir, filename)

        with filepath.open("wb") as f:
            f.write(pdf_content)

        # URL dynamique depuis config (127.0.0.1 en dev évite IPv6 localhost + ERR_CONNECTION_RESET)
        pdf_base_url = current_app.config.get("PDF_BASE_URL", "http://127.0.0.1:5000")
        uploads_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")

        pdf_url = f"{pdf_base_url}{uploads_base}/invoices/{filename}"

        logger.info("PDF de facture partenaire généré: %s", pdf_url)
        return pdf_url

    def regenerate_pdf(self, partner_invoice_id: int) -> str:
        """Régénère le PDF d'une facture partenaire avec les données les plus récentes.

        Cette méthode recharge TOUTES les données depuis la base de données
        (adresses, courses, montants, format) pour générer un PDF à jour.

        Args:
            partner_invoice_id: ID de la facture partenaire

        Returns:
            URL du nouveau PDF généré

        Raises:
            ValueError: Si la facture partenaire n'existe pas
            Exception: Si la génération PDF échoue
        """
        # Forcer le rechargement depuis la DB pour avoir les données à jour
        db.session.expire_all()

        partner_invoice = PartnerInvoice.query.get(partner_invoice_id)
        if not partner_invoice:
            raise ValueError(f"Facture partenaire {partner_invoice_id} introuvable")

        # Récupérer les transferts associés à cette facture avec eager loading
        # pour avoir les données les plus récentes (booking, client, adresses)
        transfers = (
            db.session.query(BookingTransfer)
            .join(
                partner_invoice_transfers,
                BookingTransfer.id == partner_invoice_transfers.c.booking_transfer_id,
            )
            .filter(
                partner_invoice_transfers.c.partner_invoice_id == partner_invoice.id
            )
            .all()
        )

        if not transfers:
            raise ValueError(
                f"Aucun transfert associé à la facture partenaire {partner_invoice_id}"
            )

        # Forcer le rechargement des relations pour chaque transfert
        for transfer in transfers:
            db.session.refresh(transfer)
            if transfer.booking:
                db.session.refresh(transfer.booking)
                if transfer.booking.client:
                    db.session.refresh(transfer.booking.client)
            if transfer.executing_company:
                db.session.refresh(transfer.executing_company)

        # Recharger le partenariat pour avoir les adresses à jour
        if partner_invoice.partnership:
            db.session.refresh(partner_invoice.partnership)
            if partner_invoice.partnership.owner_company:
                db.session.refresh(partner_invoice.partnership.owner_company)
            if partner_invoice.partnership.partner_company:
                db.session.refresh(partner_invoice.partnership.partner_company)

        logger.info(
            "Régénération PDF facture partenaire %s avec %d transferts (données rechargées)",
            partner_invoice.invoice_number,
            len(transfers),
        )

        # Générer le nouveau PDF
        try:
            pdf_url = self._generate_invoice_pdf(partner_invoice, transfers)
            partner_invoice.pdf_url = pdf_url
            db.session.commit()

            logger.info(
                "PDF régénéré avec succès pour facture partenaire %s: %s",
                partner_invoice.invoice_number,
                pdf_url,
            )
            return pdf_url
        except Exception as e:
            logger.exception(
                "Erreur lors de la régénération PDF pour facture partenaire %s: %s",
                partner_invoice_id,
                str(e),
            )
            raise ValueError(f"Erreur lors de la régénération du PDF: {e}") from e

    def mark_as_sent(self, partner_invoice_id: int, company_id: int) -> PartnerInvoice:
        """Marque une facture partenaire comme envoyée (workflow Brouillon → Envoyée).

        Args:
            partner_invoice_id: ID de la facture
            company_id: ID de l'entreprise qui envoie (doit être executing_company_id)

        Returns:
            PartnerInvoice mise à jour

        Raises:
            ValueError: Si facture introuvable, company non autorisée ou statut != DRAFT
        """
        partner_invoice = PartnerInvoice.query.get(partner_invoice_id)
        if not partner_invoice:
            raise ValueError(f"Facture partenaire {partner_invoice_id} introuvable")
        if partner_invoice.executing_company_id != company_id:
            raise ValueError(
                "Seule l'entreprise exécutante peut marquer cette facture comme envoyée"
            )
        if partner_invoice.status != PartnerInvoiceStatus.DRAFT:
            raise ValueError(
                f"Déjà envoyée ou statut invalide (statut actuel: {partner_invoice.status})"
            )
        partner_invoice.status = PartnerInvoiceStatus.SENT
        partner_invoice.sent_at = datetime.now(UTC)
        db.session.commit()
        logger.info(
            "Facture partenaire %s marquée comme envoyée (company_id=%s)",
            partner_invoice_id,
            company_id,
        )
        return partner_invoice

    def mark_as_paid(self, partner_invoice_id: int) -> PartnerInvoice:
        """Marque une facture partenaire comme payée.

        Args:
            partner_invoice_id: ID de la facture

        Returns:
            PartnerInvoice mise à jour
        """
        partner_invoice = PartnerInvoice.query.get(partner_invoice_id)
        if not partner_invoice:
            raise ValueError(f"Facture partenaire {partner_invoice_id} introuvable")

        partner_invoice.status = PartnerInvoiceStatus.PAID
        partner_invoice.paid_at = datetime.now(UTC)

        db.session.commit()

        return partner_invoice

    def get_monthly_invoice(
        self, partnership_id: int, year: int, month: int
    ) -> PartnerInvoice | None:
        """Récupère la facture mensuelle d'un partenariat.

        Args:
            partnership_id: ID du partenariat
            year: Année
            month: Mois

        Returns:
            PartnerInvoice ou None
        """
        return PartnerInvoice.query.filter_by(
            partnership_id=partnership_id, period_year=year, period_month=month
        ).first()

    def get_pending_transfers_count(
        self, partnership_id: int, year: int, month: int
    ) -> int:
        """Compte les transferts validés non facturés pour une période.

        Args:
            partnership_id: ID du partenariat
            year: Année
            month: Mois

        Returns:
            Nombre de transferts en attente de facturation
        """
        DECEMBER = 12
        start_date = datetime(year, month, 1, tzinfo=UTC)
        if month == DECEMBER:
            end_date = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=UTC)

        # Transferts validés dans la période
        validated_transfers = (
            BookingTransfer.query.filter_by(
                partnership_id=partnership_id,
                status=TransferStatus.COMPLETED,
                is_validated=True,
            )
            .filter(
                BookingTransfer.validated_at >= start_date,
                BookingTransfer.validated_at < end_date,
            )
            .all()
        )

        # Exclure ceux déjà facturés
        factured_transfer_ids = db.session.query(
            partner_invoice_transfers.c.booking_transfer_id
        ).all()
        factured_ids = {row[0] for row in factured_transfer_ids}

        pending = [t for t in validated_transfers if t.id not in factured_ids]

        return len(pending)

    def get_pending_amount(self, partnership_id: int, year: int, month: int) -> Decimal:
        """Calcule le montant total des transferts en attente de facturation.

        Args:
            partnership_id: ID du partenariat
            year: Année
            month: Mois

        Returns:
            Montant total en attente
        """
        DECEMBER = 12
        start_date = datetime(year, month, 1, tzinfo=UTC)
        if month == DECEMBER:
            end_date = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=UTC)

        validated_transfers = (
            BookingTransfer.query.filter_by(
                partnership_id=partnership_id,
                status=TransferStatus.COMPLETED,
                is_validated=True,
            )
            .filter(
                BookingTransfer.validated_at >= start_date,
                BookingTransfer.validated_at < end_date,
            )
            .all()
        )

        factured_transfer_ids = db.session.query(
            partner_invoice_transfers.c.booking_transfer_id
        ).all()
        factured_ids = {row[0] for row in factured_transfer_ids}

        total = Decimal("0")
        for transfer in validated_transfers:
            if transfer.id not in factured_ids and transfer.partner_cost:
                total += transfer.partner_cost

        return total
