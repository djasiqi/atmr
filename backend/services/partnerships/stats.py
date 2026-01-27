# services/partnership_stats_service.py
"""Service pour calculer les statistiques de partenariats."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, or_

from ext import db
from models.booking_transfer import BookingTransfer
from models.enums import PartnershipStatus, TransferStatus
from models.partner_invoice import PartnerInvoice, PartnerInvoiceStatus
from models.partnership import Partnership
from services.partnerships.exceptions import StatsComputationError

logger = logging.getLogger(__name__)


class PartnershipStatsService:
    """Service pour calculer les statistiques de partenariats."""

    @staticmethod
    def get_global_stats(
        company_id: int, month: int | None = None, year: int | None = None
    ) -> dict[str, Any]:
        """Calcule les statistiques globales de partenariats pour une entreprise.

        Args:
            company_id: ID de l'entreprise
            month: Mois (1-12), None pour mois en cours
            year: Année, None pour année en cours

        Returns:
            Dictionnaire avec les statistiques globales
        """
        now = datetime.now(UTC)
        if month is None:
            month = now.month
        if year is None:
            year = now.year

        # Dates du mois
        MONTHS_IN_YEAR = 12
        start_of_month = datetime(year, month, 1, tzinfo=UTC)
        if month == MONTHS_IN_YEAR:
            end_of_month = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end_of_month = datetime(year, month + 1, 1, tzinfo=UTC)

        # Partenaires actifs
        active_partnerships = (
            db.session.query(func.count(Partnership.id))
            .filter(
                or_(
                    Partnership.owner_company_id == company_id,
                    Partnership.partner_company_id == company_id,
                ),
                Partnership.status == PartnershipStatus.ACCEPTED,
                Partnership.is_active.is_(True),
            )
            .scalar()
            or 0
        )

        # Courses envoyées (où l'entreprise est propriétaire)
        sent_transfers = (
            db.session.query(func.count(BookingTransfer.id))
            .join(Partnership)
            .filter(
                BookingTransfer.owner_company_id == company_id,
                BookingTransfer.requested_at >= start_of_month,
                BookingTransfer.requested_at < end_of_month,
            )
            .scalar()
            or 0
        )

        # Courses reçues (où l'entreprise est exécutante)
        # Utiliser requested_at au lieu de accepted_at pour être cohérent avec les courses envoyées
        # Si A envoie une course à B, B doit la voir comme reçue dès la demande (requested_at)
        received_transfers = (
            db.session.query(func.count(BookingTransfer.id))
            .join(Partnership)
            .filter(
                BookingTransfer.executing_company_id == company_id,
                BookingTransfer.requested_at >= start_of_month,
                BookingTransfer.requested_at < end_of_month,
            )
            .scalar()
            or 0
        )

        # Montant à payer aux partenaires (factures où l'autre entreprise a émis la facture)
        # Si executing_company_id != company_id, alors l'entreprise actuelle doit payer
        amount_to_pay = db.session.query(
            func.coalesce(func.sum(PartnerInvoice.total_amount), 0)
        ).join(Partnership).filter(
            or_(
                Partnership.owner_company_id == company_id,
                Partnership.partner_company_id == company_id,
            ),
            PartnerInvoice.executing_company_id != company_id,
            PartnerInvoice.status.in_(
                [PartnerInvoiceStatus.SENT, PartnerInvoiceStatus.DRAFT]
            ),
        ).scalar() or Decimal("0")

        # Montant à recevoir des partenaires (factures où l'entreprise actuelle a émis la facture)
        # Si executing_company_id == company_id, alors l'entreprise actuelle doit recevoir
        amount_to_receive = db.session.query(
            func.coalesce(func.sum(PartnerInvoice.total_amount), 0)
        ).join(Partnership).filter(
            or_(
                Partnership.owner_company_id == company_id,
                Partnership.partner_company_id == company_id,
            ),
            PartnerInvoice.executing_company_id == company_id,
            PartnerInvoice.status.in_(
                [PartnerInvoiceStatus.SENT, PartnerInvoiceStatus.DRAFT]
            ),
        ).scalar() or Decimal("0")

        # Solde net
        net_balance = float(amount_to_receive - amount_to_pay)

        return {
            "active_partnerships": active_partnerships,
            "sent_transfers_current_month": sent_transfers,
            "received_transfers_current_month": received_transfers,
            "amount_to_pay": float(amount_to_pay),
            "amount_to_receive": float(amount_to_receive),
            "net_balance": net_balance,
            "period": {"year": year, "month": month},
        }

    @staticmethod
    def get_partnership_stats(
        partnership: Partnership,
        company_id: int,
        month: int | None = None,
        year: int | None = None,
    ) -> dict[str, Any]:
        """Calcule les statistiques pour un partenariat spécifique.

        Args:
            partnership: Le partenariat
            company_id: ID de l'entreprise actuelle
            month: Mois (1-12), None pour mois en cours
            year: Année, None pour année en cours

        Returns:
            Dictionnaire avec les statistiques du partenariat

        Raises:
            StatsComputationError: Si le calcul échoue (champ manquant, erreur SQL, etc.)
        """
        try:
            return PartnershipStatsService._get_partnership_stats_impl(
                partnership, company_id, month, year
            )
        except Exception as exc:
            logger.exception(
                "Erreur lors du calcul des stats de partenariat",
                extra={"partnership_id": partnership.id, "company_id": company_id},
            )
            raise StatsComputationError(
                "Impossible de calculer les statistiques du partenariat"
            ) from exc

    @staticmethod
    def _get_partnership_stats_impl(
        partnership: Partnership,
        company_id: int,
        month: int | None,
        year: int | None,
    ) -> dict[str, Any]:
        """Implémentation du calcul (séparée pour permettre le try/except au niveau public)."""
        now = datetime.now(UTC)
        if month is None:
            month = now.month
        if year is None:
            year = now.year

        # Dates du mois
        MONTHS_IN_YEAR = 12
        start_of_month = datetime(year, month, 1, tzinfo=UTC)
        if month == MONTHS_IN_YEAR:
            end_of_month = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            end_of_month = datetime(year, month + 1, 1, tzinfo=UTC)

        # ✅ Note: is_owner sera déterminé APRÈS le calcul des sent_transfers/received_transfers

        # Courses envoyées (où l'entreprise actuelle est propriétaire)
        # Utiliser requested_at pour être cohérent avec les courses reçues
        sent_transfers = (
            db.session.query(func.count(BookingTransfer.id))
            .filter(
                BookingTransfer.partnership_id == partnership.id,
                BookingTransfer.owner_company_id == company_id,
                BookingTransfer.requested_at >= start_of_month,
                BookingTransfer.requested_at < end_of_month,
            )
            .scalar()
            or 0
        )

        # Courses reçues (où l'entreprise actuelle est exécutante)
        # Utiliser requested_at au lieu de accepted_at pour être cohérent avec les courses envoyées
        # Si A envoie une course à B, B doit la voir comme reçue dès la demande (requested_at)
        received_transfers = (
            db.session.query(func.count(BookingTransfer.id))
            .filter(
                BookingTransfer.partnership_id == partnership.id,
                BookingTransfer.executing_company_id == company_id,
                BookingTransfer.requested_at >= start_of_month,
                BookingTransfer.requested_at < end_of_month,
            )
            .scalar()
            or 0
        )

        # ✅ Déterminer le rôle réel basé sur les transferts
        # Si on a ENVOYÉ des courses, on est ÉMETTEUR (on doit payer)
        # Si on a REÇU des courses, on est EXÉCUTEUR (on doit recevoir)
        is_owner = sent_transfers > 0  # True si on a envoyé des courses (émetteur)

        # CA généré (somme des client_price des transferts acceptés/complétés)
        total_revenue = db.session.query(
            func.coalesce(func.sum(BookingTransfer.client_price), 0)
        ).filter(
            BookingTransfer.partnership_id == partnership.id,
            BookingTransfer.status.in_(
                [TransferStatus.ACCEPTED, TransferStatus.COMPLETED]
            ),
            BookingTransfer.requested_at >= start_of_month,
            BookingTransfer.requested_at < end_of_month,
        ).scalar() or Decimal("0")

        # À payer (factures reçues non payées où l'entreprise est propriétaire)
        # OU montant estimé basé sur les transfer_price si aucune facture générée
        invoiced_to_pay = db.session.query(
            func.coalesce(func.sum(PartnerInvoice.total_amount), 0)
        ).filter(
            PartnerInvoice.partnership_id == partnership.id,
            PartnerInvoice.status.in_(
                [PartnerInvoiceStatus.SENT, PartnerInvoiceStatus.DRAFT]
            ),
        ).scalar() or Decimal("0")

        # Si l'entreprise est owner (émetteur) et qu'aucune facture n'a été générée,
        # calculer un montant estimé basé sur les partner_cost des transferts complétés
        # (double coalesce : partner_cost nullable, SUM(NULL) peut rester NULL selon la DB)
        if is_owner and invoiced_to_pay == 0:
            estimated_to_pay = db.session.query(
                func.coalesce(func.sum(func.coalesce(BookingTransfer.partner_cost, 0)), 0)
            ).filter(
                BookingTransfer.partnership_id == partnership.id,
                BookingTransfer.owner_company_id == company_id,
                BookingTransfer.status.in_(
                    [TransferStatus.ACCEPTED, TransferStatus.COMPLETED]
                ),
                BookingTransfer.requested_at >= start_of_month,
                BookingTransfer.requested_at < end_of_month,
            ).scalar() or Decimal("0")
            logger.info(
                "📊 [Partnership Stats] company_id=%s, is_owner=%s, estimated_to_pay=%s",
                company_id,
                is_owner,
                estimated_to_pay,
            )
            amount_to_pay = estimated_to_pay
        else:
            amount_to_pay = invoiced_to_pay if is_owner else Decimal("0")
            logger.info(
                "📊 [Partnership Stats] company_id=%s, is_owner=%s, invoiced_to_pay=%s, amount_to_pay=%s",
                company_id,
                is_owner,
                invoiced_to_pay,
                amount_to_pay,
            )

        # À recevoir (factures émises non payées où l'entreprise est partenaire)
        # OU montant estimé basé sur les partner_cost si aucune facture générée
        invoiced_to_receive = db.session.query(
            func.coalesce(func.sum(PartnerInvoice.total_amount), 0)
        ).filter(
            PartnerInvoice.partnership_id == partnership.id,
            PartnerInvoice.status.in_(
                [PartnerInvoiceStatus.SENT, PartnerInvoiceStatus.DRAFT]
            ),
        ).scalar() or Decimal("0")

        # Si l'entreprise est partner (exécuteur) et qu'aucune facture n'a été générée,
        # calculer un montant estimé basé sur les partner_cost des transferts complétés
        # (double coalesce : partner_cost nullable, SUM(NULL) peut rester NULL selon la DB)
        if not is_owner and invoiced_to_receive == 0:
            estimated_to_receive = db.session.query(
                func.coalesce(func.sum(func.coalesce(BookingTransfer.partner_cost, 0)), 0)
            ).filter(
                BookingTransfer.partnership_id == partnership.id,
                BookingTransfer.executing_company_id == company_id,
                BookingTransfer.status.in_(
                    [TransferStatus.ACCEPTED, TransferStatus.COMPLETED]
                ),
                BookingTransfer.requested_at >= start_of_month,
                BookingTransfer.requested_at < end_of_month,
            ).scalar() or Decimal("0")
            logger.info(
                "📊 [Partnership Stats] company_id=%s, is_owner=%s, estimated_to_receive=%s",
                company_id,
                is_owner,
                estimated_to_receive,
            )
            amount_to_receive = estimated_to_receive
        else:
            amount_to_receive = invoiced_to_receive if not is_owner else Decimal("0")
            logger.info(
                "📊 [Partnership Stats] company_id=%s, is_owner=%s, invoiced_to_receive=%s, amount_to_receive=%s",
                company_id,
                is_owner,
                invoiced_to_receive,
                amount_to_receive,
            )

        # Solde
        balance = float(amount_to_receive - amount_to_pay)

        return {
            "sent_transfers": sent_transfers,
            "received_transfers": received_transfers,
            "total_revenue": float(total_revenue),
            "amount_to_pay": float(amount_to_pay),
            "amount_to_receive": float(amount_to_receive),
            "balance": balance,
            "period": {"year": year, "month": month},
        }
