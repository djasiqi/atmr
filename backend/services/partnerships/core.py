# services/partnership_service.py
"""Service pour gérer les partenariats entre entreprises."""

from ext import db
from models.company import Company
from models.enums import PartnershipStatus, TransferModel
from models.partnership import Partnership


class PartnershipService:
    """Service pour la gestion des partenariats."""

    @staticmethod
    def create_partnership(
        owner_company_id: int,
        partner_company_id: int,
        default_transfer_model: TransferModel = TransferModel.SUBCONTRACT,
        default_margin_percent: float | None = None,
        default_partner_tariff_percent: float | None = None,
        auto_accept: bool = False,
        auto_invoice: bool = True,
        payment_terms_days: int = 30,
    ) -> Partnership:
        """Créer un nouveau partenariat entre deux entreprises.

        Args:
            owner_company_id: ID de l'entreprise propriétaire
            partner_company_id: ID de l'entreprise partenaire
            default_transfer_model: Modèle de transfert par défaut
            default_margin_percent: Marge que l'entreprise propriétaire garde (ex: 20%)
            default_partner_tariff_percent: % du prix client pour le partenaire (ex: 80%)
            auto_accept: Auto-acceptation des transferts
            auto_invoice: Facturation automatique
            payment_terms_days: Délai de paiement en jours

        Returns:
            Partnership créé

        Raises:
            ValueError: Si les entreprises sont identiques ou si le partenariat existe déjà
        """
        if owner_company_id == partner_company_id:
            raise ValueError("Une entreprise ne peut pas être partenaire d'elle-même")

        # Vérifier si une demande existe déjà (pending ou accepted)
        existing = (
            Partnership.query.filter_by(
                owner_company_id=owner_company_id,
                partner_company_id=partner_company_id,
            )
            .filter(
                Partnership.status.in_(
                    [PartnershipStatus.PENDING, PartnershipStatus.ACCEPTED]
                )
            )
            .first()
        )

        if existing:
            if existing.status == PartnershipStatus.PENDING:
                raise ValueError("Une demande de partenariat est déjà en attente")
            raise ValueError("Ce partenariat existe déjà")

        # Vérifier que les entreprises existent
        owner_company = Company.query.get(owner_company_id)
        if not owner_company:
            raise ValueError(f"Entreprise propriétaire {owner_company_id} introuvable")

        partner_company = Company.query.get(partner_company_id)
        if not partner_company:
            raise ValueError(f"Entreprise partenaire {partner_company_id} introuvable")

        # SQLAlchemy 2.0 avec Mapped nécessite d'assigner les attributs après création
        partnership = Partnership()
        partnership.owner_company_id = owner_company_id
        partnership.partner_company_id = partner_company_id
        partnership.default_transfer_model = default_transfer_model
        from decimal import Decimal

        partnership.default_margin_percent = (
            Decimal(str(default_margin_percent))
            if default_margin_percent is not None
            else None
        )
        partnership.default_partner_tariff_percent = (
            Decimal(str(default_partner_tariff_percent))
            if default_partner_tariff_percent is not None
            else None
        )
        partnership.auto_accept_rules = auto_accept
        partnership.auto_invoice = auto_invoice
        partnership.payment_terms_days = payment_terms_days
        partnership.status = PartnershipStatus.PENDING  # Demande en attente

        db.session.add(partnership)
        db.session.commit()
        return partnership

    @staticmethod
    def get_partnerships_for_company(company_id: int) -> list[Partnership]:
        """Récupérer tous les partenariats d'une entreprise (en tant que propriétaire ou partenaire).

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste des partenariats actifs
        """
        return (
            Partnership.query.filter(
                (Partnership.owner_company_id == company_id)
                | (Partnership.partner_company_id == company_id)
            )
            .filter_by(status=PartnershipStatus.ACCEPTED, is_active=True)
            .all()
        )

    @staticmethod
    def get_owner_partnerships_for_company(company_id: int) -> list[Partnership]:
        """Récupérer les partenariats où l'entreprise est propriétaire (pour transfert de courses).

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste des partenariats actifs où l'entreprise est propriétaire
        """
        return Partnership.query.filter_by(
            owner_company_id=company_id,
            status=PartnershipStatus.ACCEPTED,
            is_active=True,
        ).all()

    @staticmethod
    def get_partnership(
        owner_company_id: int, partner_company_id: int
    ) -> Partnership | None:
        """Récupérer un partenariat spécifique.

        Args:
            owner_company_id: ID de l'entreprise propriétaire
            partner_company_id: ID de l'entreprise partenaire

        Returns:
            Partnership ou None si non trouvé
        """
        return Partnership.query.filter_by(
            owner_company_id=owner_company_id,
            partner_company_id=partner_company_id,
            is_active=True,
        ).first()

    @staticmethod
    def get_partnership_by_id(partnership_id: int) -> Partnership | None:
        """Récupérer un partenariat par son ID.

        Args:
            partnership_id: ID du partenariat

        Returns:
            Partnership ou None si non trouvé
        """
        return Partnership.query.get(partnership_id)

    @staticmethod
    def update_partnership(
        partnership_id: int,
        default_transfer_model: TransferModel | None = None,
        default_margin_percent: float | None = None,
        default_partner_tariff_percent: float | None = None,
        auto_accept: bool | None = None,
        auto_invoice: bool | None = None,
        payment_terms_days: int | None = None,
        status: PartnershipStatus | None = None,
    ) -> Partnership:
        """Mettre à jour un partenariat.

        Args:
            partnership_id: ID du partenariat
            default_transfer_model: Nouveau modèle de transfert
            default_margin_percent: Nouvelle marge
            default_partner_tariff_percent: Nouveau tarif partenaire
            auto_accept: Nouvelle règle d'auto-acceptation
            auto_invoice: Nouvelle règle de facturation automatique
            payment_terms_days: Nouveau délai de paiement

        Returns:
            Partnership mis à jour

        Raises:
            ValueError: Si le partenariat n'existe pas
        """
        partnership = Partnership.query.get(partnership_id)
        if not partnership:
            raise ValueError(f"Partenariat {partnership_id} introuvable")

        if default_transfer_model is not None:
            partnership.default_transfer_model = default_transfer_model
        if default_margin_percent is not None:
            partnership.default_margin_percent = default_margin_percent
        if default_partner_tariff_percent is not None:
            partnership.default_partner_tariff_percent = default_partner_tariff_percent
        if auto_accept is not None:
            partnership.auto_accept_rules = auto_accept
        if auto_invoice is not None:
            partnership.auto_invoice = auto_invoice
        if payment_terms_days is not None:
            partnership.payment_terms_days = payment_terms_days
        if status is not None:
            partnership.status = status

        db.session.commit()
        return partnership

    @staticmethod
    def deactivate_partnership(partnership_id: int) -> Partnership:
        """Désactiver un partenariat.

        Args:
            partnership_id: ID du partenariat

        Returns:
            Partnership désactivé

        Raises:
            ValueError: Si le partenariat n'existe pas
        """
        partnership = Partnership.query.get(partnership_id)
        if not partnership:
            raise ValueError(f"Partenariat {partnership_id} introuvable")

        partnership.is_active = False
        db.session.commit()
        return partnership

    @staticmethod
    def delete_partnership(partnership_id: int, company_id: int) -> None:
        """Supprime complètement un partenariat.

        Args:
            partnership_id: ID du partenariat
            company_id: ID de l'entreprise qui demande la suppression

        Raises:
            ValueError: Si le partenariat n'existe pas ou si l'entreprise n'est pas autorisée
        """
        import logging

        logger = logging.getLogger(__name__)

        partnership = Partnership.query.get(partnership_id)
        if not partnership:
            logger.warning(
                "Tentative de suppression d'un partenariat inexistant: %s",
                partnership_id,
            )
            raise ValueError(f"Partenariat {partnership_id} introuvable")

        # Vérifier que l'entreprise est liée au partenariat
        if company_id not in {
            partnership.owner_company_id,
            partnership.partner_company_id,
        }:
            logger.warning(
                "Tentative de suppression non autorisée: company_id=%s, partnership_id=%s",
                company_id,
                partnership_id,
            )
            raise ValueError("Vous n'êtes pas autorisé à supprimer ce partenariat")

        try:
            # Vérifier s'il y a des transferts ou factures liés
            from models.booking_transfer import BookingTransfer
            from models.partner_invoice import PartnerInvoice

            transfers_count = BookingTransfer.query.filter_by(
                partnership_id=partnership_id
            ).count()
            invoices_count = PartnerInvoice.query.filter_by(
                partnership_id=partnership_id
            ).count()

            if transfers_count > 0 or invoices_count > 0:
                logger.warning(
                    (
                        "Tentative de suppression d'un partenariat avec des dépendances: "
                        "partnership_id=%s, transfers=%s, invoices=%s"
                    ),
                    partnership_id,
                    transfers_count,
                    invoices_count,
                )
                # Note: Avec ondelete="CASCADE", ces enregistrements seront supprimés automatiquement
                # mais on log quand même pour information

            # Supprimer le partenariat de la base de données
            db.session.delete(partnership)
            db.session.commit()
            logger.info(
                "Partenariat %s supprimé avec succès par company_id=%s",
                partnership_id,
                company_id,
            )
        except Exception as e:
            db.session.rollback()
            logger.exception(
                "Erreur lors de la suppression du partenariat %s: %s", partnership_id, e
            )
            raise

    @staticmethod
    def accept_partnership_request(partnership_id: int, company_id: int) -> Partnership:
        """Accepter une demande de partenariat.

        Args:
            partnership_id: ID du partenariat
            company_id: ID de l'entreprise qui accepte (doit être le partenaire)

        Returns:
            Partnership accepté

        Raises:
            ValueError: Si le partenariat n'existe pas ou si l'entreprise n'est pas autorisée
        """
        partnership = Partnership.query.get(partnership_id)
        if not partnership:
            raise ValueError(f"Partenariat {partnership_id} introuvable")

        if partnership.partner_company_id != company_id:
            raise ValueError(
                "Seule l'entreprise partenaire peut accepter cette demande"
            )

        if partnership.status != PartnershipStatus.PENDING:
            raise ValueError("Cette demande ne peut plus être acceptée")

        partnership.status = PartnershipStatus.ACCEPTED
        partnership.is_active = True
        db.session.commit()
        return partnership

    @staticmethod
    def reject_partnership_request(partnership_id: int, company_id: int) -> Partnership:
        """Refuser une demande de partenariat.

        Args:
            partnership_id: ID du partenariat
            company_id: ID de l'entreprise qui refuse (doit être le partenaire)

        Returns:
            Partnership refusé

        Raises:
            ValueError: Si le partenariat n'existe pas ou si l'entreprise n'est pas autorisée
        """
        partnership = Partnership.query.get(partnership_id)
        if not partnership:
            raise ValueError(f"Partenariat {partnership_id} introuvable")

        if partnership.partner_company_id != company_id:
            raise ValueError("Seule l'entreprise partenaire peut refuser cette demande")

        if partnership.status != PartnershipStatus.PENDING:
            raise ValueError("Cette demande ne peut plus être refusée")

        partnership.status = PartnershipStatus.REJECTED
        partnership.is_active = False
        db.session.commit()
        return partnership

    @staticmethod
    def get_pending_requests_for_company(company_id: int) -> list[Partnership]:
        """Récupérer les demandes de partenariat en attente pour une entreprise.

        Args:
            company_id: ID de l'entreprise (en tant que partenaire)

        Returns:
            Liste des demandes en attente
        """
        return Partnership.query.filter_by(
            partner_company_id=company_id,
            status=PartnershipStatus.PENDING,
        ).all()
