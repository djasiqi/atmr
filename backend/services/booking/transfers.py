# services/booking_transfer_service.py
"""Service pour gérer les transferts de courses à des partenaires."""

import logging
from datetime import UTC, datetime
from decimal import Decimal

from ext import db
from models.booking import Booking
from models.booking_transfer import BookingTransfer
from models.enums import BookingStatus, TransferModel, TransferStatus
from models.partnership import Partnership
from services.realtime.socketio import emit_company_event

logger = logging.getLogger(__name__)


class BookingTransferService:
    """Service pour la gestion des transferts de courses."""

    @staticmethod
    def propose_transfer(
        booking_id: int,
        partnership_id: int,
        transfer_model: TransferModel | None = None,
    ) -> BookingTransfer:
        """Proposer une course à un partenaire.

        Args:
            booking_id: ID de la course
            partnership_id: ID du partenariat
            transfer_model: Modèle de transfert (optionnel, utilise celui du partenariat)

        Returns:
            BookingTransfer créé

        Raises:
            ValueError: Si la course n'appartient pas à l'entreprise propriétaire
        """
        # ✅ FIX CRITIQUE: Convertir explicitement booking_id en int pour éviter
        # l'erreur SQL "operator does not exist: integer = character varying"
        # Les annotations de type Python ne sont pas enforçées à l'exécution
        try:
            booking_id = int(booking_id)
            partnership_id = int(partnership_id)
            # #region agent log
            try:
                import json
                from pathlib import Path

                log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                                "location": "transfers.py:44",
                                "message": "Service: booking_id and partnership_id converted",
                                "data": {
                                    "booking_id": booking_id,
                                    "booking_id_type": type(booking_id).__name__,
                                    "partnership_id": partnership_id,
                                    "partnership_id_type": type(
                                        partnership_id
                                    ).__name__,
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        except (ValueError, TypeError) as e:
            # #region agent log
            try:
                import json
                from pathlib import Path

                log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "sessionId": "debug-session",
                                "runId": "run1",
                                "hypothesisId": "H3",
                                "location": "transfers.py:46",
                                "message": "Service: conversion FAILED",
                                "data": {
                                    "booking_id_raw": str(booking_id),
                                    "partnership_id_raw": str(partnership_id),
                                    "error": str(e),
                                },
                                "timestamp": int(__import__("time").time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            raise ValueError(
                f"booking_id et partnership_id doivent être des entiers valides: {e}"
            ) from e

        booking = Booking.query.get_or_404(booking_id)
        partnership = Partnership.query.get_or_404(partnership_id)

        # Vérifier que la course appartient à l'entreprise (propriétaire ou partenaire)
        # Une entreprise peut transférer des courses si elle est propriétaire OU partenaire
        # (car elle peut vouloir transférer des courses qu'elle a reçues d'un partenaire)
        if booking.company_id not in (
            partnership.owner_company_id,
            partnership.partner_company_id,
        ):
            raise ValueError("La course n'appartient pas à l'entreprise du partenariat")

        # Vérifier que le partenariat est actif
        if not partnership.is_active:
            raise ValueError("Le partenariat n'est pas actif")

        # Vérifier que la course n'est pas déjà transférée
        # #region agent log
        try:
            import json
            from pathlib import Path

            log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H3",
                            "location": "transfers.py:68",
                            "message": "Before filter_by query",
                            "data": {
                                "booking_id": booking_id,
                                "booking_id_type": type(booking_id).__name__,
                            },
                            "timestamp": int(__import__("time").time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        existing_transfer = (
            BookingTransfer.query.filter_by(booking_id=booking_id)
            .filter(
                BookingTransfer.status.in_(
                    [TransferStatus.PENDING, TransferStatus.ACCEPTED]
                )
            )
            .first()
        )
        # #region agent log
        try:
            import json
            from pathlib import Path

            log_path = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "H3",
                            "location": "transfers.py:76",
                            "message": "After filter_by query",
                            "data": {"has_existing_transfer": bool(existing_transfer)},
                            "timestamp": int(__import__("time").time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        if existing_transfer:
            raise ValueError("Cette course est déjà en cours de transfert")

        # Vérifier que la course n'est pas déjà assignée à une autre entreprise
        if booking.executing_company_id and booking.executing_company_id not in (
            partnership.owner_company_id,
            partnership.partner_company_id,
        ):
            raise ValueError("Cette course est déjà assignée à une autre entreprise")

        # Vérifier que la course est dans un statut approprié
        if booking.status not in [
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ]:
            raise ValueError(
                f"La course doit être en statut PENDING, ACCEPTED ou ASSIGNED pour être transférée (actuel: {booking.status.value})"
            )

        # Utiliser le modèle de transfert du partenariat si non spécifié
        if transfer_model is None:
            transfer_model = partnership.default_transfer_model

        # Calculer le coût partenaire selon le partenariat
        # Utiliser le pourcentage défini lors de la demande de partenariat
        client_price = Decimal(str(booking.amount))

        if partnership.default_partner_tariff_percent:
            # Ex: 80% du montant client pour le partenaire
            partner_cost = client_price * (
                partnership.default_partner_tariff_percent / Decimal("100")
            )
        elif partnership.default_margin_percent:
            # Ex: 20% de marge pour A, donc 80% pour B
            partner_cost = client_price * (
                1 - partnership.default_margin_percent / Decimal("100")
            )
        else:
            # Fallback: 90% par défaut si rien n'est configuré
            partner_cost = client_price * Decimal("0.9")

        platform_fee = Decimal("0")

        # Révoquer le chauffeur et réinitialiser le statut lors du transfert
        # La course passe à PENDING sans chauffeur assigné
        # ✅ CORRECTION : Changer le statut AVANT de mettre driver_id à None
        # pour éviter l'erreur de validation "driver_id ne peut pas être NULL si status=ASSIGNED"
        original_driver_id = booking.driver_id
        original_status = booking.status
        # Changer le statut d'abord (pour éviter la validation qui vérifie driver_id si status=ASSIGNED)
        booking.status = BookingStatus.PENDING
        # Ensuite mettre driver_id à None (maintenant que le statut n'est plus ASSIGNED)
        booking.driver_id = None

        # #region agent log
        try:
            import json
            from pathlib import Path

            with Path(r"c:\Users\jasiq\atmr\.cursor\debug.log").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "D",
                            "location": "booking_transfer_service.py:propose_transfer",
                            "message": "Transfer proposed, driver revoked",
                            "data": {
                                "booking_id": booking_id,
                                "partnership_id": partnership_id,
                                "original_driver_id": original_driver_id,
                                "original_status": original_status.value
                                if hasattr(original_status, "value")
                                else str(original_status),
                                "new_status": booking.status.value,
                                "new_driver_id": booking.driver_id,
                            },
                            "timestamp": int(__import__("time").time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

        # Créer le transfert
        # SQLAlchemy 2.0 avec Mapped nécessite d'assigner les attributs après création
        # Déterminer quelle entreprise est propriétaire et quelle est partenaire dans le contexte du transfert
        # L'entreprise propriétaire est celle qui possède la course (booking.company_id)
        # L'entreprise partenaire est l'autre entreprise du partenariat
        if booking.company_id == partnership.owner_company_id:
            # L'entreprise actuelle est propriétaire, elle transfère à la partenaire
            transfer_owner_company_id = partnership.owner_company_id
            transfer_executing_company_id = partnership.partner_company_id
        else:
            # L'entreprise actuelle est partenaire, elle transfère à la propriétaire
            transfer_owner_company_id = partnership.partner_company_id
            transfer_executing_company_id = partnership.owner_company_id

        transfer = BookingTransfer()
        transfer.booking_id = booking_id
        transfer.partnership_id = partnership_id
        transfer.transfer_model = transfer_model
        transfer.owner_company_id = transfer_owner_company_id
        transfer.executing_company_id = transfer_executing_company_id
        transfer.client_price = client_price
        transfer.partner_cost = partner_cost
        transfer.platform_fee = platform_fee
        transfer.currency = "CHF"  # TODO: Récupérer depuis booking ou company
        transfer.status = TransferStatus.PENDING

        db.session.add(transfer)

        # Définir executing_company_id dès maintenant pour que l'entreprise partenaire
        # puisse voir la course dans son dashboard (même si le transfert n'est pas encore accepté)
        # Le statut reste PENDING jusqu'à acceptation du transfert
        booking.executing_company_id = transfer_executing_company_id
        # ✅ Le driver_id a déjà été mis à None plus haut (ligne 236) après le changement de statut
        # Pas besoin de le refaire ici pour éviter les problèmes de validation

        logger.info(
            (
                "[BookingTransferService] Transfer created: booking_id=%s, "
                "executing_company_id=%s, booking.status=%s, transfer.status=%s"
            ),
            booking_id,
            booking.executing_company_id,
            booking.status.value
            if hasattr(booking.status, "value")
            else str(booking.status),
            transfer.status.value
            if hasattr(transfer.status, "value")
            else str(transfer.status),
        )

        # Auto-acceptation si configurée
        if partnership.auto_accept_rules:
            transfer.status = TransferStatus.ACCEPTED
            transfer.accepted_at = datetime.now(UTC)
            # Changer le statut de la course à ACCEPTED (elle est déjà PENDING après révoquation du chauffeur)
            booking.status = BookingStatus.ACCEPTED
            logger.info(
                "[BookingTransferService] Auto-accept enabled: booking.status changed to ACCEPTED"
            )

        db.session.commit()

        # Recharger le booking depuis la DB pour vérifier que les changements sont bien persistés
        db.session.refresh(booking)
        db.session.refresh(transfer)

        logger.info(
            (
                "[BookingTransferService] After commit and refresh: "
                "booking.id=%s, booking.company_id=%s, booking.executing_company_id=%s, "
                "booking.status=%s, transfer.status=%s"
            ),
            booking.id,
            booking.company_id,
            booking.executing_company_id,
            booking.status.value
            if hasattr(booking.status, "value")
            else str(booking.status),
            transfer.status.value
            if hasattr(transfer.status, "value")
            else str(transfer.status),
        )

        # Notifier les entreprises concernées via Socket.IO
        try:
            transfer_dict = transfer.to_dict()
            # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence
            # Notifier l'entreprise propriétaire (celle qui possède la course)
            emit_company_event(
                transfer_owner_company_id,
                "transfer_proposed",
                {
                    "transfer": transfer_dict,
                    "booking_id": booking_id,
                    "partnership_id": partnership_id,
                },
            )
            # Notifier l'entreprise partenaire (celle qui va exécuter la course)
            emit_company_event(
                transfer_executing_company_id,
                "transfer_received",
                {
                    "transfer": transfer_dict,
                    "booking_id": booking_id,
                    "partnership_id": partnership_id,
                },
            )
        except Exception as e:
            logger.warning("Erreur lors de l'envoi des notifications Socket.IO: %s", e)

        return transfer

    @staticmethod
    def accept_transfer(transfer_id: int, executing_company_id: int) -> BookingTransfer:
        """Accepter un transfert.

        Args:
            transfer_id: ID du transfert
            executing_company_id: ID de l'entreprise qui accepte

        Returns:
            BookingTransfer accepté

        Raises:
            ValueError: Si l'entreprise n'est pas autorisée ou si le transfert ne peut plus être accepté
        """
        # ✅ FIX CRITIQUE: Convertir explicitement les IDs en int
        try:
            transfer_id = int(transfer_id)
            executing_company_id = int(executing_company_id)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"transfer_id et executing_company_id doivent être des entiers valides: {e}"
            ) from e

        transfer = BookingTransfer.query.get_or_404(transfer_id)

        if transfer.executing_company_id != executing_company_id:
            raise ValueError(
                "Cette entreprise n'est pas autorisée à accepter ce transfert"
            )

        if transfer.status != TransferStatus.PENDING:
            raise ValueError("Ce transfert ne peut plus être accepté")

        # Vérifier que la course est toujours dans un statut approprié
        if transfer.booking.status not in [
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ]:
            raise ValueError(
                f"La course doit être en statut PENDING, ACCEPTED ou ASSIGNED pour être acceptée (actuel: {transfer.booking.status.value})"
            )

        transfer.status = TransferStatus.ACCEPTED
        transfer.accepted_at = datetime.now(UTC)

        # Modèle Owner vs Executor : company_id reste l'owner (facture client), executing_company_id = exécutant (facture partenaire).
        transfer.booking.executing_company_id = transfer.executing_company_id
        # Ne pas changer booking.company_id : il doit rester l'owner (A) pour que A continue à voir la course et facturer le client.
        # B voit la course via executing_company_id et facture A (partner_cost).
        # Changer le statut de la course à ACCEPTED pour qu'elle apparaisse dans le dispatch
        # de l'entreprise partenaire (la course est toujours PENDING à ce stade car
        # elle a été réinitialisée lors de la proposition du transfert)
        transfer.booking.status = BookingStatus.ACCEPTED
        # S'assurer que le chauffeur est bien None (déjà révoqué lors de la proposition)
        transfer.booking.driver_id = None

        # Garde-fou : booking.company_id doit rester l'owner (détecte une réintroduction de mutation)
        if transfer.booking.company_id != transfer.owner_company_id:
            logger.warning(
                "[accept_transfer] Incohérence owner: booking.company_id=%s doit rester owner_company_id=%s (booking_id=%s, transfer_id=%s)",
                transfer.booking.company_id,
                transfer.owner_company_id,
                transfer.booking_id,
                transfer.id,
            )

        # #region agent log
        try:
            import json
            from pathlib import Path

            with Path(r"c:\Users\jasiq\atmr\.cursor\debug.log").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": "C",
                            "location": "booking_transfer_service.py:accept_transfer",
                            "message": "Transfer accepted, booking updated",
                            "data": {
                                "transfer_id": transfer.id,
                                "booking_id": transfer.booking_id,
                                "executing_company_id": transfer.executing_company_id,
                                "booking_status": transfer.booking.status.value,
                                "booking_company_id": transfer.booking.company_id,
                                "booking_executing_company_id": transfer.booking.executing_company_id,
                            },
                            "timestamp": int(__import__("time").time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

        db.session.commit()

        # Notifier les entreprises concernées via Socket.IO
        try:
            transfer_dict = transfer.to_dict()
            # Notifier l'entreprise propriétaire
            emit_company_event(
                transfer.owner_company_id,
                "transfer:accepted",
                {
                    "transfer": transfer_dict,
                    "booking_id": transfer.booking_id,
                },
            )
            # Notifier l'entreprise partenaire
            emit_company_event(
                transfer.executing_company_id,
                "transfer:accepted",
                {
                    "transfer": transfer_dict,
                    "booking_id": transfer.booking_id,
                },
            )
        except Exception as e:
            logger.warning("Erreur lors de l'envoi des notifications Socket.IO: %s", e)

        return transfer

    @staticmethod
    def reject_transfer(transfer_id: int, executing_company_id: int) -> BookingTransfer:
        """Refuser ou annuler un transfert.

        - Si appelé par le receveur (executing_company) : REFUSER le transfert
        - Si appelé par l'émetteur (owner_company) : ANNULER le transfert

        Args:
            transfer_id: ID du transfert
            executing_company_id: ID de l'entreprise qui refuse/annule

        Returns:
            BookingTransfer refusé/annulé

        Raises:
            ValueError: Si l'entreprise n'est pas autorisée ou si le transfert ne peut plus être refusé/annulé
        """
        # ✅ FIX CRITIQUE: Convertir explicitement les IDs en int
        try:
            transfer_id = int(transfer_id)
            executing_company_id = int(executing_company_id)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"transfer_id et executing_company_id doivent être des entiers valides: {e}"
            ) from e

        transfer = BookingTransfer.query.get_or_404(transfer_id)

        # ✅ Permettre au receveur (executing_company) OU à l'émetteur (owner_company) de refuser/annuler
        is_receiver = transfer.executing_company_id == executing_company_id
        is_sender = transfer.owner_company_id == executing_company_id

        if not (is_receiver or is_sender):
            raise ValueError(
                "Cette entreprise n'est pas autorisée à refuser/annuler ce transfert"
            )

        if transfer.status != TransferStatus.PENDING:
            raise ValueError("Ce transfert ne peut plus être refusé/annulé")

        transfer.status = TransferStatus.REJECTED
        transfer.rejected_at = datetime.now(UTC)

        db.session.commit()

        # Notifier les entreprises concernées via Socket.IO
        try:
            transfer_dict = transfer.to_dict()
            # Notifier l'entreprise propriétaire
            emit_company_event(
                transfer.owner_company_id,
                "transfer:rejected",
                {
                    "transfer": transfer_dict,
                    "booking_id": transfer.booking_id,
                },
            )
            # Notifier l'entreprise partenaire
            emit_company_event(
                transfer.executing_company_id,
                "transfer:rejected",
                {
                    "transfer": transfer_dict,
                    "booking_id": transfer.booking_id,
                },
            )
        except Exception as e:
            logger.warning("Erreur lors de l'envoi des notifications Socket.IO: %s", e)

        return transfer

    @staticmethod
    def validate_completion(
        transfer_id: int, validator_user_id: int
    ) -> BookingTransfer:
        """Valider la complétion d'une course transférée.

        Args:
            transfer_id: ID du transfert
            validator_user_id: ID de l'utilisateur qui valide

        Returns:
            BookingTransfer validé

        Raises:
            ValueError: Si la course n'est pas complétée
        """
        # ✅ FIX CRITIQUE: Convertir explicitement les IDs en int
        try:
            transfer_id = int(transfer_id)
            validator_user_id = int(validator_user_id)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"transfer_id et validator_user_id doivent être des entiers valides: {e}"
            ) from e

        transfer = BookingTransfer.query.get_or_404(transfer_id)

        if transfer.booking.status != BookingStatus.COMPLETED:
            raise ValueError("La course doit être complétée avant validation")

        if transfer.status != TransferStatus.ACCEPTED:
            raise ValueError("Seuls les transferts acceptés peuvent être validés")

        transfer.is_validated = True
        transfer.validated_at = datetime.now(UTC)
        transfer.validated_by = validator_user_id
        transfer.status = TransferStatus.COMPLETED
        transfer.completed_at = datetime.now(UTC)

        db.session.commit()

        # Déclencher la facturation automatique si configurée
        # NOTE: Pour la facturation mensuelle consolidée, on ne crée plus de facture immédiate
        # Les transferts validés seront regroupés dans une facture mensuelle via PartnerInvoiceService
        # On garde cette logique pour compatibilité, mais elle peut être désactivée
        if transfer.partnership.auto_invoice:
            # Option 1: Facturation immédiate (comportement actuel)
            # Décommenter pour activer la facturation course par course
            # try:
            #     from services.booking.invoices import InvoiceTransferService
            #     invoice_service = InvoiceTransferService()
            #     invoice_service.create_invoices_for_transfer(transfer)
            # except Exception as e:
            #     logger.error(
            #         "Erreur lors de la création des factures pour le transfert %s: %s",
            #         transfer_id,
            #         e,
            #     )
            #     # Ne pas bloquer la validation si la facturation échoue
            #     # L'utilisateur pourra créer les factures manuellement

            # Option 2: Facturation mensuelle consolidée (recommandé)
            # Les transferts validés seront inclus dans la facture mensuelle générée via
            # POST /api/v1/partnerships/<id>/invoices avec year et month
            logger.info(
                "Transfert %s validé - sera inclus dans la facture mensuelle consolidée",
                transfer_id,
            )

        # Notifier les entreprises concernées via Socket.IO
        try:
            transfer_dict = transfer.to_dict()
            # Notifier l'entreprise propriétaire
            emit_company_event(
                transfer.owner_company_id,
                "transfer:validated",
                {
                    "transfer": transfer_dict,
                    "booking_id": transfer.booking_id,
                    "validated_by": validator_user_id,
                },
            )
            # Notifier l'entreprise partenaire
            emit_company_event(
                transfer.executing_company_id,
                "transfer:validated",
                {
                    "transfer": transfer_dict,
                    "booking_id": transfer.booking_id,
                    "validated_by": validator_user_id,
                },
            )
        except Exception as e:
            logger.warning("Erreur lors de l'envoi des notifications Socket.IO: %s", e)

        return transfer

    @staticmethod
    def get_transfer(transfer_id: int) -> BookingTransfer | None:
        """Récupérer un transfert par son ID.

        Args:
            transfer_id: ID du transfert

        Returns:
            BookingTransfer ou None
        """
        # ✅ FIX CRITIQUE: Convertir explicitement transfer_id en int
        try:
            transfer_id = int(transfer_id)
        except (ValueError, TypeError):
            logger.warning(
                "get_transfer: Invalid transfer_id type: %s",
                type(transfer_id).__name__,
            )
            return None

        return BookingTransfer.query.get(transfer_id)

    @staticmethod
    def get_transfers_for_booking(booking_id: int) -> list[BookingTransfer]:
        """Récupérer tous les transferts d'une course.

        Args:
            booking_id: ID de la course

        Returns:
            Liste des transferts
        """
        # ✅ FIX CRITIQUE: Convertir explicitement booking_id en int
        try:
            booking_id = int(booking_id)
        except (ValueError, TypeError):
            logger.warning(
                "get_transfers_for_booking: Invalid booking_id type: %s",
                type(booking_id).__name__,
            )
            return []

        return BookingTransfer.query.filter_by(booking_id=booking_id).all()

    @staticmethod
    def get_transfers_for_partnership(
        partnership_id: int, status: TransferStatus | None = None
    ) -> list[BookingTransfer]:
        """Récupérer tous les transferts d'un partenariat.

        Args:
            partnership_id: ID du partenariat
            status: Filtrer par statut (optionnel)

        Returns:
            Liste des transferts
        """
        # ✅ FIX CRITIQUE: Convertir explicitement partnership_id en int
        try:
            partnership_id = int(partnership_id)
        except (ValueError, TypeError):
            logger.warning(
                "get_transfers_for_partnership: Invalid partnership_id type: %s",
                type(partnership_id).__name__,
            )
            return []

        query = BookingTransfer.query.filter_by(partnership_id=partnership_id)
        if status:
            query = query.filter_by(status=status)
        return query.all()
