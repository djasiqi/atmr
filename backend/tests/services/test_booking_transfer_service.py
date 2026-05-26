# tests/services/test_booking_transfer_service.py
"""Tests unitaires pour BookingTransferService."""

from datetime import UTC, datetime

import pytest

from ext import db
from models.booking import Booking
from models.enums import BookingStatus, TransferModel, TransferStatus
from services.booking.transfers import BookingTransferService
from services.partnerships.core import PartnershipService
from tests.factories import (
    BookingFactory,
    ClientFactory,
    CompanyFactory,
)


class TestBookingTransferService:
    """Tests pour BookingTransferService."""

    def test_propose_transfer(self, app):
        """Test de proposition d'un transfert."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            client = ClientFactory(company=company_a)
            db.session.commit()

            # Créer un partenariat
            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
                default_transfer_model=TransferModel.SUBCONTRACT,
                default_margin_percent=20.0,
            )
            db.session.commit()

            # Créer une course
            booking = BookingFactory(
                company=company_a,
                client=client,
                status=BookingStatus.ACCEPTED,
                amount=100.0,
            )
            db.session.commit()

            # Proposer le transfert
            transfer = BookingTransferService.propose_transfer(
                booking_id=booking.id,
                partnership_id=partnership.id,
            )

            assert transfer is not None
            assert transfer.booking_id == booking.id
            assert transfer.partnership_id == partnership.id
            assert transfer.owner_company_id == company_a.id
            assert transfer.executing_company_id == company_b.id
            assert transfer.status == TransferStatus.PENDING
            assert transfer.client_price == 100.0
            assert transfer.partner_cost is not None
            assert transfer.partner_cost < transfer.client_price  # Marge appliquée

    def test_propose_transfer_wrong_company(self, app):
        """Test qu'on ne peut pas proposer une course d'une autre entreprise."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            company_c = CompanyFactory()
            client = ClientFactory(company=company_c)
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            booking = BookingFactory(
                company=company_c,  # Course d'une autre entreprise
                client=client,
                status=BookingStatus.ACCEPTED,
            )
            db.session.commit()

            with pytest.raises(ValueError, match="n'appartient pas"):
                BookingTransferService.propose_transfer(
                    booking_id=booking.id,
                    partnership_id=partnership.id,
                )

    def test_propose_transfer_already_transferred(self, app):
        """Test qu'on ne peut pas proposer une course déjà transférée."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            client = ClientFactory(company=company_a)
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            booking = BookingFactory(
                company=company_a,
                client=client,
                status=BookingStatus.ACCEPTED,
            )
            db.session.commit()

            # Proposer le premier transfert
            BookingTransferService.propose_transfer(
                booking_id=booking.id,
                partnership_id=partnership.id,
            )
            db.session.commit()

            # Essayer de proposer un deuxième transfert
            with pytest.raises(ValueError, match="déjà en cours"):
                BookingTransferService.propose_transfer(
                    booking_id=booking.id,
                    partnership_id=partnership.id,
                )

    def test_accept_transfer(self, app):
        """Test d'acceptation d'un transfert."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            client = ClientFactory(company=company_a)
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            booking = BookingFactory(
                company=company_a,
                client=client,
                status=BookingStatus.ACCEPTED,
            )
            db.session.commit()

            transfer = BookingTransferService.propose_transfer(
                booking_id=booking.id,
                partnership_id=partnership.id,
            )
            db.session.commit()

            # Accepter le transfert
            accepted = BookingTransferService.accept_transfer(
                transfer_id=transfer.id,
                executing_company_id=company_b.id,
            )

            assert accepted.status == TransferStatus.ACCEPTED
            assert accepted.accepted_at is not None
            assert booking.executing_company_id == company_b.id

    def test_accept_transfer_wrong_company(self, app):
        """Test qu'on ne peut accepter que si on est l'entreprise exécutante."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            company_c = CompanyFactory()
            client = ClientFactory(company=company_a)
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            booking = BookingFactory(
                company=company_a,
                client=client,
                status=BookingStatus.ACCEPTED,
            )
            db.session.commit()

            transfer = BookingTransferService.propose_transfer(
                booking_id=booking.id,
                partnership_id=partnership.id,
            )
            db.session.commit()

            # Essayer d'accepter avec une autre entreprise
            with pytest.raises(ValueError, match="n'est pas autorisée"):
                BookingTransferService.accept_transfer(
                    transfer_id=transfer.id,
                    executing_company_id=company_c.id,
                )

    def test_reject_transfer(self, app):
        """Test de refus d'un transfert."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            client = ClientFactory(company=company_a)
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            booking = BookingFactory(
                company=company_a,
                client=client,
                status=BookingStatus.ACCEPTED,
            )
            db.session.commit()

            transfer = BookingTransferService.propose_transfer(
                booking_id=booking.id,
                partnership_id=partnership.id,
            )
            db.session.commit()

            # Refuser le transfert
            rejected = BookingTransferService.reject_transfer(
                transfer_id=transfer.id,
                executing_company_id=company_b.id,
            )

            assert rejected.status == TransferStatus.REJECTED
            assert rejected.rejected_at is not None

    def test_validate_completion(self, app):
        """Test de validation de complétion."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            client = ClientFactory(company=company_a)
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
                auto_invoice=False,  # Désactiver facturation auto pour le test
            )
            db.session.commit()

            booking = BookingFactory(
                company=company_a,
                client=client,
                status=BookingStatus.COMPLETED,  # Course complétée
            )
            db.session.commit()

            transfer = BookingTransferService.propose_transfer(
                booking_id=booking.id,
                partnership_id=partnership.id,
            )
            db.session.commit()

            # Accepter d'abord
            BookingTransferService.accept_transfer(
                transfer_id=transfer.id,
                executing_company_id=company_b.id,
            )
            db.session.commit()

            # Valider la complétion
            validated = BookingTransferService.validate_completion(
                transfer_id=transfer.id,
                validator_user_id=1,  # ID utilisateur fictif
            )

            assert validated.is_validated is True
            assert validated.validated_at is not None
            assert validated.validated_by == 1
            assert validated.status == TransferStatus.COMPLETED

    def test_validate_completion_not_completed(self, app):
        """Test qu'on ne peut valider que si la course est complétée."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            client = ClientFactory(company=company_a)
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            booking = BookingFactory(
                company=company_a,
                client=client,
                status=BookingStatus.ACCEPTED,  # Pas complétée
            )
            db.session.commit()

            transfer = BookingTransferService.propose_transfer(
                booking_id=booking.id,
                partnership_id=partnership.id,
            )
            db.session.commit()

            BookingTransferService.accept_transfer(
                transfer_id=transfer.id,
                executing_company_id=company_b.id,
            )
            db.session.commit()

            # Essayer de valider une course non complétée
            with pytest.raises(ValueError, match="doit être complétée"):
                BookingTransferService.validate_completion(
                    transfer_id=transfer.id,
                    validator_user_id=1,
                )
