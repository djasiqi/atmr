# tests/services/test_partnership_service.py
"""Tests unitaires pour PartnershipService."""

from decimal import Decimal

import pytest

from ext import db
from models.enums import TransferModel
from models.partnership import Partnership
from services.partnerships.core import PartnershipService
from tests.factories import CompanyFactory


class TestPartnershipService:
    """Tests pour PartnershipService."""

    def test_create_partnership(self, app):
        """Test de création d'un partenariat."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
                default_transfer_model=TransferModel.SUBCONTRACT,
                default_margin_percent=Decimal("20.0"),
                auto_accept=False,
                auto_invoice=True,
                payment_terms_days=30,
            )

            assert partnership is not None
            assert partnership.owner_company_id == company_a.id
            assert partnership.partner_company_id == company_b.id
            assert partnership.default_transfer_model == TransferModel.SUBCONTRACT
            assert partnership.default_margin_percent == Decimal("20.0")
            assert partnership.auto_accept_rules is False
            assert partnership.auto_invoice is True
            assert partnership.payment_terms_days == 30
            assert partnership.is_active is True

    def test_create_partnership_duplicate(self, app):
        """Test qu'on ne peut pas créer un partenariat en double."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            db.session.commit()

            # Créer le premier partenariat
            PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            # Essayer de créer un doublon
            with pytest.raises(ValueError, match="déjà existe"):
                PartnershipService.create_partnership(
                    owner_company_id=company_a.id,
                    partner_company_id=company_b.id,
                )

    def test_create_partnership_same_company(self, app):
        """Test qu'on ne peut pas créer un partenariat avec soi-même."""
        with app.app_context():
            company = CompanyFactory()
            db.session.commit()

            with pytest.raises(ValueError, match="même entreprise"):
                PartnershipService.create_partnership(
                    owner_company_id=company.id,
                    partner_company_id=company.id,
                )

    def test_get_partnerships_for_company(self, app):
        """Test de récupération des partenariats d'une entreprise."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            company_c = CompanyFactory()
            db.session.commit()

            # Créer des partenariats
            partnership_ab = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            partnership_ac = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_c.id,
            )
            db.session.commit()

            # Récupérer les partenariats de company_a
            partnerships = PartnershipService.get_partnerships_for_company(company_a.id)

            assert len(partnerships) == 2
            partnership_ids = {p.id for p in partnerships}
            assert partnership_ab.id in partnership_ids
            assert partnership_ac.id in partnership_ids

    def test_get_partnership_by_id(self, app):
        """Test de récupération d'un partenariat par ID."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            retrieved = PartnershipService.get_partnership_by_id(partnership.id)

            assert retrieved is not None
            assert retrieved.id == partnership.id
            assert retrieved.owner_company_id == company_a.id
            assert retrieved.partner_company_id == company_b.id

    def test_update_partnership(self, app):
        """Test de mise à jour d'un partenariat."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
                default_margin_percent=Decimal("10.0"),
            )
            db.session.commit()

            # Mettre à jour
            updated = PartnershipService.update_partnership(
                partnership_id=partnership.id,
                default_margin_percent=Decimal("25.0"),
                auto_accept=True,
            )

            assert updated.default_margin_percent == Decimal("25.0")
            assert updated.auto_accept_rules is True

    def test_deactivate_partnership(self, app):
        """Test de désactivation d'un partenariat."""
        with app.app_context():
            company_a = CompanyFactory()
            company_b = CompanyFactory()
            db.session.commit()

            partnership = PartnershipService.create_partnership(
                owner_company_id=company_a.id,
                partner_company_id=company_b.id,
            )
            db.session.commit()

            assert partnership.is_active is True

            PartnershipService.deactivate_partnership(partnership.id)
            db.session.refresh(partnership)

            assert partnership.is_active is False

