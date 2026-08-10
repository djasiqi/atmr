"""Tests pour la persistance de la signature email dans CompanyBillingSettings."""

import pytest

from models import CompanyBillingSettings


@pytest.mark.integration
class TestCompanySettingsEmailSignature:
    """Tests pour vérifier la persistance de email_signature_text."""

    def test_email_signature_text_persistence(self, db, test_company):
        """Test que email_signature_text est persisté et récupéré correctement."""
        if not test_company:
            pytest.skip("Required fixture missing")

        # Arrange: Créer ou récupérer les billing settings
        billing = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        if not billing:
            billing = CompanyBillingSettings(company_id=test_company.id)
            db.session.add(billing)
            db.session.commit()

        # Act: Définir une signature
        signature_text = (
            "Khalid ALAOUI\n"
            "Associé gérant – Emmenez-moi Sàrl\n"
            "022 512 02 03 | 079 291 50 37\n"
            "info@casa-famiglia.ch\n"
            "www.transport-emmenez-moi.ch\n"
            "Route de Chevrens 145, 1247 Anières"
        )
        billing.email_signature_text = signature_text
        db.session.commit()

        # Simuler un redémarrage backend
        db.session.expire_all()

        # Assert: Vérifier que la signature est persistée
        billing_reloaded = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        assert billing_reloaded is not None
        assert billing_reloaded.email_signature_text == signature_text

        # Assert: Vérifier via to_dict()
        result_dict = billing_reloaded.to_dict()
        assert "email_signature_text" in result_dict
        assert result_dict["email_signature_text"] == signature_text

    def test_email_signature_text_empty_persistence(self, db, test_company):
        """Test que email_signature_text peut être vide/null."""
        if not test_company:
            pytest.skip("Required fixture missing")

        billing = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        if not billing:
            billing = CompanyBillingSettings(company_id=test_company.id)
            db.session.add(billing)
            db.session.commit()

        # Act: Définir puis vider la signature
        billing.email_signature_text = "Test signature"
        db.session.commit()
        billing.email_signature_text = None
        db.session.commit()

        # Assert: Vérifier que c'est bien None
        db.session.expire_all()
        billing_reloaded = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        assert billing_reloaded.email_signature_text is None

        result_dict = billing_reloaded.to_dict()
        assert result_dict.get("email_signature_text") is None

    def test_email_signature_mode_and_html_template_persistence(self, db, test_company):
        """Test que email_signature_mode et email_signature_html_template sont persistés."""
        if not test_company:
            pytest.skip("Required fixture missing")

        billing = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        if not billing:
            billing = CompanyBillingSettings(company_id=test_company.id)
            db.session.add(billing)
            db.session.commit()

        # Act: Définir mode HTML avec template
        html_template = (
            "<table><tr><td><strong>{{ name }}</strong><br>{{ phone }}</td>"
            '<td style="border-left: 2px solid #1b4b7a;">Colonne droite</td></tr></table>'
        )
        billing.email_signature_mode = "html"
        billing.email_signature_html_template = html_template
        test_company.logo_url = "https://example.com/logo.png"
        db.session.commit()

        # Assert: Vérifier persistance
        db.session.expire_all()
        billing_reloaded = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        assert billing_reloaded.email_signature_mode == "html"
        assert billing_reloaded.email_signature_html_template == html_template
        assert test_company.logo_url == "https://example.com/logo.png"

        result_dict = billing_reloaded.to_dict()
        assert result_dict["email_signature_mode"] == "html"
        assert result_dict["email_signature_html_template"] == html_template

    def test_email_signature_form_mode_persistence(self, db, test_company):
        """Test que les champs du mode 'form' sont persistés."""
        if not test_company:
            pytest.skip("Required fixture missing")

        billing = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        if not billing:
            billing = CompanyBillingSettings(company_id=test_company.id)
            db.session.add(billing)
            db.session.commit()

        # Act: Définir mode form avec champs normalisés
        billing.email_signature_mode = "form"
        billing.signature_name = "Khalid ALAOUI"
        billing.signature_title = "Associé gérant"
        billing.signature_company = "Emmenez-moi Sàrl"
        billing.signature_phone_main = "022 512 02 03"
        billing.signature_phone_mobile = "079 291 50 37"
        billing.signature_email = "info@casa-famiglia.ch"
        billing.signature_website = "www.transport-emmenez-moi.ch"
        billing.signature_address_line = "Route de Chevrens 145"
        billing.signature_zip = "1247"
        billing.signature_city = "Anières"
        test_company.logo_url = "https://example.com/logo.png"
        db.session.commit()

        # Assert: Vérifier persistance
        db.session.expire_all()
        billing_reloaded = CompanyBillingSettings.query.filter_by(
            company_id=test_company.id
        ).first()
        assert billing_reloaded.email_signature_mode == "form"
        assert billing_reloaded.signature_name == "Khalid ALAOUI"
        assert billing_reloaded.signature_title == "Associé gérant"
        assert billing_reloaded.signature_company == "Emmenez-moi Sàrl"
        assert billing_reloaded.signature_phone_main == "022 512 02 03"
        assert billing_reloaded.signature_phone_mobile == "079 291 50 37"
        assert billing_reloaded.signature_email == "info@casa-famiglia.ch"
        assert billing_reloaded.signature_website == "www.transport-emmenez-moi.ch"
        assert billing_reloaded.signature_address_line == "Route de Chevrens 145"
        assert billing_reloaded.signature_zip == "1247"
        assert billing_reloaded.signature_city == "Anières"
        assert test_company.logo_url == "https://example.com/logo.png"

        result_dict = billing_reloaded.to_dict()
        assert result_dict["email_signature_mode"] == "form"
        assert result_dict["signature_name"] == "Khalid ALAOUI"
        assert result_dict["signature_title"] == "Associé gérant"
        assert result_dict["signature_company"] == "Emmenez-moi Sàrl"
        assert result_dict["signature_phone_main"] == "022 512 02 03"
        assert result_dict["signature_phone_mobile"] == "079 291 50 37"
        assert result_dict["signature_email"] == "info@casa-famiglia.ch"
        assert result_dict["signature_website"] == "www.transport-emmenez-moi.ch"
        assert result_dict["signature_address_line"] == "Route de Chevrens 145"
        assert result_dict["signature_zip"] == "1247"
        assert result_dict["signature_city"] == "Anières"
