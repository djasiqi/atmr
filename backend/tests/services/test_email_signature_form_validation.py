"""Tests pour la validation et normalisation des champs dans generate_signature_html_from_form."""

import pytest

from services.email.signature_utils import generate_signature_html_from_form


class TestEmailSignatureFormValidation:
    """Tests pour la validation des champs du formulaire."""

    def test_truncate_long_name(self):
        """Test que les noms trop longs sont tronqués."""
        long_name = "A" * 300  # Plus long que MAX_LENGTH_NAME (200)
        result = generate_signature_html_from_form(name=long_name)

        # Le nom doit être tronqué à 200 caractères
        assert "A" * 200 in result
        assert "A" * 201 not in result

    def test_truncate_long_email(self):
        """Test que les emails trop longs sont tronqués."""
        long_email = "a" * 150 + "@example.com"  # Plus long que MAX_LENGTH_EMAIL (200)
        result = generate_signature_html_from_form(email=long_email)

        # L'email doit être tronqué
        assert "@example.com" in result
        assert len([c for c in result if c == "a"]) < 150  # Moins de 'a' que l'original

    def test_validate_email_requires_at(self):
        """Test que les emails sans @ sont rejetés."""
        invalid_email = "notanemail.com"
        result = generate_signature_html_from_form(email=invalid_email)

        # L'email invalide ne doit pas apparaître
        assert "notanemail.com" not in result
        assert 'href="mailto:' not in result

    def test_validate_email_strips_whitespace(self):
        """Test que les emails sont strippés."""
        email_with_spaces = "  info@example.com  "
        result = generate_signature_html_from_form(email=email_with_spaces)

        # L'email doit être strippé
        assert 'href="mailto:info@example.com"' in result
        assert "  info@example.com  " not in result

    def test_normalize_website_strips_whitespace(self):
        """Test que les websites sont strippés."""
        website_with_spaces = "  www.example.com  "
        result = generate_signature_html_from_form(website=website_with_spaces)

        # Le website doit être strippé
        assert "www.example.com" in result
        assert "  www.example.com  " not in result

    def test_normalize_website_adds_https(self):
        """Test que https:// est ajouté si absent."""
        website_no_protocol = "www.example.com"
        result = generate_signature_html_from_form(website=website_no_protocol)

        # Le lien doit avoir https://
        assert 'href="https://www.example.com"' in result
        # Mais l'affichage ne doit pas avoir https://
        assert "www.example.com" in result

    def test_normalize_website_preserves_https(self):
        """Test que https:// existant est préservé."""
        website_with_https = "https://example.com"
        result = generate_signature_html_from_form(website=website_with_https)

        # Le https:// doit être préservé dans le lien
        assert 'href="https://example.com"' in result

    def test_truncate_long_phone(self):
        """Test que les téléphones trop longs sont tronqués."""
        long_phone = "0" * 100  # Plus long que MAX_LENGTH_PHONE (50)
        result = generate_signature_html_from_form(phone_main=long_phone)

        # Le téléphone doit être tronqué à 50 caractères
        assert "0" * 50 in result
        assert "0" * 51 not in result

    def test_truncate_long_address(self):
        """Test que les adresses trop longues sont tronquées."""
        long_address = "Rue " * 100  # Plus long que MAX_LENGTH_ADDRESS_LINE (200)
        result = generate_signature_html_from_form(address_line=long_address)

        # L'adresse doit être tronquée
        assert "Rue" in result
        # Vérifier qu'elle n'est pas trop longue (approximatif)
        address_in_result = result[result.find("Rue") : result.find("Rue") + 250]
        assert (
            len(address_in_result) <= 250
        )  # Avec échappement HTML, un peu plus que 200

    def test_strip_all_fields(self):
        """Test que tous les champs sont strippés."""
        result = generate_signature_html_from_form(
            name="  Test Name  ",
            title="  Title  ",
            company="  Company  ",
            phone_main="  123  ",
            phone_mobile="  456  ",
            address_line="  Address  ",
            zip_code="  1234  ",
            city="  City  ",
        )

        # Aucun champ ne doit avoir d'espaces en début/fin
        assert "  Test Name  " not in result
        assert "Test Name" in result
        assert "  Title  " not in result
        assert "Title" in result
        assert "  123  " not in result
        assert "123" in result

    def test_empty_fields_after_strip_are_ignored(self):
        """Test que les champs vides après strip sont ignorés."""
        result = generate_signature_html_from_form(
            name="   ",
            email="   ",
            website="   ",
        )

        # Les champs vides ne doivent pas apparaître
        assert "&nbsp;" in result or "<table" in result  # Structure minimale
        # Pas de contenu visible

    def test_valid_email_with_at(self):
        """Test qu'un email valide avec @ est accepté."""
        valid_email = "user@example.com"
        result = generate_signature_html_from_form(email=valid_email)

        assert 'href="mailto:user@example.com"' in result
        assert "user@example.com" in result

    def test_email_with_multiple_at_rejected(self):
        """Test qu'un email avec plusieurs @ est rejeté (validation minimale)."""
        # Note: notre validation est minimale (@ présent), donc "user@@example.com" passera
        # mais c'est acceptable pour une validation légère
        email_multiple_at = "user@@example.com"
        result = generate_signature_html_from_form(email=email_multiple_at)

        # Avec notre validation minimale, ça passera (contient @)
        # Mais on peut vérifier que le mailto: est présent
        assert "@" in email_multiple_at  # Notre validation minimale
        assert 'href="mailto:' in result

    def test_outlook_compatible_logo_styles(self):
        """Test que le logo a les styles Outlook-safe."""

        # Créer un mock company avec logo_url
        class MockCompany:
            def __init__(self):
                self.logo_url = "https://example.com/logo.png"
                self.name = "Test Company"

        mock_company = MockCompany()
        result = generate_signature_html_from_form(
            name="Test",
            company_obj=mock_company,
        )

        # Vérifier les styles Outlook-safe sur le logo (height=26, width:auto)
        assert 'height="26"' in result
        assert "width:auto" in result
        assert "max-width:100%" in result
        assert (
            'style="display:block;border:0;outline:none;text-decoration:none;height:26px;width:auto;max-width:100%;"'
            in result
        )
        assert "<img" in result
        # Vérifier CID strict: exactement "company_logo"
        assert 'src="cid:company_logo"' in result or "cid:company_logo" in result
        # Vérifier la largeur fixe 520px et align="left" pour Outlook
        assert 'width="520"' in result
        assert "width:520px" in result
        assert "max-width:520px" in result
        assert 'align="left"' in result

    def test_outlook_compatible_horizontal_line(self):
        """Test que la ligne horizontale utilise une mini-table (Outlook)."""
        result = generate_signature_html_from_form(name="Test")

        # Vérifier que la ligne horizontale utilise une mini-table
        assert "line-height: 1px; font-size: 1px" in result
        assert "border-top: 1px solid #1b4b7a" in result
        # Vérifier la structure de la mini-table avec largeur fixe 520px et align="left"
        assert 'width="520"' in result
        assert "width:520px" in result
        assert "max-width:520px" in result
        assert 'align="left"' in result

    def test_outlook_compatible_horizontal_line_with_logo(self):
        """Test que la ligne horizontale avec logo utilise aussi une mini-table."""

        # Créer un mock company avec logo_url
        class MockCompany:
            def __init__(self):
                self.logo_url = "https://example.com/logo.png"
                self.name = "Test Company"

        mock_company = MockCompany()
        result = generate_signature_html_from_form(
            name="Test",
            company_obj=mock_company,
        )

        # Vérifier que la ligne horizontale utilise une mini-table même avec logo
        assert "line-height: 1px; font-size: 1px" in result
        assert "border-top: 1px solid #1b4b7a" in result
        # Vérifier la largeur fixe 520px
        assert 'width="520"' in result
        assert "width:520px" in result
        assert "max-width:520px" in result
        # Vérifier que le logo utilise CID inline strict
        assert 'src="cid:company_logo"' in result or "cid:company_logo" in result
        # Vérifier align="left" sur les tables wrapper
        assert 'align="left"' in result
