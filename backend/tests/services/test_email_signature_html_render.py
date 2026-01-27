"""Tests pour le rendu de templates HTML de signature email."""

import pytest

from models import Company
from services.email.signature_utils import render_signature_html_template


@pytest.fixture
def mock_company():
    """Mock d'une entreprise avec toutes les données."""
    company = type("Company", (), {})()
    company.name = "Emmenez-moi Sàrl"
    company.contact_phone = "022 512 02 03"
    company.contact_email = "info@casa-famiglia.ch"
    company.billing_email = "facturation@casa-famiglia.ch"
    company.domicile_address_line1 = "Route de Chevrens 145"
    company.domicile_zip = "1247"
    company.domicile_city = "Anières"
    company.logo_url = "https://example.com/logo.png"
    return company


@pytest.fixture
def mock_billing_settings():
    """Mock de billing settings avec logo custom."""
    # Plus de signature_logo_url, on utilise uniquement company.logo_url
    return type("BillingSettings", (), {})()


class TestEmailSignatureHtmlRender:
    """Tests pour render_signature_html_template."""

    def test_render_simple_template(self, mock_company):
        """Test rendu basique avec variables."""
        template = "<strong>{{ name }}</strong><br>{{ phone }}"
        result = render_signature_html_template(template, mock_company)

        assert "Emmenez-moi Sàrl" in result
        assert "022 512 02 03" in result
        assert "<strong>" in result

    def test_render_all_variables(self, mock_company):
        """Test que toutes les variables whitelistées fonctionnent."""
        template = (
            "{{ name }}<br>"
            "{{ phone }}<br>"
            "{{ email }}<br>"
            "{{ address }}<br>"
            "{% if logo_url %}{{ logo_url }}{% endif %}"
        )
        result = render_signature_html_template(template, mock_company)

        assert "Emmenez-moi Sàrl" in result
        assert "022 512 02 03" in result
        assert "info@casa-famiglia.ch" in result  # contact_email prioritaire
        assert "Route de Chevrens 145" in result
        assert "1247 Anières" in result
        assert "https://example.com/logo.png" in result

    def test_render_with_company_logo_url(self, mock_company, mock_billing_settings):
        """Test que company.logo_url est utilisé (plus de signature_logo_url)."""
        template = "{% if logo_url %}{{ logo_url }}{% endif %}"
        result = render_signature_html_template(
            template, mock_company, mock_billing_settings
        )

        # Le logo_url de la company doit être utilisé (pas de signature_logo_url)
        assert "test_logo.png" in result or "/uploads/company_logos/test_logo.png" in result

    def test_render_address_formatting(self, mock_company):
        """Test que l'adresse est formatée avec <br>."""
        template = "{{ address }}"
        result = render_signature_html_template(template, mock_company)

        assert "Route de Chevrens 145" in result
        assert "1247 Anières" in result
        assert "<br>" in result  # Les lignes sont séparées par <br>

    def test_render_email_priority(self, mock_company):
        """Test que billing_email a priorité sur contact_email."""
        template = "{{ email }}"
        result = render_signature_html_template(template, mock_company)

        # billing_email prioritaire
        assert "facturation@casa-famiglia.ch" in result
        assert "info@casa-famiglia.ch" not in result

    def test_render_empty_template(self, mock_company):
        """Test que template vide retourne chaîne vide."""
        result = render_signature_html_template("", mock_company)
        assert result == ""

        result2 = render_signature_html_template("   ", mock_company)
        assert result2 == ""

    def test_render_blocks_script_tags(self, mock_company):
        """Test que les balises <script> sont supprimées (sécurité XSS)."""
        template = (
            "{{ name }}<script>alert('xss')</script>"
            "<img src='x' onerror='alert(1)'>"
        )
        result = render_signature_html_template(template, mock_company)

        assert "Emmenez-moi Sàrl" in result
        assert "<script>" not in result
        assert "alert('xss')" not in result
        # Les attributs onclick/onerror sont supprimés
        assert "onerror" not in result

    def test_render_blocks_iframe_tags(self, mock_company):
        """Test que les balises <iframe> sont supprimées."""
        template = "{{ name }}<iframe src='evil.com'></iframe>"
        result = render_signature_html_template(template, mock_company)

        assert "Emmenez-moi Sàrl" in result
        assert "<iframe" not in result

    def test_render_blocks_event_handlers(self, mock_company):
        """Test que les attributs onclick/onload/etc sont supprimés."""
        template = (
            "{{ name }}"
            '<div onclick="alert(1)">Click</div>'
            '<img onload="evil()" src="x">'
        )
        result = render_signature_html_template(template, mock_company)

        assert "Emmenez-moi Sàrl" in result
        assert "onclick=" not in result
        assert "onload=" not in result

    def test_render_table_based_template(self, mock_company):
        """Test rendu d'un template table-based (comme recommandé)."""
        template = """
        <table cellpadding="0" cellspacing="0" border="0" style="font-family: Arial, sans-serif; font-size: 11px;">
          <tr>
            <td style="vertical-align: top; padding-right: 12px;">
              <strong>{{ name }}</strong><br>
              {{ phone }}<br>
              {{ email }}
            </td>
            <td width="1" style="border-left: 2px solid #1b4b7a; padding-left: 12px;">
              Colonne droite
            </td>
          </tr>
        </table>
        <div style="border-top: 1px solid #1b4b7a; margin-top: 12px;">
          {% if logo_url %}
            <img src="{{ logo_url }}" height="26" alt="Logo" />
          {% endif %}
        </div>
        """
        result = render_signature_html_template(template, mock_company)

        assert "Emmenez-moi Sàrl" in result
        assert "022 512 02 03" in result
        assert "info@casa-famiglia.ch" in result
        assert "border-left: 2px solid #1b4b7a" in result
        assert "border-top: 1px solid #1b4b7a" in result
        assert "https://example.com/logo.png" in result
        assert "<table" in result
        assert "<img" in result

    def test_render_invalid_template_returns_empty(self, mock_company):
        """Test qu'un template invalide (syntaxe Jinja2) retourne chaîne vide."""
        template = "{{ name }}{{ invalid_syntax {% }}"
        result = render_signature_html_template(template, mock_company)

        # En cas d'erreur, on retourne vide (log warning)
        assert result == ""

    def test_render_missing_company_returns_empty(self):
        """Test qu'un company None retourne chaîne vide."""
        template = "{{ name }}"
        result = render_signature_html_template(template, None)

        assert result == ""


class TestGenerateSimpleSignatureHtml:
    """Tests pour generate_simple_signature_html."""

    def test_generate_simple_signature_with_all_fields(self):
        """Test génération avec tous les champs remplis."""
        from services.email.signature_utils import generate_simple_signature_html

        result = generate_simple_signature_html(
            contact_name="Khalid ALAOUI",
            phone="022 512 02 03",
            email="info@casa-famiglia.ch",
            website="www.transport-emmenez-moi.ch",
            address="Route de Chevrens 145\n1247 Anières",
            logo_url="https://example.com/logo.png",
        )

        assert "Khalid ALAOUI" in result
        assert "022 512 02 03" in result
        assert "info@casa-famiglia.ch" in result
        assert "www.transport-emmenez-moi.ch" in result
        assert "Route de Chevrens 145" in result
        assert "1247 Anières" in result
        assert "https://example.com/logo.png" in result
        assert "<table" in result
        assert "border-left: 2px solid #1b4b7a" in result
        assert "border-top: 1px solid #1b4b7a" in result
        assert "<img" in result

    def test_generate_simple_signature_minimal(self):
        """Test génération avec seulement nom et téléphone."""
        from services.email.signature_utils import generate_simple_signature_html

        result = generate_simple_signature_html(
            contact_name="Test Name",
            phone="123 456 789",
        )

        assert "Test Name" in result
        assert "123 456 789" in result
        assert "<table" in result
        # Pas de logo si logo_url non fourni
        assert "<img" not in result

    def test_generate_simple_signature_escapes_html(self):
        """Test que le HTML est échappé pour éviter XSS."""
        from services.email.signature_utils import generate_simple_signature_html

        result = generate_simple_signature_html(
            contact_name="<script>alert('xss')</script>",
            email="test@example.com",
        )

        assert "<script>" not in result
        assert "alert('xss')" not in result
        assert "&lt;script&gt;" in result or "&lt;" in result  # HTML échappé

    def test_generate_simple_signature_address_newlines(self):
        """Test que les \n dans l'adresse sont convertis en <br>."""
        from services.email.signature_utils import generate_simple_signature_html

        result = generate_simple_signature_html(
            address="Ligne 1\nLigne 2\nLigne 3",
        )

        assert "Ligne 1" in result
        assert "Ligne 2" in result
        assert "Ligne 3" in result
        assert "<br>" in result

    def test_generate_simple_signature_empty_returns_empty(self):
        """Test que si aucun champ n'est rempli, retourne HTML minimal."""
        from services.email.signature_utils import generate_simple_signature_html

        result = generate_simple_signature_html()

        # Devrait quand même retourner la structure HTML (table vide)
        assert "<table" in result
