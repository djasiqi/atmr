"""Tests pour l'injection de signature email dans les emails."""

import pytest

from services.email.signature_utils import inject_signature_into_html


def create_mock_billing_settings(mode="text", signature_text=None, **kwargs):
    """Helper pour créer un mock de billing_settings."""
    settings = type("BillingSettings", (), {})()
    settings.email_signature_mode = mode
    settings.email_signature_text = signature_text
    # Champs mode "form"
    for key, value in kwargs.items():
        setattr(settings, key, value)
    return settings


class TestEmailSignatureInjection:
    """Tests pour inject_signature_into_html."""

    def test_inject_signature_with_body_tag(self):
        """Test injection normale avec tag </body>."""
        html = "<html><body><p>Bonjour</p></body></html>"
        billing_settings = create_mock_billing_settings(
            mode="text", signature_text="Khalid ALAOUI\nAssocié gérant"
        )
        html_result, _ = inject_signature_into_html(html, billing_settings=billing_settings)

        assert "</body>" in html_result
        assert "Khalid ALAOUI" in html_result
        assert "Associé gérant" in html_result
        assert "—" in html_result  # Séparateur
        assert html_result.count("</body>") == 1
        # Vérifier que la signature est avant </body>
        assert html_result.index("—") < html_result.index("</body>")

    def test_inject_signature_with_html_tag_only(self):
        """Test injection avec fallback sur </html> si pas de </body>."""
        html = "<html><p>Bonjour</p></html>"
        billing_settings = create_mock_billing_settings(
            mode="text", signature_text="Test signature"
        )
        html_result, _ = inject_signature_into_html(html, billing_settings=billing_settings)

        assert "Test signature" in html_result
        assert "—" in html_result
        assert html_result.index("—") < html_result.index("</html>")

    def test_inject_signature_no_tags(self):
        """Test injection avec fallback append si aucun tag."""
        html = "<p>Bonjour</p>"
        billing_settings = create_mock_billing_settings(
            mode="text", signature_text="Test signature"
        )
        html_result, _ = inject_signature_into_html(html, billing_settings=billing_settings)

        assert html in html_result
        assert "Test signature" in html_result
        assert html_result.endswith("Test signature")

    def test_inject_signature_empty(self):
        """Test que signature vide retourne html inchangé."""
        html = "<html><body><p>Bonjour</p></body></html>"
        billing_settings = create_mock_billing_settings(mode="text", signature_text=None)
        html_result, _ = inject_signature_into_html(html, billing_settings=billing_settings)
        assert html_result == html

        billing_settings2 = create_mock_billing_settings(mode="text", signature_text="")
        html_result2, _ = inject_signature_into_html(html, billing_settings=billing_settings2)
        assert html_result2 == html

        billing_settings3 = create_mock_billing_settings(mode="text", signature_text="   ")
        html_result3, _ = inject_signature_into_html(html, billing_settings=billing_settings3)
        assert html_result3 == html

    def test_inject_signature_escapes_html(self):
        """Test que le HTML dans la signature est échappé (sécurité)."""
        html = "<html><body><p>Test</p></body></html>"
        billing_settings = create_mock_billing_settings(
            mode="text", signature_text="<script>alert('xss')</script>"
        )
        html_result, _ = inject_signature_into_html(html, billing_settings=billing_settings)

        # Le script ne doit pas être exécutable
        assert "<script>" not in html_result
        assert "&lt;script&gt;" in html_result or "alert" not in html_result

    def test_inject_signature_multiline(self):
        """Test que les sauts de ligne sont convertis en <br>."""
        html = "<html><body><p>Test</p></body></html>"
        billing_settings = create_mock_billing_settings(
            mode="text", signature_text="Ligne 1\nLigne 2\nLigne 3"
        )
        html_result, _ = inject_signature_into_html(html, billing_settings=billing_settings)

        assert "Ligne 1" in html_result
        assert "Ligne 2" in html_result
        assert "Ligne 3" in html_result
        # Vérifier qu'il y a des <br> (les \n sont convertis)
        assert "<br>" in html_result

    def test_inject_signature_real_world_example(self):
        """Test avec un exemple réaliste de signature."""
        html = """
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6;">
            <p>Bonjour Client,</p>
            <p>Veuillez trouver ci-joint la facture <strong>INV-001</strong>.</p>
            <p>Cordialement,<br><strong>Company</strong></p>
        </body>
        </html>
        """
        signature_text = (
            "Khalid ALAOUI\n"
            "Associé gérant – Emmenez-moi Sàrl\n"
            "022 512 02 03 | 079 291 50 37\n"
            "info@casa-famiglia.ch"
        )
        billing_settings = create_mock_billing_settings(
            mode="text", signature_text=signature_text
        )
        html_result, _ = inject_signature_into_html(html, billing_settings=billing_settings)

        # Vérifier que le contenu original est intact
        assert "Bonjour Client" in html_result
        assert "INV-001" in html_result
        # Vérifier que la signature est présente
        assert "Khalid ALAOUI" in html_result
        assert "Associé gérant" in html_result
        assert "022 512 02 03" in html_result
        assert "info@casa-famiglia.ch" in html_result
        # Vérifier le séparateur
        assert "—" in html_result
        # Vérifier la structure HTML
        assert "</body>" in html_result
        assert "</html>" in html_result

    def test_inject_signature_no_billing_settings(self):
        """Test que sans billing_settings, retourne html inchangé."""
        html = "<html><body><p>Test</p></body></html>"
        html_result, logo_info = inject_signature_into_html(html, billing_settings=None)
        assert html_result == html
        assert logo_info is None
