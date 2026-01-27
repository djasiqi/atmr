"""Tests pour le fallback SVG dans la signature email."""

import pytest

from services.email.signature_utils import inject_signature_into_html


@pytest.fixture
def mock_company_svg():
    """Mock d'une entreprise avec logo SVG."""
    company = type("Company", (), {})()
    company.name = "Test Company"
    company.logo_url = "/uploads/company_logos/logo.svg"
    return company


@pytest.fixture
def mock_billing_settings():
    """Mock de billing settings avec signature form."""
    settings = type("BillingSettings", (), {})()
    settings.email_signature_mode = "form"
    settings.signature_name = "Test Name"
    settings.signature_email = "test@example.com"
    return settings


class TestEmailSignatureSvgFallback:
    """Tests pour le fallback SVG vers URL absolue."""

    def test_svg_logo_fallback_to_url(self, mock_company_svg, mock_billing_settings, monkeypatch):
        """Test que logo SVG déclenche fallback vers URL absolue (pas de CID)."""
        # Mock _get_logo_bytes pour retourner SVG
        def mock_get_logo_bytes(logo_url):
            if logo_url.endswith(".svg"):
                svg_content = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
                return (svg_content, "image/svg+xml")
            return (None, None)

        monkeypatch.setattr(
            "services.email.signature_utils._get_logo_bytes",
            mock_get_logo_bytes,
        )
        monkeypatch.setattr(
            "services.email.signature_utils._make_logo_url_absolute",
            lambda _: "https://example.com/uploads/company_logos/logo.svg",
        )

        html = "<html><body><p>Test</p></body></html>"
        html_result, logo_info = inject_signature_into_html(
            html, company=mock_company_svg, billing_settings=mock_billing_settings
        )

        # Assert: SVG doit déclencher fallback (pas de CID)
        assert logo_info is None  # Pas d'attachement inline pour SVG
        # Le HTML doit contenir l'URL absolue, pas cid:company_logo
        assert "cid:company_logo" not in html_result
        assert "https://example.com/uploads/company_logos/logo.svg" in html_result

    def test_png_logo_uses_cid(self, mock_billing_settings, monkeypatch):
        """Test que logo PNG utilise CID inline (pas de fallback)."""
        # Mock company avec logo PNG
        company = type("Company", (), {})()
        company.name = "Test Company"
        company.logo_url = "/uploads/company_logos/logo.png"

        def mock_get_logo_bytes(logo_url):
            if logo_url.endswith(".png"):
                png_content = b"\x89PNG\r\n\x1a\n"
                return (png_content, "image/png")
            return (None, None)

        monkeypatch.setattr(
            "services.email.signature_utils._get_logo_bytes",
            mock_get_logo_bytes,
        )

        html = "<html><body><p>Test</p></body></html>"
        html_result, logo_info = inject_signature_into_html(
            html, company=company, billing_settings=mock_billing_settings
        )

        # Assert: PNG doit utiliser CID inline
        assert logo_info is not None
        assert logo_info["cid"] == "company_logo"  # CID strict
        assert logo_info["mime_type"] == "image/png"
        # Le HTML doit contenir cid:company_logo
        assert "cid:company_logo" in html_result
