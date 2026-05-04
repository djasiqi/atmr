"""Tests pour vérifier que le CID HTML correspond au contentId Brevo."""

import pytest

from services.email.signature_utils import generate_signature_html_from_form


class TestEmailSignatureCidMatch:
    """Tests pour vérifier la correspondance CID strict."""

    def test_cid_html_matches_brevo_contentid(self):
        """Test que le HTML src="cid:company_logo" correspond exactement à contentId="company_logo"."""

        # Créer un mock company avec logo_url
        class MockCompany:
            def __init__(self):
                self.logo_url = "https://example.com/logo.png"
                self.name = "Test Company"

        mock_company = MockCompany()
        html_result = generate_signature_html_from_form(
            name="Test",
            company_obj=mock_company,
        )

        # Vérifier que le HTML contient exactement src="cid:company_logo"
        import re

        img_src_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_result)
        assert img_src_match is not None, "Aucun src trouvé dans le HTML"
        html_img_src = img_src_match.group(1)

        # Le src doit être exactement "cid:company_logo" (sans chevrons)
        assert html_img_src == "cid:company_logo", (
            f"HTML src={html_img_src} ne correspond pas à 'cid:company_logo'. "
            "Le contentId Brevo doit être exactement 'company_logo' (sans chevrons)."
        )

        # Vérifier qu'il n'y a pas de chevrons dans le CID
        msg_cid = "Le CID ne doit pas contenir de chevrons < >"
        assert "<" not in html_img_src, msg_cid
        assert ">" not in html_img_src, msg_cid
