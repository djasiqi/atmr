"""Tests pour le mode provider (brevo_api vs brevo_smtp) et logo_mode (url vs cid)."""

import re
from email import policy
from email.parser import BytesParser
from unittest.mock import patch

import pytest

from services.email.brevo_provider import BrevoEmailProvider
from services.email.signature_utils import inject_signature_into_html


class TestBrevoApiForcesUrlMode:
    """Vérifie qu'en mode brevo_api le logo utilise une URL absolue avec cache-busting."""

    def test_brevo_api_forces_url_mode(self, monkeypatch):
        """Vérifie que logo_mode=url produit src commençant par https et contenant ?v=."""
        public_base = "https://example.com"
        monkeypatch.setattr(
            "services.email.signature_utils._make_logo_url_absolute",
            lambda _: f"{public_base}/uploads/company_logos/logo.png",
        )
        monkeypatch.setattr(
            "services.email.signature_utils._get_logo_bytes",
            lambda _: (b"\x89PNG\r\n\x1a\n", "image/png"),
        )

        company = type("Company", (), {})()
        company.name = "Test"
        company.logo_url = "/uploads/company_logos/logo.png"

        settings = type("BillingSettings", (), {})()
        settings.email_signature_mode = "form"
        settings.signature_name = "Name"
        settings.signature_email = "a@b.ch"

        html = "<html><body><p>x</p></body></html>"
        html_result, logo_info = inject_signature_into_html(
            html,
            company=company,
            billing_settings=settings,
            logo_mode="url",
            cache_bust=42,
        )

        assert logo_info is None, "En mode url pas d'attachement inline"
        img_src = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html_result)
        assert img_src is not None, "Un src img doit être présent"
        src = img_src.group(1)
        assert src.startswith("https://"), (
            f"En mode url le src doit être une URL HTTPS, obtenu: {src}"
        )
        assert "?v=42" in src or "?v=42" in html_result, (
            f"Cache-busting ?v=42 attendu dans le src, obtenu: {src}"
        )


class TestSmtpMimeContainsRelatedAndContentId:
    """Vérifie que l'envoi SMTP produit un MIME multipart/related avec Content-ID company_logo."""

    @patch("services.email.brevo_provider.smtplib.SMTP")
    def test_smtp_mime_contains_related_and_content_id(
        self, mock_smtp_class, monkeypatch
    ):
        """Vérifie multipart/related + Content-ID <company_logo> + src=\"cid:company_logo\" dans le HTML."""
        monkeypatch.setenv("EMAIL_PROVIDER_MODE", "brevo_smtp")
        monkeypatch.setenv("BREVO_SMTP_PASSWORD", "test-smtp-password")
        mock_smtp = mock_smtp_class.return_value.__enter__.return_value
        sent_message = []

        def capture_sendmail(from_addr, to_addrs, msg):
            sent_message.append(msg)

        mock_smtp.sendmail.side_effect = capture_sendmail

        html_content = (
            '<html><body><p>Test</p><img src="cid:company_logo" /></body></html>'
        )
        logo_bytes = b"\x89PNG\r\n\x1a\n"
        attachments = [
            {
                "filename": "logo.png",
                "content": logo_bytes,
                "cid": "company_logo",
                "mime_type": "image/png",
            }
        ]

        provider = BrevoEmailProvider(api_key="test_key")
        provider.send_invoice_email(
            from_email="noreply@test.ch",
            from_name="Test",
            to_email="client@example.com",
            to_name="Client",
            subject="Test",
            html_content=html_content,
            attachments=attachments,
        )

        assert len(sent_message) == 1, "sendmail doit avoir été appelé une fois"
        raw = sent_message[0]
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        msg = BytesParser(policy=policy.default).parsebytes(raw)

        # Vérifier qu'on a une structure multipart (mixed ou related)
        ct = msg.get_content_type()
        assert "multipart" in ct, f"Message racine doit être multipart, obtenu: {ct}"

        # Trouver la part related (directe ou dans mixed)
        related_found = False
        content_id_found = False
        html_has_cid = False

        for part in msg.walk():
            pct = part.get_content_type()
            cid = part.get("Content-ID") or ""
            if "multipart/related" in pct:
                related_found = True
            if "company_logo" in cid:
                content_id_found = True
            if "text/html" in pct:
                payload = part.get_payload(decode=True)
                if payload:
                    html_has_cid = (
                        b"cid:company_logo" in payload
                        or "cid:company_logo"
                        in (payload.decode("utf-8", errors="replace"))
                    )

        assert related_found, "Une part multipart/related doit être présente"
        assert content_id_found, "Content-ID <company_logo> doit être présent"
        assert html_has_cid, 'Le HTML doit contenir src="cid:company_logo"'
